"""
MQTT client wrapper for BBC Stories.

Single topic: bbc/stories
- Cron publishes story JSON to bbc/stories
- Both bots subscribe to bbc/stories (broker fans out to all subscribers)

Broker: eclipse-mosquitto on localhost:1883 (no auth).
"""

import json
import logging
import os
import threading
import time
from collections.abc import Callable

from paho.mqtt.client import Client
from paho.mqtt.enums import CallbackAPIVersion

logger = logging.getLogger(__name__)

MQTT_BROKER = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
STORY_TOPIC = "bbc/stories"
QOS = 1  # at-least-once delivery


def _on_connect(client, userdata, flags, rc, properties=None) -> None:
    if rc == 0:
        logger.info("[mqtt] Connected to %s", MQTT_BROKER)
    else:
        logger.warning("[mqtt] Connect failed with rc=%d", rc)


def _on_disconnect(client, userdata, rc, properties=None) -> None:
    logger.warning("[mqtt] Disconnected (rc=%d). Reconnecting...", rc)


def _on_message(client, userdata, msg) -> None:
    try:
        payload = json.loads(msg.payload.decode())
        callback = userdata["callback"]
        callback(payload)
    except Exception as e:
        logger.error("[mqtt] Failed to handle message: %s", e, exc_info=True)


class MQTTPublisher:
    """
    Thread-safe MQTT publisher. Connects on first publish and stays connected.
    """

    def __init__(self) -> None:
        self._client: Client | None = None
        self._lock = threading.Lock()

    def publish(self, payload: dict) -> None:
        """
        Publish a story payload to bbc/stories.
        Creates the client on first call (lazy connect).
        """
        client: Client
        with self._lock:
            if self._client is None:
                client = Client(CallbackAPIVersion.VERSION2)
                client.on_connect = _on_connect
                client.on_disconnect = _on_disconnect
                self._client = client
            else:
                client = self._client

        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()

        story_json = json.dumps(payload)
        result = client.publish(STORY_TOPIC, story_json, qos=QOS)
        result.wait_for_publish(timeout=5.0)
        logger.info(
            "[mqtt] Published story to %s: %s", STORY_TOPIC, payload.get("headline", "?")[:60]
        )

    def stop(self) -> None:
        """Gracefully stop the publisher."""
        with self._lock:
            if self._client:
                self._client.loop_stop()
                self._client.disconnect()
                self._client = None


class MQTTSubscriber:
    """
    MQTT subscriber that calls a callback for each story payload received.
    Runs in a background thread, auto-reconnects on disconnect.

    Pass a stable client_id to enable persistent sessions (QoS 1 messages
    queued by the broker while the subscriber is offline are delivered on
    reconnect). Without a client_id the session is ephemeral and messages
    published while disconnected are lost.
    """

    def __init__(self, callback: Callable[[dict], None], client_id: str = "") -> None:
        self._callback = callback
        self._client_id = client_id
        self._client: Client | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _create_client(self) -> Client:
        # clean_session=False requires a stable client_id so the broker can
        # match the reconnecting client to its durable session.
        persistent = bool(self._client_id)
        client = Client(
            CallbackAPIVersion.VERSION2,
            client_id=self._client_id or "",
            clean_session=not persistent,
            userdata={"callback": self._callback},
        )
        client.on_connect = _on_connect
        client.on_disconnect = _on_disconnect
        client.on_message = _on_message
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.subscribe(STORY_TOPIC, qos=QOS)
        return client

    def start(self) -> None:
        """Start the subscriber in a background thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[mqtt] Subscriber started, listening on %s", STORY_TOPIC)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                client = self._create_client()
                self._client = client
                client.loop_forever()
            except Exception as e:
                logger.error("[mqtt] Subscriber error: %s", e, exc_info=True)
                if not self._stop_event.is_set():
                    time.sleep(5)  # back off before reconnecting

    def stop(self) -> None:
        """Stop the subscriber gracefully."""
        self._stop_event.set()
        if self._client:
            self._client.disconnect()
            self._client = None
        if self._thread:
            self._thread.join(timeout=5)


# ── Backward-compatible shim (notifier.py still imports from here) ────────────


_publisher: MQTTPublisher | None = None


def write_to_queue(story_payload: dict, platform: str) -> None:  # noqa: ARG001
    """
    Publish a story to the MQTT topic. platform arg is ignored (single topic fans out).
    Kept for backward compatibility with notifier.py.
    """
    global _publisher
    if _publisher is None:
        _publisher = MQTTPublisher()
    _publisher.publish(story_payload)


def stop() -> None:
    global _publisher
    if _publisher:
        _publisher.stop()
        _publisher = None
