"""
Pub/sub via shared JSON file with durable subscriber semantics.

Each platform tracks consumed story IDs in its own section of the queue file.
When a story is published with "both", two separate entries are written
(discord + telegram) so each platform can independently consume and mark done.

Flow:
  - Publisher: write_to_queue(payload, "both") → writes TWO entries (discord + telegram)
  - Subscriber: consume_stories_for("discord") → reads all "discord" entries,
    marks them as consumed (doesn't delete), returns them for sending
"""

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

STORY_QUEUE_PATH = Path(os.getenv("SHARED_QUEUE_PATH", "/app/shared/queue.json"))


def _read_queue() -> dict:
    """Read the queue file, returning the full structure."""
    if not STORY_QUEUE_PATH.exists():
        return {"entries": [], "consumed": {}}
    try:
        with open(STORY_QUEUE_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[pubsub] Failed to read queue, starting fresh: %s", e)
        return {"entries": [], "consumed": {}}


def _write_queue(data: dict) -> None:
    """Atomically write the queue structure."""
    STORY_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(STORY_QUEUE_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, STORY_QUEUE_PATH)


def write_to_queue(story_payload: dict, platform: Literal["discord", "telegram", "both"]) -> None:
    """
    Write one or more entries to the queue for the target platform(s).

    When platform is "both", writes TWO entries so each bot consumes independently.
    """
    now = datetime.now(UTC).isoformat()
    story_id = story_payload.get("url", "")  # use URL as the unique story ID

    targets = ["discord", "telegram"] if platform == "both" else [platform]
    new_entries = [{"id": story_id, "platform": p, "story": story_payload, "published_at": now} for p in targets]

    data = _read_queue()
    data["entries"].extend(new_entries)
    _write_queue(data)
    logger.debug("[pubsub] Wrote %d entry/entries for platform=%s", len(new_entries), platform)


def consume_stories_for(platform: Literal["discord", "telegram"]) -> list[dict]:
    """
    Atomically read and return all unconsumed entries for the given platform,
    then mark them as consumed. Idempotent — safe to call multiple times.
    """
    data = _read_queue()
    consumed_key = f"consumed_{platform}"
    if consumed_key not in data:
        data[consumed_key] = []

    consumed_set = set(data[consumed_key])
    ours, remaining = [], []

    for entry in data["entries"]:
        if entry.get("platform") == platform and entry["id"] not in consumed_set:
            ours.append(entry)
            consumed_set.add(entry["id"])
        else:
            remaining.append(entry)

    data["entries"] = remaining
    data[consumed_key] = sorted(consumed_set)
    _write_queue(data)

    logger.debug("[pubsub] Consumed %d entries for %s (total consumed=%d)", len(ours), platform, len(consumed_set))
    return ours


def get_pending_count(platform: Literal["discord", "telegram"]) -> int:
    """Return how many unconsumed entries exist for the platform."""
    data = _read_queue()
    consumed_key = f"consumed_{platform}"
    consumed_set = set(data.get(consumed_key, []))
    return sum(1 for e in data["entries"] if e.get("platform") == platform and e["id"] not in consumed_set)


def clear_consumed(platform: Literal["discord", "telegram"]) -> None:
    """Clear the consumed tracking set for a platform (allows re-sending old stories)."""
    data = _read_queue()
    data[f"consumed_{platform}"] = []
    _write_queue(data)


def clear_all() -> None:
    """Clear all entries and consumed tracking."""
    _write_queue({"entries": [], "consumed": {}})