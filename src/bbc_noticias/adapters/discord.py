"""
Discord adapter — implements PlatformAdapter for Discord.
"""

import asyncio
import logging
import os
import threading
import time

import discord

from .base import PlatformAdapter, StoryPayload
from .. import pubsub

logger = logging.getLogger(__name__)

# Forum channel IDs
STORIES_CHANNEL_ID = int(os.getenv("DISCORD_STORIES_CHANNEL_ID", "0"))


def _make_thread_name(title: str) -> str:
    """Sanitise a story title into a valid thread name."""
    name = title.replace("**", "").replace("*", "").strip()
    return name[:100]


class DiscordAdapter(PlatformAdapter):
    """Discord-specific posting logic."""

    def __init__(self, client: discord.Client):  # type: ignore[reportAttributeAccessIssue]
        self.client = client

    # ── PlatformAdapter interface ─────────────────────────────────────────

    async def post_channel(self, payload: StoryPayload, channel_override=None) -> str:
        """
        Post headline to the stories forum channel (or override channel).
        Returns the message ID.
        """
        channel = (
            channel_override
            or self.client.get_channel(STORIES_CHANNEL_ID)
            or self.client.fetch_channel(STORIES_CHANNEL_ID)
        )
        if not isinstance(channel, discord.TextChannel):  # type: ignore[reportAttributeAccessIssue]
            raise RuntimeError(
                f"Channel {getattr(channel, 'id', '?')} not found or not a TextChannel"
            )

        msg = await channel.send(payload.headline)
        logger.info(
            "[discord] Posted headline to channel %s: %s",
            getattr(channel, "id", STORIES_CHANNEL_ID),
            payload.headline[:60],
        )
        return str(msg.id)

    async def create_thread(
        self, payload: StoryPayload, channel_msg_id: str, channel_override=None
    ) -> str:
        """
        Create a public thread on the channel message for discussion.
        Returns the thread ID.
        """
        channel = channel_override or self.client.get_channel(STORIES_CHANNEL_ID)
        if not isinstance(channel, discord.TextChannel):  # type: ignore[reportAttributeAccessIssue]
            raise RuntimeError(
                f"Channel {getattr(channel, 'id', '?')} not found or not a TextChannel"
            )

        try:
            message = await channel.fetch_message(int(channel_msg_id))
        except discord.NotFound:  # type: ignore[reportAttributeAccessIssue]
            raise RuntimeError(
                f"Message {channel_msg_id} not found in channel {getattr(channel, 'id', '?')}"
            ) from None

        thread_name = _make_thread_name(payload.topic_title)
        thread = await message.create_thread(name=thread_name)
        logger.info("[discord] Created thread '%s' (id=%s)", thread_name, thread.id)
        return str(thread.id)

    async def post_thread(self, thread_id: str, payload: StoryPayload) -> None:
        """Post the simplified article + original link to the thread."""
        thread = self.client.get_channel(int(thread_id))
        if thread is None:
            raise RuntimeError(f"Thread {thread_id} not found")
        if not isinstance(thread, discord.Thread):
            raise RuntimeError(f"Channel {thread_id} is not a thread")

        # Build content, splitting at Discord's 2000-char limit
        text = payload.text
        chunk_size = 1900
        if len(text) <= chunk_size:
            content = f"> {payload.summary}\n\n{payload.bullets}\n\n---\n\n{text}\n\n---\n\n🔗 [Artículo original]({payload.url})"
            await thread.send(content)  # type: ignore[reportAttributeAccessIssue]
        else:
            # Send summary + bullets first, then the article in chunks
            await thread.send(f"> {payload.summary}\n\n{payload.bullets}")  # type: ignore[reportAttributeAccessIssue]
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                await thread.send(chunk)  # type: ignore[reportAttributeAccessIssue]
            await thread.send(f"\n🔗 [Artículo original]({payload.url})")  # type: ignore[reportAttributeAccessIssue]

        logger.info("[discord] Posted article to thread %s", thread_id)

    async def add_reaction(self, channel_msg_id: str, channel_override=None) -> None:
        """Add a checkmark reaction to the channel message."""
        channel = channel_override or self.client.get_channel(STORIES_CHANNEL_ID)
        if not isinstance(channel, discord.TextChannel):  # type: ignore[reportAttributeAccessIssue]
            return
        try:
            message = await channel.fetch_message(int(channel_msg_id))
            await message.add_reaction("✅")
        except Exception as e:
            logger.warning(
                "[discord] Could not add reaction to %s: %s", channel_msg_id, e, exc_info=True
            )

    # ── Full flow ────────────────────────────────────────────────────────

    async def send_story(self, payload: StoryPayload, interaction_channel=None) -> None:
        """
        Full flow: post headline → react → open thread → post article → mark sent.

        If interaction_channel is provided (user-initiated command), also send a reply
        to that channel confirming publication.
        """
        channel_override = interaction_channel  # shorthand

        msg_id = await self.post_channel(payload, channel_override)
        await self.add_reaction(msg_id, channel_override)
        thread_id = await self.create_thread(payload, msg_id, channel_override)
        await self.post_thread(thread_id, payload)
        self.mark_sent(payload.url)
        logger.info("[discord] Story sent: %s", payload.headline[:60])

    # ── Sent-stories tracking ────────────────────────────────────────────

    def story_is_sent(self, url: str) -> bool:
        from ..queue_service import queue_service as _qs

        return _qs.is_sent(url)

    def mark_sent(self, url: str) -> None:
        from ..queue_service import queue_service as _qs

        _qs.mark_sent(url)

    # ── Queue subscriber (pub/sub) ───────────────────────────────────────────

    @staticmethod
    async def consume_and_send() -> int:
        """
        Atomically consume all pending stories for Discord from the queue
        and send them. Returns the number of stories sent.
        """
        entries = pubsub.consume_stories_for("discord")
        if not entries:
            return 0

        sent = 0
        for entry in entries:
            try:
                payload = StoryPayload(**entry["story"])
                # Use a fresh client session for sending queued stories
                # (self.client is the bot's connection; send via webhooks)
                await DiscordAdapter._send_payload_via_client(payload)
                sent += 1
            except Exception as e:
                logger.error("[discord] Failed to send queued story: %s", e, exc_info=True)

        return sent

    @staticmethod
    async def _send_payload_via_client(payload: StoryPayload) -> None:
        """
        Send a story payload using the shared DiscordAdapter client.
        Finds the client from the module-level client variable.
        """
        # Import the module-level client set up in discord_bot.py
        from .. import discord_bot

        adapter = discord_bot.client.adapter
        await adapter.send_story(payload)

    def start_subscriber(self) -> None:
        """
        Poll the queue every 10 seconds and send pending Discord stories.
        Runs in a background daemon thread.
        """

        def poll():
            while True:
                try:
                    asyncio.run(self.consume_and_send())
                except Exception as e:
                    logger.error("[discord] Queue subscriber error: %s", e, exc_info=True)
                time.sleep(10)

        t = threading.Thread(target=poll, daemon=True)
        t.start()
        logger.info("[discord] Queue subscriber started")
