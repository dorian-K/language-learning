"""
Discord adapter — implements PlatformAdapter for Discord.
"""

import logging
import os
import discord
from discord import app_commands

from .base import PlatformAdapter, StoryPayload


logger = logging.getLogger(__name__)

# Forum channel IDs
STORIES_CHANNEL_ID = int(os.getenv("DISCORD_STORIES_CHANNEL_ID", "0"))


def _make_thread_name(title: str) -> str:
    """Sanitise a story title into a valid thread name."""
    name = title.replace("**", "").replace("*", "").strip()
    return name[:100]


class DiscordAdapter(PlatformAdapter):
    """Discord-specific posting logic."""

    def __init__(self, client: discord.Client):
        self.client = client

    # ── PlatformAdapter interface ─────────────────────────────────────────

    async def post_channel(self, payload: StoryPayload) -> str:
        """
        Post headline to the stories forum channel.
        Returns the message ID.
        """
        channel = self.client.get_channel(STORIES_CHANNEL_ID)
        if not isinstance(channel, discord.TextChannel):
            raise RuntimeError(f"Channel {STORIES_CHANNEL_ID} not found or not a TextChannel")

        msg = await channel.send(payload.headline)
        logger.info("[discord] Posted headline to channel %s: %s", STORIES_CHANNEL_ID, payload.headline[:60])
        return str(msg.id)

    async def create_thread(self, payload: StoryPayload, channel_msg_id: str) -> str:
        """
        Create a public thread on the channel message for discussion.
        Returns the thread ID.
        """
        channel = self.client.get_channel(STORIES_CHANNEL_ID)
        if not isinstance(channel, discord.TextChannel):
            raise RuntimeError(f"Channel {STORIES_CHANNEL_ID} not found or not a TextChannel")

        try:
            message = await channel.fetch_message(int(channel_msg_id))
        except discord.NotFound:
            raise RuntimeError(f"Message {channel_msg_id} not found in channel {STORIES_CHANNEL_ID}")

        thread_name = _make_thread_name(payload.topic_title)
        thread = await message.create_thread(
            name=thread_name,
            invitable=False,  # public thread
        )
        logger.info("[discord] Created thread '%s' (id=%s)", thread_name, thread.id)
        return str(thread.id)

    async def post_thread(self, thread_id: str, payload: StoryPayload) -> None:
        """Post the simplified article + original link to the thread."""
        thread = self.client.get_channel(int(thread_id))
        if thread is None:
            raise RuntimeError(f"Thread {thread_id} not found")

        content = (
            f"> {payload.summary}\n\n"
            f"{payload.bullets}\n\n"
            f"🔗 [Artículo original]({payload.url})"
        )
        await thread.send(content)
        logger.info("[discord] Posted article to thread %s", thread_id)

    async def add_reaction(self, channel_msg_id: str) -> None:
        """Add a checkmark reaction to the channel message."""
        channel = self.client.get_channel(STORIES_CHANNEL_ID)
        if not isinstance(channel, discord.TextChannel):
            return
        try:
            message = await channel.fetch_message(int(channel_msg_id))
            await message.add_reaction("✅")
        except Exception as e:
            logger.warning("[discord] Could not add reaction to %s: %s", channel_msg_id, e)

    # ── Full flow ────────────────────────────────────────────────────────

    async def send_story(self, payload: StoryPayload) -> None:
        """
        Full flow: post headline → react → open thread → post article → mark sent.
        """
        msg_id = await self.post_channel(payload)
        await self.add_reaction(msg_id)
        thread_id = await self.create_thread(payload, msg_id)
        await self.post_thread(thread_id, payload)
        self.mark_sent(payload.url)
        logger.info("[discord] Story sent: %s", payload.headline[:60])

    # ── Sent-stories tracking ────────────────────────────────────────────

    def story_is_sent(self, url: str) -> bool:
        from .. import queue_service
        return queue_service.queue_service.is_sent(url)

    def mark_sent(self, url: str) -> None:
        from .. import queue_service
        queue_service.queue_service.mark_sent(url)
