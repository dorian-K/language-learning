"""
Abstract base for platform adapters (Discord, Telegram, etc.).
Each adapter implements posting/reaction logic for its platform.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class StoryPayload:
    """Platform-agnostic story content ready to be posted."""
    headline: str        # Formatted: emoji + bold title
    summary: str         # B1-adapted article summary
    bullets: str         # B1 bullet points
    url: str             # Original article URL
    topic_title: str     # Thread/topic subject line


class PlatformAdapter(ABC):
    """
    Interface for platform-specific posting.

    Subclasses implement:
    - post_channel:    Send initial message to a channel/forum
    - create_thread:   Open a thread/topic on the given channel message
    - post_thread:     Send the simplified article to the thread/topic
    - add_reaction:    React to the channel message (optional)

    Convenience method:
    - send_story:      Full flow; calls post_channel → add_reaction →
                       create_thread → post_thread → mark_sent
    """

    @abstractmethod
    async def post_channel(self, payload: StoryPayload) -> str:
        """
        Post the headline to the channel/forum.
        Returns the platform message ID of the posted headline.
        """
        ...

    @abstractmethod
    async def create_thread(self, payload: StoryPayload, channel_msg_id: str) -> str:
        """
        Open a thread/topic on the given channel message.
        Returns the thread/topic ID.
        """
        ...

    @abstractmethod
    async def post_thread(self, thread_id: str, payload: StoryPayload) -> None:
        """Post the simplified article to the thread/topic."""
        ...

    @abstractmethod
    async def add_reaction(self, channel_msg_id: str) -> None:
        """React to the channel message (e.g. ✅ to acknowledge)."""
        ...

    # ── Convenience ────────────────────────────────────────────────────────

    async def send_story(self, payload: StoryPayload) -> None:
        """
        Full flow: post headline → react → open thread → post article → mark sent.
        Platforms that don't support threads can override to just post_channel.
        """
        msg_id = await self.post_channel(payload)
        await self.add_reaction(msg_id)
        thread_id = await self.create_thread(payload, msg_id)
        await self.post_thread(thread_id, payload)
        self.mark_sent(payload.url)
        logger.info("[%s] Story sent: %s", self.__class__.__name__, payload.headline[:60])

    # ── Sent-stories tracking ────────────────────────────────────────────

    def story_is_sent(self, url: str) -> bool:
        """Check if story URL is already tracked as sent."""
        from ..queue_service import queue_service
        return queue_service.is_sent(url)

    def mark_sent(self, url: str) -> None:
        """Record a URL as sent to prevent re-sending."""
        from ..queue_service import queue_service
        queue_service.mark_sent(url)
