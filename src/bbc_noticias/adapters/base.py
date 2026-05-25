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

    headline: str  # Formatted: emoji + bold title
    summary: str  # B1-adapted article summary
    bullets: str  # B1 bullet points
    text: str  # Full simplified article text
    url: str  # Original article URL
    topic_title: str  # Thread/topic subject line


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
    async def post_channel(self, payload: StoryPayload, interaction_channel=None) -> str: ...

    @abstractmethod
    async def create_thread(
        self, payload: StoryPayload, channel_msg_id: str, interaction_channel=None
    ) -> str: ...

    @abstractmethod
    async def post_thread(self, thread_id: str, payload: StoryPayload) -> None: ...

    @abstractmethod
    async def add_reaction(self, channel_msg_id: str, interaction_channel=None) -> None: ...

    # ── Convenience ────────────────────────────────────────────────────────

    async def send_story(self, payload: StoryPayload, interaction_channel=None) -> None:
        """
        Full flow: post headline → react → open thread → post article → mark sent.

        interaction_channel: if provided, used as the target channel instead of the
        configured STORIES_CHANNEL_ID (useful for replying in the command-issued channel).
        """
        msg_id = await self.post_channel(payload, interaction_channel)
        await self.add_reaction(msg_id, interaction_channel)
        thread_id = await self.create_thread(payload, msg_id, interaction_channel)
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
