"""
Queue service — unified interface over queue.py and sent_stories.py.

Provides:
- Story queuing (cron → bot via shared volume)
- Sent-story tracking (both in-memory set and persisted file)
- Convenience: mark_sent + is_sent
"""

import logging
from typing import Optional

from . import queue as _queue
from . import sent_stories as _sent_stories


logger = logging.getLogger(__name__)


class QueueService:
    """
    Facade over the shared queue file (cron → bot) and the sent-stories tracker.

    The queue is a JSON file in a Docker volume shared between bbc-cron and bbc-bot.
    The sent-stories tracker is a plain text file in the data volume.
    """

    # ── Queue (cron → bot) ────────────────────────────────────────────────

    def enqueue(self, story: dict) -> None:
        """Add a story to the pending queue (called by cron after webhook send)."""
        _queue.enqueue_story(story)

    def pop_queued(self) -> Optional[dict]:
        """Pop the oldest pending story (called by bot when user clicks)."""
        return _queue.pop_story()

    def peek_queued(self) -> list[dict]:
        """Return all pending stories without removing them."""
        return _queue.peek_pending()

    def queued_count(self) -> int:
        return _queue.pending_count()

    def is_queued(self, url: str) -> bool:
        return _queue.is_already_queued(url)

    # ── Sent stories tracker ───────────────────────────────────────────────

    def is_sent(self, url: str) -> bool:
        """True if story URL was already sent (from queue or button)."""
        return _sent_stories.is_sent(url)

    def mark_sent(self, url: str) -> None:
        """Record a URL as sent to prevent re-sending."""
        _sent_stories.mark_sent(url)
        logger.info("[queue] Marked sent: %s", url)

    def filter_unsent(self, urls: list[str]) -> list[str]:
        """Return only URLs that haven't been sent yet."""
        return _sent_stories.filter_unsent(urls)


# Singleton instance
queue_service = QueueService()
