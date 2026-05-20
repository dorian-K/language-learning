"""
Shared queue for cron → bot communication.
Both containers mount the same volume and read/write this file.
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

QUEUE_PATH = Path(os.getenv("SHARED_QUEUE_PATH", "/app/shared/queue.json"))


def _load() -> dict:
    if not QUEUE_PATH.exists():
        return {"pending": [], "sent": []}
    try:
        with open(QUEUE_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[queue] Failed to read queue: %s", e)
        return {"pending": [], "sent": []}


def _save(data: dict) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def enqueue_story(story: dict) -> None:
    """Add a story to the pending queue (called by cron after webhook send).

    Accepts a StoryPayload dataclass (passed as dict via dataclasses.asdict),
    or a raw RSS story dict. Normalises field names so pop_story always returns
    a StoryPayload-compatible dict.
    """
    import dataclasses

    data = _load()

    # Accept both dict and dataclass instances
    if dataclasses.is_dataclass(story):
        story = dataclasses.asdict(story)

    url = story.get("url") or story.get("link") or ""
    title = story.get("headline") or story.get("title") or ""

    if is_already_queued(url):
        return

    entry = {
        "title": title,
        "link": url,
        "headline": story.get("headline", title),
        "summary": story.get("summary", ""),
        "bullets": story.get("bullets", ""),
        "text": story.get("text", ""),
        "url": url,
        "topic_title": story.get("topic_title", title),
        "source": story.get("source", ""),
        "pub_date": story.get("pub_date", ""),
        "queued_at": datetime.now(UTC).isoformat(),
    }
    data["pending"].append(entry)
    _save(data)
    logger.info("[queue] Enqueued story: %s", title)


def pop_story() -> dict | None:
    """Pop the oldest pending story (called by bot when user clicks button)."""
    data = _load()
    if not data["pending"]:
        return None
    story = data["pending"].pop(0)
    data["sent"].append({**story, "dequeued_at": datetime.now(UTC).isoformat()})
    _save(data)
    logger.info("[queue] Dequeued story: %s", story.get("title", "?"))
    return story


def peek_pending() -> list[dict]:
    """Return all pending stories without removing them."""
    return _load().get("pending", [])


def pending_count() -> int:
    """Return number of pending stories."""
    return len(peek_pending())


def is_already_queued(url: str) -> bool:
    """Check if a story URL is already in pending or sent.

    Stories from RSS use 'link' as the URL key.
    """
    if not url:
        return False
    data = _load()
    for s in data["pending"] + data["sent"]:
        if s.get("link") == url or s.get("url") == url:
            return True
    return False
