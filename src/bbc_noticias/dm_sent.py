"""
Per-user DM sent-story tracker.

Stores which story URLs have already been sent to each Telegram user so that
each on-demand request delivers a different story per person.

Format: JSON file at data/dm_sent.json → { "user_id": ["url1", "url2", ...] }
Separate from sent_stories.py (channel tracking) — the same story can appear
in both the daily channel post and a user's DM.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DM_TRACKER_FILE = Path(__file__).parent.parent.parent / "data" / "dm_sent.json"


def _load() -> dict[str, list[str]]:
    if not DM_TRACKER_FILE.exists():
        return {}
    try:
        with open(DM_TRACKER_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[dm_sent] Failed to read tracker: %s", e)
        return {}


def _save(data: dict[str, list[str]]) -> None:
    DM_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(DM_TRACKER_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, DM_TRACKER_FILE)


def get_sent_urls(user_id: int) -> set[str]:
    """Return all story URLs already sent to this user."""
    return set(_load().get(str(user_id), []))


def mark_sent(user_id: int, url: str) -> None:
    """Record that a story URL was sent to this user."""
    data = _load()
    key = str(user_id)
    if key not in data:
        data[key] = []
    if url not in data[key]:
        data[key].append(url)
        _save(data)


def filter_unsent(user_id: int, urls: list[str]) -> list[str]:
    """Return only URLs this user has not yet received."""
    sent = get_sent_urls(user_id)
    return [u for u in urls if u not in sent]
