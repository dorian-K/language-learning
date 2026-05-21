"""Pub/sub via shared JSON file. Thread-safe single-writer."""

import json
import os
from typing import Literal

STORY_QUEUE_PATH = os.getenv("SHARED_QUEUE_PATH", "/app/shared/queue.json")


def read_queue() -> list[dict]:
    """Read all entries from the queue file."""
    if not os.path.exists(STORY_QUEUE_PATH):
        return []
    with open(STORY_QUEUE_PATH) as f:
        return json.load(f)


def clear_queue() -> None:
    """Clear all entries from the queue file."""
    with open(STORY_QUEUE_PATH, "w") as f:
        json.dump([], f)


def consume_stories_for(platform: Literal["discord", "telegram"]) -> list[dict]:
    """
    Atomically read and remove stories for the given platform from the queue.

    Reads all entries, filters those matching the platform (or "both"),
    then atomically rewrites the remaining entries back to the queue file.
    """
    queue = read_queue()
    ours, others = [], []
    for entry in queue:
        if entry.get("platform") in (platform, "both"):
            ours.append(entry)
        else:
            others.append(entry)
    # Atomic write: write to .tmp, then rename
    tmp = STORY_QUEUE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(others, f)
    os.replace(tmp, STORY_QUEUE_PATH)
    return ours