"""
Sent stories tracker — records article URLs that have been sent to webhook.

File format: one URL per line, in data/sent_stories.txt
"""

from pathlib import Path

TRACKER_FILE = Path(__file__).parent.parent.parent / "data" / "sent_stories.txt"


def _ensure_dir() -> None:
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_sent_urls() -> set[str]:
    """Return all URLs that have been sent already."""
    if not TRACKER_FILE.exists():
        return set()
    with open(TRACKER_FILE, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_sent(url: str) -> None:
    """Append a URL to the tracker file."""
    _ensure_dir()
    with open(TRACKER_FILE, "a", encoding="utf-8") as f:
        f.write(url.strip() + "\n")


def is_sent(url: str) -> bool:
    """Check if a URL has been sent already."""
    return url.strip() in get_sent_urls()


def filter_unsent(urls: list[str]) -> list[str]:
    """Return only URLs that haven't been sent yet."""
    sent = get_sent_urls()
    return [u for u in urls if u.strip() not in sent]
