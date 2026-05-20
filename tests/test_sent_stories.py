"""
Unit tests for sent_stories.py — tracks which article URLs have been sent.
"""

import tempfile
from pathlib import Path

import pytest

from src.bbc_noticias.sent_stories import (
    filter_unsent,
    get_sent_urls,
    is_sent,
    mark_sent,
)

# Use a temp file for isolation
_TEMP_FILE = Path(tempfile.gettempdir()) / "test_sent_stories.txt"


@pytest.fixture(autouse=True)
def clean_temp_file():
    if _TEMP_FILE.exists():
        _TEMP_FILE.unlink()
    # Patch TRACKER_FILE to use our temp path
    import src.bbc_noticias.sent_stories as tracker

    original = tracker.TRACKER_FILE
    tracker.TRACKER_FILE = _TEMP_FILE
    yield
    tracker.TRACKER_FILE = original
    if _TEMP_FILE.exists():
        _TEMP_FILE.unlink()


class TestSentStories:
    def test_get_sent_urls_empty_file(self):
        assert get_sent_urls() == set()

    def test_mark_sent_append_url(self):
        mark_sent("https://bbc.com/story1")
        assert get_sent_urls() == {"https://bbc.com/story1"}

    def test_mark_sent_multiple_urls(self):
        mark_sent("https://bbc.com/story1")
        mark_sent("https://bbc.com/story2")
        assert get_sent_urls() == {"https://bbc.com/story1", "https://bbc.com/story2"}

    def test_mark_sent_trims_whitespace(self):
        mark_sent("  https://bbc.com/story1  \n")
        assert "https://bbc.com/story1" in get_sent_urls()

    def test_is_sent(self):
        mark_sent("https://bbc.com/story1")
        assert is_sent("https://bbc.com/story1") is True
        assert is_sent("https://bbc.com/story2") is False

    def test_is_sent_whitespace(self):
        mark_sent("  https://bbc.com/story1  ")
        assert is_sent("https://bbc.com/story1") is True

    def test_filter_unsent(self):
        sent = ["https://bbc.com/story1", "https://bbc.com/story2"]
        for url in sent:
            mark_sent(url)
        urls = [
            "https://bbc.com/story1",  # already sent
            "https://bbc.com/story2",  # already sent
            "https://bbc.com/story3",  # new
            "https://bbc.com/story4",  # new
        ]
        result = filter_unsent(urls)
        assert result == ["https://bbc.com/story3", "https://bbc.com/story4"]

    def test_filter_unsent_empty_list(self):
        assert filter_unsent([]) == []

    def test_mark_sent_idempotent(self):
        # Same URL marked twice should only appear once in file
        mark_sent("https://bbc.com/story1")
        mark_sent("https://bbc.com/story1")
        urls = get_sent_urls()
        assert list(urls).count("https://bbc.com/story1") == 1
