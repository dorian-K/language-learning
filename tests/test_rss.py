"""
Unit tests for rss.py — no API keys needed, tests RSS parsing logic.
"""

from unittest.mock import MagicMock

from src.bbc_noticias.rss import parse_rss_datetime


class TestParseRssDatetime:
    def test_parses_rfc822_format(self):
        result = parse_rss_datetime("Fri, 08 May 2026 12:00:00 GMT")
        assert result is not None
        assert result.year == 2026
        assert result.month == 5
        assert result.day == 8

    def test_parses_rfc822_no_timezone(self):
        result = parse_rss_datetime("Sat, 02 Jan 2026 09:30:00 +0000")
        assert result is not None
        assert result.year == 2026

    def test_handles_none(self):
        assert parse_rss_datetime(None) is None

    def test_handles_empty_string(self):
        assert parse_rss_datetime("") is None


def _mock_response(content: bytes, status_code: int = 200) -> MagicMock:
    """Build a mock response that works as a context manager."""
    m = MagicMock()
    m.status_code = status_code
    m.content = content
    m.text = content.decode("utf-8")
    m.raise_for_status = MagicMock()
    return m
