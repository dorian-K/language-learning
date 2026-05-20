"""
Unit tests for BBC Noticias bot modules that don't need API keys.

Run with:
    python3 -m unittest tests.test_all   (requires pytest or unittest installed)
Or run directly:
    python3 tests/test_all.py

If requests/dotenv are not installed, the tests will still run correctly
because conftest.py pre-mocks them before any imports.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# conftest-style setup: mock requests + dotenv before any imports
# ---------------------------------------------------------------------------
sys.modules["requests"] = MagicMock()
sys.modules["requests.exceptions"] = MagicMock()
sys.modules["dotenv"] = MagicMock()
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bbc_noticias.config import Config
from bbc_noticias.rss import fetch_stories, parse_rss_datetime
from bbc_noticias.scraper import (
    _clean_html,
    _extract_article_body,
    _fallback_extract,
    fetch_article,
)


# ---------------------------------------------------------------------------
# Helper: make a fake HTTP response with given XML bytes content
# ---------------------------------------------------------------------------
def _make_response(xml_bytes: bytes, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = xml_bytes
    resp.raise_for_status = MagicMock()
    return resp


def _make_mock_get_call(
    xml_bytes: bytes,
    status_code: int = 200,
    title: str = "Test Story",
    link: str = "https://bbc.com/test",
):
    """Create a mock requests.get that returns our test XML for each of the 4 RSS feeds.

    Uses a call-counting side_effect so each feed gets the same test story
    (they all merge into 4 identical items in the result, which is expected
    when you only override a single feed's response).
    """
    from contextlib import contextmanager

    call_responses = [
        _make_response(xml_bytes, status_code),
    ]
    call_count = [0]

    def get_side_effect(*args, **kwargs):
        resp = call_responses[0]
        call_count[0] += 1
        return resp

    import bbc_noticias.rss as rss_mod

    original_get = rss_mod.requests.get
    rss_mod.requests.get = get_side_effect

    @contextmanager
    def _ctx():
        try:
            yield
        finally:
            rss_mod.requests.get = original_get

    return _ctx()


# ---------------------------------------------------------------------------
# rss.py
# ---------------------------------------------------------------------------


class TestParseRssDatetime(TestCase):
    def test_parses_rfc822_gmt(self):
        r = parse_rss_datetime("Fri, 08 May 2026 12:00:00 GMT")
        self.assertIsNotNone(r)
        self.assertEqual(r.year, 2026)
        self.assertEqual(r.month, 5)
        self.assertEqual(r.day, 8)

    def test_parses_rfc822_with_offset(self):
        r = parse_rss_datetime("Fri, 08 May 2026 12:00:00 +0000")
        self.assertIsNotNone(r)
        self.assertEqual(r.year, 2026)

    def test_parses_rfc822_rfc2822_variant(self):
        r = parse_rss_datetime("Sat, 02 Jan 2026 09:30:00 +0000")
        self.assertIsNotNone(r)
        self.assertEqual(r.year, 2026)

    def test_none_returns_none(self):
        self.assertIsNone(parse_rss_datetime(None))

    def test_empty_returns_none(self):
        self.assertIsNone(parse_rss_datetime(""))

    def test_invalid_returns_none(self):
        self.assertIsNone(parse_rss_datetime("not a valid date"))


class TestFetchStories(TestCase):
    def _rss_xml(self, title: str, link: str, pub_date: str | None = None) -> bytes:
        pd = pub_date or datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<rss version="2.0"><channel><title>BBC Mundo Test</title>'
            f"<item><title>{title}</title><link>{link}</link>"
            f"<pubDate>{pd}</pubDate></item></channel></rss>"
        ).encode()

    def test_parses_single_story(self):
        """All 4 active feeds return the same test item → 4 merged stories."""
        xml = self._rss_xml("Mi primera historia", "https://bbc.com/123")
        with _make_mock_get_call(xml):
            stories = fetch_stories(max_age_hours=24)
        self.assertEqual(len(stories), 4)  # one per feed (4 active feeds)
        self.assertEqual(stories[0]["title"], "Mi primera historia")
        self.assertEqual(stories[0]["link"], "https://bbc.com/123")

    def test_parses_multiple_stories(self):
        """Each feed has 2 items → 8 merged stories total (4 active feeds)."""
        recent_date = (datetime.now(UTC) - timedelta(hours=2)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        xml = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<rss version="2.0"><channel><title>BBC</title>'
            f"<item><title>Story A</title><link>https://a.com</link><pubDate>{recent_date}</pubDate></item>"
            f"<item><title>Story B</title><link>https://b.com</link><pubDate>{recent_date}</pubDate></item>"
            f"</channel></rss>"
        ).encode()
        with _make_mock_get_call(xml):
            stories = fetch_stories(max_age_hours=24)
        self.assertEqual(len(stories), 8)  # 2 per feed × 4 feeds
        self.assertEqual(stories[0]["title"], "Story A")
        self.assertEqual(stories[1]["title"], "Story B")

    def test_filters_stories_older_than_max_age_hours(self):
        """Old story is filtered out from all feeds → 0 stories."""
        old_date = (datetime.now(UTC) - timedelta(hours=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        xml = self._rss_xml("Historia vieja", "https://bbc.com/vieja", old_date)
        with _make_mock_get_call(xml):
            stories = fetch_stories(max_age_hours=24)
        self.assertEqual(len(stories), 0)

    def test_includes_stories_within_max_age_hours(self):
        """Recent story is kept in all feeds → 4 stories (one per feed, 4 active feeds)."""
        recent_date = (datetime.now(UTC) - timedelta(hours=12)).strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        )
        xml = self._rss_xml("Historia reciente", "https://bbc.com/reciente", recent_date)
        with _make_mock_get_call(xml):
            stories = fetch_stories(max_age_hours=24)
        self.assertEqual(len(stories), 4)

    def test_skips_items_without_title(self):
        """Items with empty titles are skipped → 0 stories."""
        xml = (
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<rss version="2.0"><channel><title>T</title>'
            b"<item><title></title><link>https://x.com</link><pubDate>Fri, 08 May 2026 12:00:00 GMT</pubDate></item>"
            b"</channel></rss>"
        )
        with _make_mock_get_call(xml):
            stories = fetch_stories(max_age_hours=24)
        self.assertEqual(len(stories), 0)

    def test_skips_items_without_link(self):
        """Items with empty links are skipped → 0 stories."""
        xml = (
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<rss version="2.0"><channel><title>T</title>'
            b"<item><title>Has title</title><link></link><pubDate>Fri, 08 May 2026 12:00:00 GMT</pubDate></item>"
            b"</channel></rss>"
        )
        with _make_mock_get_call(xml):
            stories = fetch_stories(max_age_hours=24)
        self.assertEqual(len(stories), 0)

    def test_returns_empty_on_invalid_xml(self):
        """Invalid XML → empty story list."""
        with _make_mock_get_call(b"not xml at all"):
            stories = fetch_stories(max_age_hours=24)
        self.assertEqual(len(stories), 0)

    def test_uses_source_from_channel_title(self):
        """Source label comes from channel/title element."""
        recent_date = (datetime.now(UTC) - timedelta(hours=2)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        xml = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<rss version="2.0"><channel><title>BBC Mundo Portada</title>'
            f"<item><title>S</title><link>https://x.com</link><pubDate>{recent_date}</pubDate></item>"
            f"</channel></rss>"
        ).encode()
        with _make_mock_get_call(xml):
            stories = fetch_stories(max_age_hours=24)
        self.assertEqual(stories[0]["source"], "BBC Mundo Portada")


# ---------------------------------------------------------------------------
# scraper.py
# ---------------------------------------------------------------------------


class TestCleanHtml(TestCase):
    def test_strips_all_html_tags(self):
        self.assertEqual(_clean_html("<p>Hello <strong>world</strong></p>"), "Hello world")

    def test_strips_nested_tags(self):
        result = _clean_html("<div><p>Para1 <span><b>bold</b></span></p><p>Para2</p></div>")
        self.assertEqual(result, "Para1 bold\nPara2")

    def test_replaces_br_with_newline(self):
        result = _clean_html("Line 1<br />Line 2<br>Line 3")
        self.assertIn("Line 1", result)
        self.assertIn("Line 2", result)
        self.assertIn("Line 3", result)

    def test_replaces_div_with_newline(self):
        result = _clean_html("<div>Para1</div><div>Para2</div>")
        self.assertIn("Para1", result)
        self.assertIn("Para2", result)

    def test_removes_script_tags_and_content(self):
        result = _clean_html("<script>alert('x')</script><p>OK</p>")
        self.assertNotIn("alert", result)
        self.assertIn("OK", result)

    def test_removes_style_tags_and_content(self):
        result = _clean_html("<style>body { color: red; }</style><p>Visible</p>")
        self.assertNotIn("color", result)
        self.assertIn("Visible", result)

    def test_removes_nav_tags(self):
        result = _clean_html("<nav>Navigation</nav><article>Content</article>")
        self.assertNotIn("Navigation", result)
        self.assertIn("Content", result)

    def test_decodes_html_entities(self):
        result = _clean_html("&amp; &lt; &gt; &quot; &#39; &ntilde;")
        self.assertIn("&", result)
        self.assertIn("<", result)
        self.assertIn(">", result)
        self.assertIn('"', result)
        self.assertIn("'", result)
        self.assertIn("ñ", result)

    def test_decodes_spanish_accents(self):
        result = _clean_html("&aacute; &eacute; &iacute; &oacute; &uacute;")
        self.assertIn("á", result)
        self.assertIn("é", result)
        self.assertIn("í", result)
        self.assertIn("ó", result)
        self.assertIn("ú", result)

    def test_normalizes_multiple_spaces(self):
        result = _clean_html("<p>Hello    world   test</p>")
        self.assertNotIn("  ", result)

    def test_strips_leading_trailing_whitespace_per_line(self):
        result = _clean_html("<p>  Hello  </p>")
        self.assertNotIn("  ", result)
        self.assertTrue(result.startswith("Hello"))

    def test_omits_empty_lines(self):
        result = _clean_html("<p>Para1</p>\n\n<p></p>\n\n<p>Para2</p>")
        self.assertNotIn("\n\n", result)

    def test_handles_empty_input(self):
        self.assertEqual(_clean_html(""), "")


class TestExtractArticleBody(TestCase):
    def test_extracts_article_tag(self):
        html = "<article><p>Article paragraph</p><p>Another</p></article>"
        result = _extract_article_body(html)
        self.assertIsNotNone(result)
        self.assertIn("Article paragraph", result)
        self.assertIn("Another", result)

    def test_extracts_data_component_article_body(self):
        html = '<div data-component="article-body"><p>Content here</p></div>'
        result = _extract_article_body(html)
        self.assertIsNotNone(result)
        self.assertIn("Content here", result)

    def test_extracts_section_with_body_content_id(self):
        html = '<section id="body-content"><p>Body content text</p></section>'
        result = _extract_article_body(html)
        self.assertIsNotNone(result)
        self.assertIn("Body content text", result)

    def test_returns_none_when_no_pattern_matches(self):
        html = '<div class="sidebar"><p>Nothing article-like here</p></div>'
        self.assertIsNone(_extract_article_body(html))


class TestFallbackExtract(TestCase):
    def test_extracts_main_tag(self):
        html = "<main><p>Main content</p></main>"
        result = _fallback_extract(html)
        self.assertIsNotNone(result)
        self.assertIn("Main content", result)

    def test_returns_none_when_no_main_tag(self):
        html = "<div><p>No main tag present</p></div>"
        self.assertIsNone(_fallback_extract(html))


class TestFetchArticle(TestCase):
    @patch("bbc_noticias.scraper.requests.get")
    def test_returns_text_on_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "<html><body><article><p>Noticia de prueba.</p><p>Más información aquí.</p></article></body></html>"

        result = fetch_article("https://www.bbc.com/test")
        self.assertIsNotNone(result)
        self.assertIn("Noticia de prueba", result)
        self.assertIn("Más información aquí", result)

    @patch("bbc_noticias.scraper.requests.get")
    def test_uses_article_body_extraction(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = '<html><body><article data-component="article-body"><p>From article body</p></article></body></html>'

        result = fetch_article("https://www.bbc.com/test2")
        self.assertIsNotNone(result)
        self.assertIn("From article body", result)

    @patch("bbc_noticias.scraper.requests.get")
    def test_returns_none_on_404(self, mock_get):
        mock_get.return_value.status_code = 404

        self.assertIsNone(fetch_article("https://www.bbc.com/not-found"))

    @patch("bbc_noticias.scraper.requests.get")
    def test_returns_none_on_500(self, mock_get):
        mock_get.return_value.status_code = 500

        self.assertIsNone(fetch_article("https://www.bbc.com/error"))

    @patch("bbc_noticias.scraper.requests.get")
    def test_returns_none_on_timeout(self, mock_get):
        import requests as req_module

        mock_get.side_effect = req_module.Timeout("Connection timed out")

        self.assertIsNone(fetch_article("https://www.bbc.com/slow"))

    @patch("bbc_noticias.scraper.requests.get")
    def test_returns_none_on_connection_error(self, mock_get):
        import requests as req_module

        mock_get.side_effect = req_module.ConnectionError("Connection refused")

        self.assertIsNone(fetch_article("https://www.bbc.com/refused"))

    @patch("bbc_noticias.scraper.requests.get")
    def test_returns_none_on_generic_request_exception(self, mock_get):
        mock_get.side_effect = Exception("Unexpected error")

        self.assertIsNone(fetch_article("https://www.bbc.com/broken"))

    @patch("bbc_noticias.scraper.requests.get")
    def test_sets_correct_headers(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "<html><body><article><p>Test</p></article></body></html>"

        fetch_article("https://www.bbc.com/headers")
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args.kwargs
        self.assertIn("User-Agent", call_kwargs.get("headers", {}))
        self.assertIn("Accept-Language", call_kwargs.get("headers", {}))


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------


class TestConfigDefaults(TestCase):
    def test_default_openrouter_model(self):
        self.assertEqual(Config().openrouter_model, "openrouter/auto")

    def test_default_max_age_hours(self):
        self.assertEqual(Config().max_age_hours, 24)

    def test_default_max_stories_for_selection(self):
        self.assertEqual(Config().max_stories_for_selection, 15)

    def test_default_dry_run_is_false(self):
        self.assertIs(Config().dry_run, False)


class TestConfigDryRunParsing(TestCase):
    def test_dry_run_true_string_values(self):
        for val in ("true", "True", "TRUE", "1", "yes", "Yes", "YES"):
            cfg = Config(dry_run=val)
            self.assertIs(cfg.dry_run, True, f"failed for {val!r}")

    def test_dry_run_false_string_values(self):
        for val in ("false", "False", "FALSE", "0", "no", "No", "NO", ""):
            cfg = Config(dry_run=val)
            self.assertIs(cfg.dry_run, False, f"failed for {val!r}")

    def test_dry_run_bool_passthrough(self):
        self.assertIs(Config(dry_run=True).dry_run, True)
        self.assertIs(Config(dry_run=False).dry_run, False)


class TestConfigValidate(TestCase):
    def test_missing_openrouter_api_key_flagged(self):
        cfg = Config(openrouter_api_key="")
        issues = cfg.validate()
        self.assertTrue(any("OPENROUTER_API_KEY" in i for i in issues))

    def test_missing_both_messenger_channels_flagged(self):
        cfg = Config(openrouter_api_key="valid-key")
        issues = cfg.validate()
        self.assertTrue(any(("DISCORD" in i or "TELEGRAM" in i) for i in issues))

    def test_telegram_bot_token_without_chat_id_flagged(self):
        cfg = Config(openrouter_api_key="k", telegram_bot_token="abc", telegram_chat_id="")
        issues = cfg.validate()
        self.assertTrue(any("TELEGRAM_CHAT_ID" in i for i in issues))

    def test_discord_webhook_only_is_valid(self):
        cfg = Config(
            openrouter_api_key="k", discord_webhook_url="https://discord.com/api/webhooks/123/abc"
        )
        self.assertEqual(cfg.validate(), [])

    def test_telegram_only_is_valid(self):
        cfg = Config(openrouter_api_key="k", telegram_bot_token="abc", telegram_chat_id="123456")
        self.assertEqual(cfg.validate(), [])

    def test_both_channels_valid(self):
        cfg = Config(
            openrouter_api_key="k",
            discord_webhook_url="https://discord.com/api/webhooks/123",
            telegram_bot_token="abc",
            telegram_chat_id="123",
        )
        self.assertEqual(cfg.validate(), [])


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
