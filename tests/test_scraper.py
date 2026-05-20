"""
Unit tests for scraper.py — no API keys needed, tests HTML extraction and cleaning.
"""

from unittest.mock import MagicMock, patch

from src.bbc_noticias.scraper import (
    _clean_html,
    _extract_article_body,
    _fallback_extract,
    fetch_article,
)


class TestCleanHtml:
    def test_strips_all_html_tags(self):
        html = "<p>Hello <strong>world</strong></p>"
        assert _clean_html(html) == "Hello world"

    def test_replaces_br_with_newline(self):
        html = "Line 1<br />Line 2"
        result = _clean_html(html)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_normalizes_whitespace(self):
        html = "<p>  Hello   world  </p>"
        result = _clean_html(html)
        assert "  " not in result
        assert "Hello world" in result

    def test_decodes_common_html_entities(self):
        html = "&amp; &lt; &gt; &quot; &#39; &ntilde;"
        result = _clean_html(html)
        assert "&" in result
        assert "<" in result
        assert ">" in result

    def test_removes_scripts_and_styles(self):
        html = "<script>alert('x')</script><p>Content</p><style>.foo{}</style>"
        result = _clean_html(html)
        assert "alert" not in result
        assert "Content" in result
        assert ".foo" not in result


class TestExtractArticleBody:
    def test_extracts_from_article_tag(self):
        html = "<article><p>Paragraph 1</p><p>Paragraph 2</p></article>"
        result = _extract_article_body(html)
        assert result is not None
        assert "Paragraph 1" in result
        assert "Paragraph 2" in result

    def test_extracts_from_data_component_attribute(self):
        html = '<div data-component="article-body"><p>Article text here</p></div>'
        result = _extract_article_body(html)
        assert result is not None
        assert "Article text here" in result

    def test_extracts_from_section_with_body_content_id(self):
        html = '<section id="body-content"><p>Content</p></section>'
        result = _extract_article_body(html)
        assert result is not None
        assert "Content" in result

    def test_returns_none_when_no_match(self):
        html = "<div>No article here</div>"
        result = _extract_article_body(html)
        assert result is None


class TestFallbackExtract:
    def test_extracts_from_main_tag(self):
        html = "<main><p>Main content</p></main>"
        result = _fallback_extract(html)
        assert result is not None
        assert "Main content" in result

    def test_returns_none_when_no_main(self):
        html = "<div>No main here</div>"
        result = _fallback_extract(html)
        assert result is None


class TestFetchArticle:
    @patch("src.bbc_noticias.scraper.requests.get")
    def test_fetch_article_success(self, mock_get):
        html = """
        <html><body>
        <article>
            <p>Este es un artículo de prueba.</p>
            <p>Segundo párrafo con información importante.</p>
        </article>
        </body></html>
        """
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = html
        mock_get.return_value.raise_for_status = MagicMock()

        result = fetch_article("https://www.bbc.com/test")
        assert result is not None
        assert "artículo" in result
        assert "prueba" in result

    @patch("src.bbc_noticias.scraper.requests.get")
    def test_fetch_article_http_error_returns_none(self, mock_get):
        mock_get.return_value.status_code = 404
        mock_get.return_value.raise_for_status = MagicMock(side_effect=Exception("not found"))

        result = fetch_article("https://www.bbc.com/not-found")
        assert result is None

    @patch("src.bbc_noticias.scraper.requests.get")
    def test_fetch_article_timeout_returns_none(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        result = fetch_article("https://www.bbc.com/slow")
        assert result is None

    @patch("src.bbc_noticias.scraper.requests.get")
    def test_fetch_article_with_bbc_article_tag(self, mock_get):
        html = """
        <html>
        <article data-component="article-body">
            <p>Noticia sobre tecnología y avances científicos.</p>
            <p>Más detalles del artículo.</p>
        </article>
        </html>
        """
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = html
        mock_get.return_value.raise_for_status = MagicMock()

        result = fetch_article("https://www.bbc.com/article")
        assert result is not None
        assert "tecnología" in result
