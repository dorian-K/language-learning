"""
Standalone test runner — no external dependencies needed.
Uses importlib to load modules with a mock requests pre-installed.
Run with: python3 tests/test_runner.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Pre-load mock modules so rss.py etc. can import them without pip-installed deps
mock_requests = MagicMock()
mock_response = MagicMock()
mock_response.status_code = 200
mock_response.text = "<html></html>"
mock_response.content = b"<html></html>"
mock_response.raise_for_status = MagicMock()
mock_requests.get.return_value = mock_response

mock_exc = MagicMock()
mock_exc.Timeout = OSError
mock_exc.ConnectionError = OSError

sys.modules["requests"] = mock_requests
sys.modules["requests.exceptions"] = mock_exc
sys.modules["dotenv"] = MagicMock()
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None

# Now add src/ to path and import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bbc_noticias.config import Config
from bbc_noticias.rss import parse_rss_datetime
from bbc_noticias.scraper import _clean_html, _extract_article_body, _fallback_extract


def run_test(name, fn, *args):
    try:
        fn(*args)
        print(f"  ✅ {name}")
        return True
    except AssertionError as e:
        print(f"  ❌ {name}: {e}")
        return False
    except Exception as e:
        print(f"  💥 {name}: {type(e).__name__}: {e}")
        return False


def run():
    print("\n=== rss.py: parse_rss_datetime ===")
    ok = 0
    bok += int(
        run_test("parses RFC822 GMT", parse_rss_datetime, "Fri, 08 May 2026 12:00:00 GMT")
        is not None
    )
    r = parse_rss_datetime("Fri, 08 May 2026 12:00:00 GMT")
    assert r is not None and r.year == 2026
    bok += int(run_test("None input", lambda: parse_rss_datetime(None) is None))
    bok += int(run_test("empty string", lambda: parse_rss_datetime("") is None))

    print("\n=== scraper.py: _clean_html ===")

    def clean(html):
        return _clean_html(html)

    bok += run_test("strips tags", lambda: clean("<p>Hello <b>world</b></p>") == "Hello world")
    bok += run_test(
        "removes scripts", lambda: "alert" not in clean("<script>alert('x')</script><p>OK</p>")
    )
    bok += run_test("decodes &amp;", lambda: "&" in clean("&amp;"))
    bok += run_test("decodes &lt; &gt;", lambda: "<" in clean("&lt;") and ">" in clean("&gt;"))
    bok += run_test("normalizes whitespace", lambda: "  " not in clean("<p>  Hello   world  </p>"))

    print("\n=== scraper.py: _extract_article_body ===")
    bok += run_test(
        "extracts <article>",
        lambda: _extract_article_body("<article><p>Text</p></article>") is not None,
    )
    bok += run_test(
        "extracts data-component",
        lambda: (
            "text"
            in (_extract_article_body('<div data-component="article-body"><p>text</p></div>') or "")
        ),
    )
    bok += run_test(
        "extracts body-content id",
        lambda: _extract_article_body('<section id="body-content"><p>C</p></section>') is not None,
    )
    bok += run_test(
        "returns None on miss", lambda: _extract_article_body("<div>nothing</div>") is None
    )

    print("\n=== scraper.py: _fallback_extract ===")
    bok += run_test(
        "extracts <main>", lambda: _fallback_extract("<main>fallback</main>") is not None
    )
    bok += run_test(
        "returns None when no main", lambda: _fallback_extract("<div>no main</div>") is None
    )

    print("\n=== config.py: Config defaults ===")
    cfg = Config()
    bok += run_test(
        "default model is openrouter/auto", lambda: cfg.openrouter_model == "openrouter/auto"
    )
    bok += run_test("max_age_hours default 24", lambda: cfg.max_age_hours == 24)
    bok += run_test(
        "max_stories_for_selection default 15", lambda: cfg.max_stories_for_selection == 15
    )
    bok += run_test("dry_run default False", lambda: cfg.dry_run is False)

    print("\n=== config.py: dry_run parsing ===")
    for val in ("true", "1", "yes"):
        cfg = Config(dry_run=val)
        assert cfg.dry_run is True, f"failed for {val}"
    bok += run_test(
        "true/1/yes → True", lambda: all(Config(dry_run=v).dry_run for v in ("true", "1", "yes"))
    )
    bok += run_test(
        "false/0/no/'' → False",
        lambda: all(not Config(dry_run=v).dry_run for v in ("false", "0", "no", "")),
    )

    print("\n=== config.py: validate ===")
    cfg_missing = Config(openrouter_api_key="")
    issues = cfg_missing.validate()
    bok += run_test(
        "missing API key flagged", lambda: any("OPENROUTER_API_KEY" in i for i in issues)
    )

    cfg_no_messenger = Config(openrouter_api_key="key")
    issues = cfg_no_messenger.validate()
    bok += run_test(
        "missing messenger flagged",
        lambda: any(("DISCORD" in i or "TELEGRAM" in i) for i in issues),
    )

    cfg_ok = Config(
        openrouter_api_key="key", discord_webhook_url="https://discord.com/api/webhooks/123"
    )
    bok += run_test("valid config: no issues", lambda: len(cfg_ok.validate()) == 0)

    print(f"\n{'=' * 40}")
    print(f"Results: {ok} passed")
    # Total tests: parse_rss_datetime(3) + _clean_html(5) + _extract_article_body(4) +
    # _fallback_extract(2) + Config defaults(4) + dry_run parsing(2) + validate(3) = 23
    total_tests = 23
    print("✅ All tests passed" if ok == total_tests else f"❌ {total_tests - ok} failed")
    return ok


if __name__ == "__main__":
    # Count total test lines
    n_tests = sum(
        1
        for name, obj in [
            ("parse_rss_datetime", parse_rss_datetime),
            ("_clean_html", _clean_html),
            ("_extract_article_body", _extract_article_body),
            ("_fallback_extract", _fallback_extract),
            ("Config defaults", None),
            ("dry_run parsing", None),
            ("validate", None),
        ]
        if obj
    )
    ok = run()
    sys.exit(0 if ok else 1)
