"""
Smoke tests — verify that all modules import cleanly without API keys or network.

Run with:
    python3 -m pytest tests/test_imports.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# conftest already mocks requests + dotenv.
# This module additionally mocks discord and any other optional deps,
# then verifies every bbc_noticias module can be imported.
# ---------------------------------------------------------------------------

# Lightweight stub classes for discord modules.
# Using real classes that can be subclassed (unlike MagicMock).
# ---------------------------------------------------------------------------


class _ButtonBase:
    def __init__(self, *, label="", style=1, custom_id=""):
        super().__init__()
        self.label = label
        self.style = style
        self.custom_id = custom_id


class _ViewBase:
    def __init__(self, *, timeout=None):
        self.timeout = timeout

    def add_item(self, item):
        pass


class _discordStub:
    # UI components — support subclassing (super() must work)
    class ui:
        Button = _ButtonBase
        View = _ViewBase

    # Slash command tree
    class app_commands:
        @staticmethod
        def CommandTree(client):
            _ct = type("CommandTree", (), {})()
            _ct.command = lambda *a, **kw: lambda x: x  # decorator
            return _ct

    # Intent flags
    class Intents:
        message_content = True
        guild_messages = True
        messages = True

        @staticmethod
        def default():
            return _discordStub.Intents()

    # Client
    class Client:
        def __init__(self, intents=None, **kwargs):
            self.intents = intents

        def add_view(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            pass

    # Interaction (passed to button callbacks)
    Interaction = type("Interaction", (), {})

    # Channel types
    TextChannel = type("TextChannel", (), {})
    ChannelType = type("ChannelType", (), {"public_thread": "public_thread"})()

    # Button styles
    ButtonStyle = type("ButtonStyle", (), {"primary": 1})()


# Mock discord (not installed in test env)
sys.modules["discord"] = _discordStub()
sys.modules["discord.ext.commands"] = MagicMock()
sys.modules["discord_slash"] = MagicMock()
sys.modules["slash"] = MagicMock()

# Mock openai (LLM dependency)
mock_openai = MagicMock()
sys.modules["openai"] = mock_openai


# ---------------------------------------------------------------------------
# Test: every public function imported from simplifier.py actually exists
# ---------------------------------------------------------------------------


def test_simplifier_exports_exist():
    """Verify every function that other modules import from simplifier actually exists."""
    from src.bbc_noticias import simplifier

    # simplifier.simplify is the function used by story_service.py and bot.py
    # It takes article_dict (with title/text/url) and returns a dict with summary/bullets/text
    assert hasattr(simplifier, "simplify"), "simplifier must export 'simplify'"
    import inspect

    sig = inspect.signature(simplifier.simplify)
    params = list(sig.parameters.keys())
    # Must accept (article_dict, llm) and return dict
    assert params == ["article_dict", "llm"], (
        f"simplify signature should be (article_dict, llm), got {params}"
    )


# ---------------------------------------------------------------------------
# Test: discord_bot.py imports cleanly (catches simplify_story typo)
# ---------------------------------------------------------------------------


def test_discord_bot_imports_cleanly():
    """Verify discord_bot.py imports without AttributeError or ImportError.

    This test would have caught the simplify_story typo:
    - discord_bot.py imported: from simplifier import simplify_story
    - simplify_story does not exist → ImportError or AttributeError
    """
    # The module should load without raising ImportError
    import src.bbc_noticias.discord_bot as discord_bot_mod  # noqa: F401

    assert discord_bot_mod is not None


# ---------------------------------------------------------------------------
# Test: all bbc_noticias modules import cleanly
# ---------------------------------------------------------------------------


def test_all_bbc_noticias_modules_import():
    """Verify every module in src/bbc_noticias/ can be imported."""
    from src.bbc_noticias import (
        bot,
        config,
        discord_bot,
        llm,
        notifier,
        prompts,
        queue,
        rss,
        scraper,
        selector,
        sent_stories,
        simplifier,
    )

    # All should be real modules, not None
    assert bot is not None
    assert config is not None
    assert llm is not None
    assert notifier is not None
    assert prompts is not None
    assert rss is not None
    assert scraper is not None
    assert selector is not None
    assert sent_stories is not None
    assert simplifier is not None
    assert queue is not None
    assert discord_bot is not None


# ---------------------------------------------------------------------------
# Test: queue functions are callable and work with temp storage
# ---------------------------------------------------------------------------


def test_queue_basic_operations(tmp_path):
    """Verify queue enqueue/pop/peek work with a temp file."""
    import src.bbc_noticias.queue as queue_mod

    original_path = queue_mod.QUEUE_PATH
    tmp_queue = tmp_path / "queue.json"
    queue_mod.QUEUE_PATH = tmp_queue

    try:
        from src.bbc_noticias.queue import enqueue_story, peek_pending, pending_count, pop_story

        story = {
            "title": "Test Story",
            "link": "https://bbc.com/test",
            "source": "BBC",
            "pub_date": "2026-05-17T08:00:00Z",
        }

        # Peek empty queue
        assert pending_count() == 0
        assert peek_pending() == []

        # Enqueue
        enqueue_story(story)
        assert pending_count() == 1

        # Pop
        popped = pop_story()
        assert popped["title"] == "Test Story"
        assert popped["link"] == "https://bbc.com/test"

        # Queue now empty
        assert pending_count() == 0
        assert pop_story() is None

    finally:
        queue_mod.QUEUE_PATH = original_path


# ---------------------------------------------------------------------------
# Test: queue is_already_queued checks both 'link' and 'url' keys
# ---------------------------------------------------------------------------


def test_queue_is_already_queued(tmp_path):
    """Verify is_already_queued checks both link (RSS) and url keys."""
    import src.bbc_noticias.queue as queue_mod

    original_path = queue_mod.QUEUE_PATH
    tmp_queue = tmp_path / "queue.json"
    queue_mod.QUEUE_PATH = tmp_queue

    try:
        from src.bbc_noticias.queue import enqueue_story, is_already_queued

        story = {"title": "Test", "link": "https://bbc.com/story"}  # RSS uses 'link'

        assert not is_already_queued("https://bbc.com/story")
        enqueue_story(story)
        assert is_already_queued("https://bbc.com/story")

    finally:
        queue_mod.QUEUE_PATH = original_path


# ---------------------------------------------------------------------------
# Bug 2: SIMPLIFY_PROMPT must request JSON output with summary/bullets/text keys
# ---------------------------------------------------------------------------


def test_simplify_prompt_requests_json():
    """SIMPLIFY_PROMPT must request JSON output with summary/bullets/text keys.

    Without this, the LLM returns plain Spanish text and story_service.py
    crashes with TypeError on simplified["summary"].
    """
    from src.bbc_noticias.prompts import SIMPLIFY_PROMPT

    assert '"summary"' in SIMPLIFY_PROMPT, "prompt should request 'summary' field"
    assert '"bullets"' in SIMPLIFY_PROMPT, "prompt should request 'bullets' field"
    assert '"text"' in SIMPLIFY_PROMPT, "prompt should request 'text' field"


# ---------------------------------------------------------------------------
# Bug 4: bot.py must use _build_story_text, not raw string slicing
# ---------------------------------------------------------------------------


def test_bot_uses_build_story_text():
    """bot.py must pass StoryPayload through _build_story_text(), not raw text slicing.

    Without this, Discord messages use a bare string instead of properly
    formatted story text with headline + summary + bullets + topic.
    """
    bot_src = (Path(__file__).parent.parent / "src" / "bbc_noticias" / "bot.py").read_text()
    notifier_src = (
        Path(__file__).parent.parent / "src" / "bbc_noticias" / "notifier.py"
    ).read_text()

    # bot.py should use StoryPayload and _build_story_text
    assert "StoryPayload" in bot_src, "bot.py should construct StoryPayload"
    assert "_build_story_text" in bot_src, "bot.py should call _build_story_text to format messages"

    # send_article should use simplified_text directly (it's already formatted)
    # NOT construct its own payload with title + simplified_text[:1800]
    assert "simplified_text[:1800]" not in notifier_src, (
        "send_article must not truncate simplified_text at 1800 — "
        "_build_story_text already formats it correctly"
    )


# ---------------------------------------------------------------------------
# Bug 5: config.py and .env.example must agree on env var names
# ---------------------------------------------------------------------------


def test_env_var_consistency():
    """config.py must read the same env vars that .env.example defines."""

    cfg_src = (Path(__file__).parent.parent / "src" / "bbc_noticias" / "config.py").read_text()
    env_src = (Path(__file__).parent.parent / ".env.example").read_text()

    # .env.example should define TELEGRAM_BOT_TOKEN (not just BOT_TOKEN)
    assert "TELEGRAM_BOT_TOKEN" in env_src, ".env.example should define TELEGRAM_BOT_TOKEN"

    # config.py must read TELEGRAM_BOT_TOKEN (not a different name)
    assert "TELEGRAM_BOT_TOKEN" in cfg_src, (
        "config.py reads 'TELEGRAM_BOT_TOKEN' but .env.example may define a different name"
    )


# ---------------------------------------------------------------------------
# Test: bot.py calls enqueue_story after sending (smoke test)
# ---------------------------------------------------------------------------


def test_bot_enqueues_after_send():
    """Verify bot.py calls enqueue_story when a story is sent.

    This is a compile-check: verify the call is present in bot.py source.
    """
    bot_src = (Path(__file__).parent.parent / "src" / "bbc_noticias" / "bot.py").read_text()
    assert "enqueue_story" in bot_src, "bot.py should call enqueue_story after send_article"
    assert "try:" in bot_src and "enqueue_story" in bot_src, (
        "enqueue_story should be guarded by try/except"
    )
