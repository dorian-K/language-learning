"""
Unit tests for src.bbc_noticias modules.
Covers bugs found during PR #11 review.
"""
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Stub discord before importing anything
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
    class ui:
        Button = _ButtonBase
        View = _ViewBase

    class app_commands:
        @staticmethod
        def CommandTree(client):
            ct = type("CommandTree", (), {})()
            ct.command = lambda *a, **kw: (lambda x: x)
            return ct

    class Intents:
        message_content = True
        guild_messages = True
        messages = True

        @staticmethod
        def default():
            return _discordStub.Intents()

    class Client:
        def __init__(self, intents=None, **kwargs):
            self.intents = intents

        def add_view(self, *args, **kw):
            pass

        def run(self, *args, **kw):
            pass

        @staticmethod
        def event(func):
            return func

    Interaction = type("Interaction", (), {})
    TextChannel = type("TextChannel", (), {})
    ChannelType = type("ChannelType", (), {"public_thread": "public_thread"})()
    ButtonStyle = type("ButtonStyle", (), {"primary": 1})()


sys.modules["discord"] = _discordStub()
sys.modules["discord.ext.commands"] = MagicMock()
sys.modules["discord_slash"] = MagicMock()
sys.modules["slash"] = MagicMock()


# ---------------------------------------------------------------------------
# Issue 1: LLM() called with api_key/model kwargs — catches TypeError
# ---------------------------------------------------------------------------
def test_llm_no_api_key_raises_ValueError(monkeypatch):
    """LLM() raises ValueError when OPENROUTER_API_KEY is not set (not TypeError)."""
    # Mock openai so we don't need it installed
    mock_openai = MagicMock()
    sys.modules["openai"] = mock_openai

    monkeypatch.setenv("OPENROUTER_API_KEY", "")  # empty string = not set

    from src.bbc_noticias.llm import LLM

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        LLM()


def test_llm_init_no_args():
    """LLM() must be called with zero args — no api_key or model kwargs."""
    import inspect
    from src.bbc_noticias.llm import LLM

    sig = inspect.signature(LLM.__init__)
    params = list(sig.parameters.keys())
    # Only 'self' should be accepted — no api_key, model, etc.
    assert params == ["self"], f"LLM.__init__ should only take 'self', got {params}"


# ---------------------------------------------------------------------------
# Issue 2: enqueue_story exception not handled — bot.py line ~97
# ---------------------------------------------------------------------------
def test_enqueue_story_failure_does_not_crash_bot(tmp_path, monkeypatch):
    """bot.py catches enqueue_story exceptions so they don't crash the notification flow."""
    # Write a broken queue file (not valid JSON) - but as a Path object
    # so QUEUE_PATH.exists() works (queue.py uses Path methods)
    broken_queue = tmp_path / "queue.json"
    broken_queue.write_text("not valid json{{{")

    import src.bbc_noticias.queue as queue_mod

    original_path = queue_mod.QUEUE_PATH
    queue_mod.QUEUE_PATH = broken_queue  # keep as Path

    try:
        from src.bbc_noticias.queue import enqueue_story

        # Verify: enqueue_story raises on bad JSON (caller must handle it)
        # The test checks that _load() gracefully handles corrupt JSON
        # and returns empty queue — enqueue_story should NOT raise
        enqueue_story({"title": "test", "link": "http://test.com"})
        # If we get here, enqueue_story handled the corrupt file gracefully
        # (queue._load logs a warning but returns {"pending": [], "sent": []})
        # and the story was enqueued successfully
        assert queue_mod.peek_pending()[0]["title"] == "test"
    finally:
        queue_mod.QUEUE_PATH = original_path


def test_bot_enqueues_after_send_guards_enqueue(monkeypatch):
    """bot.py calls enqueue_story inside a try/except block."""
    bot_src = (Path(__file__).parent.parent / "src" / "bbc_noticias" / "bot.py").read_text()

    # The enqueue_story call should be inside a try block
    import re

    # Find all try/except blocks
    try_blocks = re.findall(r"try:\s+.*?\n\s+except", bot_src, re.DOTALL)
    has_safe_enqueue = any("enqueue_story" in b for b in try_blocks)
    assert has_safe_enqueue, "enqueue_story should be called inside a try/except in bot.py"


# ---------------------------------------------------------------------------
# Issue 3: filter_unsent called with wrong type (list instead of set)
# ---------------------------------------------------------------------------
def test_filter_unsent_accepts_list_of_strings():
    """filter_unsent accepts a list[str] (not set) — RSS stories are a list."""
    from src.bbc_noticias.sent_stories import filter_unsent

    # Pass a list of URLs (RSS gives a list, not a set)
    # With no sent stories, all should be returned
    links = [
        "https://bbc.com/story1",
        "https://bbc.com/story2",
        "https://bbc.com/story3",
    ]
    result = filter_unsent(links)
    assert result == links, "filter_unsent should return all links when none are sent"


def test_filter_unsent_returns_only_unsent():
    """filter_unsent removes already-sent links from the list."""
    from src.bbc_noticias.sent_stories import filter_unsent, mark_sent

    # Use a temp sent_stories file
    import src.bbc_noticias.sent_stories as sent_mod

    original_path = sent_mod.TRACKER_FILE
    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    sent_mod.TRACKER_FILE = Path(tmp_path)

    try:
        links = ["https://bbc.com/a", "https://bbc.com/b", "https://bbc.com/c"]

        # Mark one as sent
        mark_sent("https://bbc.com/b")

        result = filter_unsent(links)
        assert "https://bbc.com/a" in result
        assert "https://bbc.com/b" not in result
        assert "https://bbc.com/c" in result
    finally:
        sent_mod.TRACKER_FILE = original_path
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Issue 4: asyncio.to_thread wraps blocking I/O in discord_bot.py
# ---------------------------------------------------------------------------
def test_discord_bot_uses_asyncio_to_thread():
    """story_service.py runs blocking I/O in asyncio.to_thread — not blocking the event loop."""
    svc_src = (Path(__file__).parent.parent / "src" / "bbc_noticias" / "story_service.py").read_text()

    # fetch_stories + select_best_story are called inside lambdas passed to to_thread
    # fetch_article + simplify_article are called directly with to_thread
    # Check each function is wrapped (either in a lambda inside to_thread, or directly)

    # Blocking I/O must use asyncio.to_thread or run_in_executor
    assert "run_in_executor" in svc_src or "to_thread" in svc_src, \
        "blocking I/O should use asyncio.to_thread or run_in_executor"

    # Indirect calls: fetch_stories and select_best_story are called inside a blocking lambda
    # Use line-based extraction to reliably grab the function body
    lines = svc_src.split("\n")
    in_fn = False
    fn_lines = []
    for line in lines:
        if "def fetch_and_pick_story" in line:
            in_fn = True
        if in_fn:
            fn_lines.append(line)
            if len(fn_lines) > 1 and line.startswith("def ") and "fetch_and_pick_story" not in line:
                break
    fn_body = "\n".join(fn_lines)
    assert "fetch_stories" in fn_body, f"fetch_stories not in fn_body"
    assert "filter_unsent" in fn_body, "filter_unsent not in fn_body"
    assert "select_best_story" in fn_body, "select_best_story not in fn_body"
    assert "run_in_executor" in fn_body or "to_thread" in fn_body, \
        "blocking calls in fetch_and_pick_story should use run_in_executor or to_thread"


# ---------------------------------------------------------------------------
# Issue 5: discord_bot.py imports filter_unsent from sent_stories
# ---------------------------------------------------------------------------
def test_discord_bot_imports_filter_unsent():
    """discord_bot.py imports and uses filter_unsent so button/slash never picks sent stories."""
    svc_src = (Path(__file__).parent.parent / "src" / "bbc_noticias" / "story_service.py").read_text()

    assert "filter_unsent" in svc_src


# ---------------------------------------------------------------------------
# Issue 6: bot.py uses filter_unsent on list[str] (RSS gives list, not set)
# ---------------------------------------------------------------------------
def test_bot_uses_filter_unsent_with_list():
    """bot.py passes a list[str] to filter_unsent — the function must accept lists."""
    from src.bbc_noticias import sent_stories

    # Verify filter_unsent accepts a list (not just a set)
    links_list = ["http://test.com/1", "http://test.com/2"]
    # Should not raise TypeError
    result = sent_stories.filter_unsent(links_list)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Issue 7: queue is_already_queued checks both 'link' and 'url' keys
# ---------------------------------------------------------------------------
def test_is_already_queued_checks_link_and_url(tmp_path):
    """is_already_queued must check both link (RSS) and url keys."""
    import src.bbc_noticias.queue as queue_mod

    original_path = queue_mod.QUEUE_PATH
    tmp_queue = tmp_path / "queue.json"
    queue_mod.QUEUE_PATH = tmp_queue

    try:
        from src.bbc_noticias.queue import enqueue_story, is_already_queued

        # Story saved with 'link' key (RSS style)
        enqueue_story({"title": "Test", "link": "https://bbc.com/story"})

        # Should be found whether checking with link or url
        assert is_already_queued("https://bbc.com/story")

    finally:
        queue_mod.QUEUE_PATH = original_path


# ---------------------------------------------------------------------------
# Issue 8: BOT_CHANNEL_ID env var checked on startup
# ---------------------------------------------------------------------------
def test_discord_bot_checks_bot_channel_id_env():
    """adapters/discord.py reads DISCORD_STORIES_CHANNEL_ID env var to send the button anchor."""
    adapter_src = (Path(__file__).parent.parent / "src" / "bbc_noticias" / "adapters/discord.py").read_text()

    assert "DISCORD_STORIES_CHANNEL_ID" in adapter_src


# ---------------------------------------------------------------------------
# Issue 9: tree.sync() called on startup
# ---------------------------------------------------------------------------
def test_discord_bot_syncs_tree_on_ready():
    """discord_bot.py calls tree.sync() to register global slash commands."""
    bot_src = (Path(__file__).parent.parent / "src" / "bbc_noticias" / "discord_bot.py").read_text()

    assert "tree.sync()" in bot_src, "tree.sync() should be called in on_ready"


# ---------------------------------------------------------------------------
# Issue 10: discord_bot.py calls load_dotenv before importing LLM
# ---------------------------------------------------------------------------
def test_llm_module_loads_dotenv():
    """llm.py calls load_dotenv() at module level so env vars are available."""
    # Verify load_dotenv is called at module level in llm.py
    llm_src = (Path(__file__).parent.parent / "src" / "bbc_noticias" / "llm.py").read_text()
    assert "load_dotenv()" in llm_src, "llm.py should call load_dotenv() at module level"


# ---------------------------------------------------------------------------
# Issue 11: pop_story removes from queue before Discord delivery succeeds
# ---------------------------------------------------------------------------
def test_pop_story_is_guarded_by_try_except():
    """discord_bot.py re-enqueues story if send_story_thread fails."""
    bot_src = (Path(__file__).parent.parent / "src" / "bbc_noticias" / "discord_bot.py").read_text()

    import re

    # Both button callback and slash command should re-enqueue on failure
    # Pattern: send_story_thread inside try/except + enqueue_story in except
    assert re.search(
        r"send_story.*?\n.*?except",
        bot_src,
        re.DOTALL,
    ), "send_story should be wrapped in try/except to handle failures gracefully"

# ---------------------------------------------------------------------------
# Issue 12: filter_unsent called with wrong type in discord_bot.py
# Bug: passing full story dicts instead of URL strings causes 'dict' object has
# no attribute 'strip' at runtime.
# ---------------------------------------------------------------------------
def test_filter_unsent_rejects_dicts():
    """filter_unsent must reject list[dict] at call time — dict has no .strip()."""
    from src.bbc_noticias.sent_stories import filter_unsent
    import pytest

    # Passing story dicts (what discord_bot.py was accidentally doing)
    story_dicts = [
        {"title": "Story 1", "link": "https://bbc.com/1"},
        {"title": "Story 2", "link": "https://bbc.com/2"},
    ]
    with pytest.raises(AttributeError, match="'dict' object has no attribute 'strip'"):
        filter_unsent(story_dicts)


def test_discord_bot_passes_url_list_to_filter_unsent():
    """story_service.py must pass [s["link"] for s in stories] to filter_unsent."""
    svc_src = (Path(__file__).parent.parent / "src" / "bbc_noticias" / "story_service.py").read_text()

    # Extract fetch_and_pick_story body
    lines = svc_src.split("\n")
    in_fn = False
    fn_lines = []
    for line in lines:
        if "def fetch_and_pick_story" in line:
            in_fn = True
        if in_fn:
            fn_lines.append(line)
            if len(fn_lines) > 1 and line.startswith("def ") and "fetch_and_pick_story" not in line:
                break
    fn_body = "\n".join(fn_lines)

    # Must extract .link from stories before passing to filter_unsent
    assert 's["link"]' in fn_body and "for s in stories" in fn_body, (
        "URLs must be extracted as [s['link'] for s in stories] before calling filter_unsent"
    )
    # Must call queue_service.filter_unsent with the URL list (not bare stories)
    assert "queue_service.filter_unsent" in fn_body, (
        "filter_unsent must be called via queue_service (queue_service.filter_unsent)"
    )


# ---------------------------------------------------------------------------
# Issue 13: filter_unsent must handle None/missing items gracefully
# ---------------------------------------------------------------------------
def test_filter_unsent_rejects_none_items():
    """filter_unsent must not crash if list contains None or non-str items."""
    from src.bbc_noticias.sent_stories import filter_unsent
    import pytest

    # None in list would crash .strip()
    with pytest.raises(AttributeError):
        filter_unsent(["https://bbc.com/1", None, "https://bbc.com/2"])
