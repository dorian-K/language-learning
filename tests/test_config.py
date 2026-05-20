"""
Unit tests for config.py — no API keys needed, tests environment loading.
"""

import os
from unittest.mock import patch

from src.bbc_noticias.config import Config, load


class TestConfigValidation:
    def test_default_values(self):
        cfg = Config()
        assert cfg.openrouter_model == "openrouter/auto"
        assert cfg.max_age_hours == 24
        assert cfg.max_stories_for_selection == 15
        assert cfg.dry_run is False

    def test_dry_run_parses_true_values(self):
        for val in ("true", "1", "yes"):
            cfg = Config(dry_run=val)
            assert cfg.dry_run is True

    def test_dry_run_parses_false_values(self):
        for val in ("false", "0", "no", ""):
            cfg = Config(dry_run=val)
            assert cfg.dry_run is False


class TestConfigValidate:
    def test_missing_openrouter_key(self):
        cfg = Config(openrouter_api_key="")
        issues = cfg.validate()
        assert any("OPENROUTER_API_KEY" in i for i in issues)

    def test_missing_both_messenger_channels(self):
        cfg = Config(openrouter_api_key="key123")
        issues = cfg.validate()
        assert any("DISCORD" in i or "TELEGRAM" in i for i in issues)

    def test_telegram_token_without_chat_id(self):
        cfg = Config(openrouter_api_key="key123", telegram_bot_token="abc", telegram_chat_id="")
        issues = cfg.validate()
        assert any("TELEGRAM_CHAT_ID" in i for i in issues)

    def test_all_valid_when_provided(self):
        cfg = Config(
            openrouter_api_key="key123",
            discord_webhook_url="https://discord.com/api/webhooks/123",
        )
        issues = cfg.validate()
        assert len(issues) == 0

    def test_telegram_only_valid(self):
        cfg = Config(
            openrouter_api_key="key123",
            telegram_bot_token="abc",
            telegram_chat_id="123",
        )
        issues = cfg.validate()
        assert len(issues) == 0


class TestLoad:
    with patch.dict(os.environ, {}, clear=True):
        cfg = load()
        # Will have issues since no API key is set, but should still load
        assert cfg.openrouter_model == "openrouter/auto"
        assert cfg.max_age_hours == 24
