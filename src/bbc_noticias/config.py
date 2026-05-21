"""
User profile — tune the bot's behaviour via environment variables.
"""

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return False


@dataclass
class Config:
    # OpenRouter
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openrouter/auto")

    # Messenger channels (both can be set; both are optional)
    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # RSS feed options
    max_age_hours: int = int(os.getenv("MAX_AGE_HOURS", "24"))

    # How many stories to include in the selection prompt
    # (OpenRouter context windows vary; 15 is a safe default)
    max_stories_for_selection: int = int(os.getenv("MAX_STORIES_FOR_SELECTION", "15"))

    # Dry-run mode — skips sending to Discord/Telegram
    dry_run: bool = field(default_factory=lambda: _parse_bool(os.getenv("DRY_RUN", "false")))

    def __post_init__(self):
        # Ensure dry_run is always a proper bool, even when passed as a string
        self.dry_run = _parse_bool(self.dry_run)

    def validate(self) -> list[str]:
        """Return list of missing/invalid config values for user to fix."""
        issues = []
        if not self.openrouter_api_key:
            issues.append("OPENROUTER_API_KEY is not set")
        if not self.discord_webhook_url and not self.telegram_bot_token:
            issues.append(
                "Neither DISCORD_WEBHOOK_URL nor TELEGRAM_BOT_TOKEN is set — "
                "the article won't be sent anywhere!"
            )
        if self.telegram_bot_token and not self.telegram_chat_id:
            issues.append("TELEGRAM_BOT_TOKEN is set but TELEGRAM_CHAT_ID is missing")
        return issues


def load() -> Config:
    cfg = Config()
    issues = cfg.validate()
    if issues:
        logger.warning("[config] Missing configuration:", exc_info=True)
        for issue in issues:
            logger.warning("  - %s", issue, exc_info=True)
    return cfg
