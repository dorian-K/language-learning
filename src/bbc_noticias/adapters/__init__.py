"""
Platform adapters — Discord, Telegram, and future platforms.

Usage:
    from src.bbc_noticias.adapters import PlatformAdapter
    from src.bbc_noticias.adapters.discord import DiscordAdapter
    from src.bbc_noticias.adapters.telegram import TelegramAdapter
"""

from .base import PlatformAdapter, StoryPayload
from .discord import DiscordAdapter
from .telegram import TelegramAdapter

__all__ = ["PlatformAdapter", "StoryPayload", "DiscordAdapter", "TelegramAdapter"]
