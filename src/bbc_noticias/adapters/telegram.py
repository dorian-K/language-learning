"""
Telegram adapter stub — future implementation for Telegram channel posting.
"""

import logging
from .base import PlatformAdapter, StoryPayload


logger = logging.getLogger(__name__)


class TelegramAdapter(PlatformAdapter):
    """
    Telegram-specific posting logic (NOT YET IMPLEMENTED).

    To implement:
    1. Store bot token + channel chat_id
    2. Implement post_channel() using telegram Bot API
    3. Implement create_thread() — Telegram has topics in forum channels
    4. Implement post_thread()
    5. Implement add_reaction() using telegram message reactions
    """

    def __init__(self, bot_token: str, channel_chat_id: str):
        self.bot_token = bot_token
        self.channel_chat_id = channel_chat_id

    async def post_channel(self, payload: StoryPayload) -> str:
        raise NotImplementedError("Telegram adapter not yet implemented")

    async def create_thread(self, payload: StoryPayload, channel_msg_id: str) -> str:
        raise NotImplementedError("Telegram adapter not yet implemented")

    async def post_thread(self, thread_id: str, payload: StoryPayload) -> None:
        raise NotImplementedError("Telegram adapter not yet implemented")

    async def add_reaction(self, channel_msg_id: str) -> None:
        raise NotImplementedError("Telegram adapter not yet implemented")
