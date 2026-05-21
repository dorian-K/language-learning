"""
Telegram bot entry point — long-running bot using python-telegram-bot.

Run alongside discord_bot.py:
    python -m src.bbc_noticias.telegram_bot

Or import TelegramAdapter and run() in your own asyncio loop.
"""

import asyncio
import logging
import os

from .adapters.telegram import TelegramAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    channel_id = os.getenv("TELEGRAM_CHANNEL_ID", "")

    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN env var is required")
        return

    adapter = TelegramAdapter(bot_token=bot_token, channel_chat_id=channel_id or None)

    # Start queue subscriber in background before polling begins
    adapter.start_subscriber()

    try:
        adapter.start()
    except KeyboardInterrupt:
        adapter.stop()


if __name__ == "__main__":
    main()
