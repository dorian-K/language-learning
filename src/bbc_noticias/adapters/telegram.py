"""
Telegram adapter — posts BBC stories to Telegram channels and DMs.

Two ways to get a story:
1. /historia command — sends a story to the user's DM (or current chat)
2. "📰 Nueva historia" inline button on any chat — sends story to the user's DM

Environment variables:
  TELEGRAM_BOT_TOKEN  — bot token from @BotFather
  TELEGRAM_CHANNEL_ID — channel ID (numeric, e.g. -1001234567890)
"""

import asyncio
import logging
import os
from uuid import uuid4

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

from .base import PlatformAdapter, StoryPayload


logger = logging.getLogger(__name__)

# Transient user sessions: chat_id → story payload (awaiting button click)
# For /historia flow where we send a preview + button before posting
_pending: dict[int, dict] = {}


def _build_story_text(payload: StoryPayload) -> str:
    """Format a story as a readable Telegram message."""
    return (
        f"📰 *{payload.headline}*\n\n"
        f"{payload.summary}\n\n"
        f"{payload.bullets}\n\n"
        f"🔗 {payload.url}"
    )


async def _send_story_to(chat_id: int, payload: StoryPayload, bot: Bot) -> None:
    """Send a formatted story to the given chat."""
    text = _build_story_text(payload)
    await bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode="Markdown",
        disable_web_page_preview=True,
    )


# ── Telegram-specific handlers (not part of PlatformAdapter) ─────────────────

async def _historia_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /historia — send a story to the user's DM (or current chat)."""
    from ..story_service import get_story_payload  # lazy to avoid circular import

    chat_id = update.effective_chat.id

    try:
        payload = await get_story_payload()
    except Exception as e:
        logger.error("[telegram] get_story_payload failed: %s", e)
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Error al obtener historia. Inténtalo de nuevo.",
        )
        return

    if not payload:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ No se encontró ninguna historia. Prueba otra vez.",
        )
        return

    await _send_story_to(chat_id, payload, context.bot)
    logger.info("[telegram] Story sent to chat %s: %s", chat_id, payload.headline[:50])


async def _button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle "📰 Nueva historia" button clicks.
    Sends the story to the user's DM (not to the channel).
    """
    query = update.callback_query
    await query.answer()  # Acknowledge immediately

    user_id = query.from_user.id

    try:
        from ..story_service import get_story_payload  # lazy to avoid circular import
        payload = await get_story_payload()
    except Exception as e:
        logger.error("[telegram] button callback failed: %s", e)
        await query.edit_message_text(
            text="❌ Error al obtener historia. Inténtalo de nuevo.",
        )
        return

    if not payload:
        await context.bot.send_message(
            chat_id=user_id,
            text="❌ No se encontró ninguna historia. Prueba otra vez.",
        )
        return

    # Send to the user's DM
    await _send_story_to(user_id, payload, context.bot)
    logger.info(
        "[telegram] Button story sent to DM %s: %s",
        user_id,
        payload.headline[:50],
    )

    # Edit the original message to confirm
    await query.edit_message_text(
        text="✅ ¡Historia enviada a tu DM! Revisa tus mensajes privados.",
    )


async def _start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — welcome the user and explain how to use the bot."""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=(
            "👋 *BBC Mundo Bot*\n\n"
            "Usa /historia para recibir una historia nueva de BBC Mundo.\n"
            "O haz clic en el botón si lo ves en un canal."
        ),
        parse_mode="Markdown",
    )


# ── TelegramAdapter ─────────────────────────────────────────────────────────

class TelegramAdapter(PlatformAdapter):
    """
    Telegram-specific posting via python-telegram-bot (v22+).

    Supports:
    - Channel posting (with inline "Nueva historia" button)
    - DM posting (direct, no button)
    - /historia slash command
    - "Nueva historia" button → sends story to user's DM
    """

    def __init__(
        self,
        bot_token: str,
        channel_chat_id: str | None = None,
        allow_dm: bool = True,
    ):
        self.bot_token = bot_token
        self.channel_chat_id = channel_chat_id
        self.allow_dm = allow_dm
        self._app: Application | None = None

    # ── Bot lifecycle ───────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the Telegram bot (long polling). Call once at startup."""
        if not self.bot_token:
            logger.warning("[telegram] TELEGRAM_BOT_TOKEN not set — Telegram disabled")
            return

        self._app = (
            Application.builder()
            .token(self.bot_token)
            .build()
        )

        # Register handlers
        self._app.add_handler(CommandHandler("historia", _historia_command))
        self._app.add_handler(CommandHandler("start", _start_command))
        self._app.add_handler(
            CallbackQueryHandler(
                _button_callback,
                pattern="nh",  # "nueva historia" prefix
            )
        )

        # Post button anchor to channel if configured
        if self.channel_chat_id:
            await self._post_channel_anchor()

        await self._app.run_polling(drop_pending_updates=True)
        logger.info("[telegram] Bot started")

    async def _post_channel_anchor(self) -> None:
        """Post the persistent button message to the channel."""
        if not self._app or not self.channel_chat_id:
            return

        keyboard = [[InlineKeyboardButton("📰 Nueva historia", callback_data="nh")]]
        markup = InlineKeyboardMarkup(keyboard)

        try:
            await self._app.bot.send_message(
                chat_id=int(self.channel_chat_id),
                text=(
                    "📰 *BBC Mundo — Nueva Historia*\n\n"
                    "Haz clic en el botón para recibir una historia en tu DM."
                ),
                parse_mode="Markdown",
                reply_markup=markup,
            )
            logger.info(
                "[telegram] Channel anchor posted to %s", self.channel_chat_id
            )
        except Exception as e:
            logger.warning("[telegram] Could not post channel anchor: %s", e)

    async def stop(self) -> None:
        """Stop the bot."""
        if self._app:
            await self._app.stop()
            logger.info("[telegram] Bot stopped")

    # ── PlatformAdapter interface ───────────────────────────────────────────

    async def post_channel(self, payload: StoryPayload) -> str:
        """
        Post headline to the configured Telegram channel with a "Nueva historia" button.
        Returns a synthetic message ID (uuid4 — Telegram doesn't need real IDs for DMs).
        """
        if not self._app or not self.channel_chat_id:
            # Fall back to DM
            return await self._post_dm(payload)

        keyboard = [[InlineKeyboardButton("📰 Nueva historia", callback_data="nh")]]
        markup = InlineKeyboardMarkup(keyboard)

        msg = await self._app.bot.send_message(
            chat_id=int(self.channel_chat_id),
            text=f"📰 *{payload.headline}*\n\nHaz clic para recibir la historia.",
            parse_mode="Markdown",
            reply_markup=markup,
        )
        return str(msg.message_id)

    async def create_thread(self, payload: StoryPayload, channel_msg_id: str) -> str:
        """
        Telegram forum channels support topics/threads.
        Creates a new thread in the forum channel.
        Returns the thread_id (topic_id in Telegram terms).
        """
        if not self._app or not self.channel_chat_id:
            return "dm"

        try:
            # Telegram forum topic ID (for forum channels, custom_id is the topic_id)
            # We use the message_id as the thread identifier
            msg = await self._app.bot.send_message(
                chat_id=int(self.channel_chat_id),
                text=f"🧵 *{payload.topic_title}*",
                parse_mode="Markdown",
                message_thread_id=int(channel_msg_id) if channel_msg_id.isdigit() else None,
            )
            return str(msg.message_id)
        except Exception as e:
            logger.warning("[telegram] create_thread failed: %s", e)
            return "dm"

    async def post_thread(self, thread_id: str, payload: StoryPayload) -> None:
        """Post the story content to the specified thread/DM."""
        chat_id = int(thread_id) if thread_id.isdigit() else None
        if chat_id:
            await _send_story_to(chat_id, payload, self._app.bot)
        else:
            # DM fallback
            pass

    async def add_reaction(self, channel_msg_id: str) -> None:
        """React to the channel message with ✅."""
        if not self._app or not self.channel_chat_id:
            return
        try:
            await self._app.bot.send_message(
                chat_id=int(self.channel_chat_id),
                text="✅",
            )
        except Exception as e:
            logger.warning("[telegram] add_reaction failed: %s", e)

    # ── Convenience ────────────────────────────────────────────────────────

    async def send_story(self, payload: StoryPayload) -> None:
        """
        Full Telegram flow:
        1. Post headline to channel/DM (with button)
        2. User clicks button → story arrives in DM

        For DM-only: just send the full story directly.
        """
        if self.channel_chat_id:
            await self.post_channel(payload)
        else:
            # No channel configured — send full story to a default DM
            # Caller should pass the target chat_id, but for now send to
            # channel_chat_id if set, otherwise this is a no-op (callers use DM directly)
            logger.warning(
                "[telegram] send_story called without channel_chat_id — "
                "use send_story_to_dm() or post_channel()"
            )

    async def send_story_to_dm(self, user_id: int, payload: StoryPayload) -> None:
        """Send a story directly to a user's DM."""
        if not self._app:
            logger.warning("[telegram] Bot not started — cannot send DM")
            return
        await _send_story_to(user_id, payload, self._app.bot)

    # ── Sent-stories tracking (inherited from PlatformAdapter) ────────────
