"""
Telegram adapter — posts BBC stories to Telegram channels and DMs.

Two independent flows:
1. Channel (automatic, daily via cron/MQTT):
   - Cron publishes story → MQTT → TelegramAdapter.send_story()
   - Full simplified story posted directly to TELEGRAM_CHANNEL_ID
   - Deduped via data/channel_sent.txt (sent_stories.py)

2. DM (on-demand, per-user):
   - Any private text message (or /historia command) → next unsent story for that user
   - Deduped per-user via data/dm_sent.json (dm_sent.py)
   - Tracking is independent: the same story can appear in both the channel and a DM

Environment variables:
  TELEGRAM_BOT_TOKEN  — bot token from @BotFather
  TELEGRAM_CHANNEL_ID — channel ID (numeric, e.g. -1001234567890)
"""

import asyncio
import html
import logging
import re

from telegram import Bot, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .. import mqtt
from .base import PlatformAdapter, StoryPayload

logger = logging.getLogger(__name__)


def _md_to_html(text: str) -> str:
    """Convert the Markdown subset produced by the LLM to Telegram HTML."""
    # Escape HTML entities first so article text can't inject tags
    text = html.escape(text)
    # ### / ## / # headings → bold
    text = re.sub(r"^#{1,6}\s*(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    # **bold**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # *italic* (single asterisk, not part of **)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    # ||spoiler|| → Telegram spoiler tag; brackets stay visible on reveal
    text = re.sub(r"\|\|([^|]+)\|\|", r"<tg-spoiler>(\1)</tg-spoiler>", text)
    return text


def _build_story_text(payload: StoryPayload) -> str:
    """Format a story as a readable Telegram message."""
    return f"{payload.headline}\n\n{payload.text}\n\n{payload.bullets}\n\n🔗 {payload.url}"


async def _send_story_to(chat_id: int, payload: StoryPayload, bot: Bot) -> None:
    """Send a formatted story to the given chat, splitting at 4096 chars if needed."""
    text = _md_to_html(_build_story_text(payload))
    max_len = 4096
    if len(text) <= max_len:
        await bot.send_message(
            chat_id=chat_id, text=text, parse_mode="HTML", disable_web_page_preview=True
        )
    else:
        # Split on double newlines to avoid breaking in the middle of a paragraph
        paragraphs = text.split("\n\n")
        chunk = ""
        for para in paragraphs:
            if len(chunk) + len(para) + 2 > max_len:
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode="HTML")
                chunk = para
            else:
                chunk = f"{chunk}\n\n{para}" if chunk else para
        if chunk:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode="HTML")


# ── Handlers ─────────────────────────────────────────────────────────────────


async def _dm_story_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle any private text message or /historia command.
    Fetches the next story this user hasn't seen and sends it to the same chat.
    Uses per-user tracking so each invocation gives a different story.
    """
    from ..story_service import get_story_for_user

    chat_id = update.effective_chat.id  # type: ignore[reportOptionalMemberAccess]
    user_id = update.effective_user.id  # type: ignore[reportOptionalMemberAccess]

    await context.bot.send_message(chat_id=chat_id, text="Buscando historia...")

    try:
        payload = await get_story_for_user(user_id)
    except Exception as e:
        logger.error("[telegram] get_story_for_user failed for %s: %s", user_id, e, exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Error al obtener historia. Inténtalo de nuevo.",
        )
        return

    if not payload:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ No hay historias nuevas disponibles. Prueba mañana.",
        )
        return

    await _send_story_to(chat_id, payload, context.bot)
    logger.info("[telegram] DM story sent to user %s: %s", user_id, payload.headline[:50])


async def _start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — welcome the user."""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,  # type: ignore[reportOptionalMemberAccess]
        text=(
            "👋 *BBC Mundo Bot*\n\n"
            "Escríbeme cualquier mensaje (o usa /historia) para recibir "
            "una historia nueva de BBC Mundo.\n"
            "Cada vez que escribes te mando una historia distinta."
        ),
        parse_mode="Markdown",
    )


# ── TelegramAdapter ─────────────────────────────────────────────────────────


class TelegramAdapter(PlatformAdapter):
    """
    Telegram-specific posting via python-telegram-bot (v22+).

    Channel flow (automatic):  send_story() → full article posted to TELEGRAM_CHANNEL_ID
    DM flow (on-demand):       any private message → _dm_story_handler → per-user next story
    """

    def __init__(
        self,
        bot_token: str,
        channel_chat_id: str | None = None,
    ):
        self.bot_token = bot_token
        self.channel_chat_id = channel_chat_id
        self._app: Application | None = None
        self._callbacks_loop: asyncio.AbstractEventLoop | None = None
        self._mqtt_sub: mqtt.MQTTSubscriber | None = None

    # ── Bot lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the Telegram bot (long polling). Call once at startup."""
        if not self.bot_token:
            logger.warning("[telegram] TELEGRAM_BOT_TOKEN not set — Telegram disabled")
            return

        async def _post_init(app: Application) -> None:
            # Called from inside the running event loop — the only safe place to
            # capture the loop and start the MQTT subscriber. run_polling() creates
            # its own loop via asyncio.run(), so capturing it before run_polling()
            # gives a different (non-running) loop and run_coroutine_threadsafe blocks forever.
            self._callbacks_loop = asyncio.get_running_loop()
            self.start_subscriber()

        self._app = Application.builder().token(self.bot_token).post_init(_post_init).build()

        # On-demand DM: any private text message
        self._app.add_handler(
            MessageHandler(filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND, _dm_story_handler)
        )
        # Also reachable via /historia in any chat
        self._app.add_handler(CommandHandler("historia", _dm_story_handler))
        self._app.add_handler(CommandHandler("start", _start_command))

        self._app.run_polling(drop_pending_updates=True)
        logger.info("[telegram] Bot started")

    async def stop(self) -> None:
        """Stop the bot."""
        if self._app:
            await self._app.stop()
            logger.info("[telegram] Bot stopped")

    # ── PlatformAdapter interface ───────────────────────────────────────────

    async def post_channel(self, payload: StoryPayload) -> str:
        """Post the full simplified story to the configured Telegram channel."""
        assert self._app is not None, "post_channel() called before bot was started"
        assert self.channel_chat_id, "post_channel() requires TELEGRAM_CHANNEL_ID to be set"

        await _send_story_to(int(self.channel_chat_id), payload, self._app.bot)
        logger.info(
            "[telegram] Posted story to channel %s: %s",
            self.channel_chat_id,
            payload.headline[:60],
        )
        return "ok"

    async def create_thread(self, payload: StoryPayload, channel_msg_id: str) -> str | None:
        """Not used in the Telegram flow — Telegram doesn't use threads."""
        return None

    async def post_thread(self, thread_id: str, payload: StoryPayload) -> None:
        """Not used in the Telegram flow."""

    async def add_reaction(self, channel_msg_id: str) -> None:
        """Not used in the Telegram channel flow."""

    # ── Convenience ────────────────────────────────────────────────────────

    async def send_story(self, payload: StoryPayload) -> None:
        """Post the full story to the configured Telegram channel and mark it as sent."""
        assert self.channel_chat_id, (
            "send_story() requires TELEGRAM_CHANNEL_ID — set it in .env"
        )
        await self.post_channel(payload)
        self.mark_sent(payload.url)

    # ── MQTT subscriber ─────────────────────────────────────────────────────

    def start_subscriber(self) -> None:
        """
        Subscribe to the bbc/stories MQTT topic and post stories to the channel as they arrive.
        Must be called from within the running event loop (via Application.post_init).
        """
        from ..adapters.base import StoryPayload

        loop = self._callbacks_loop
        assert loop is not None, (
            "start_subscriber() must be called from Application.post_init, not before run_polling(). "
            "run_polling() creates its own loop via asyncio.run(), so the loop captured before it "
            "is never started and run_coroutine_threadsafe on it blocks forever."
        )
        assert loop.is_running(), (
            "event loop passed to start_subscriber() is not running — "
            "call start_subscriber() from the post_init hook, not from __init__ or before run_polling()"
        )

        async def on_story_async(payload: dict) -> None:
            try:
                story = StoryPayload(**payload)
                await self.send_story(story)
            except Exception as e:
                logger.error("[telegram] Failed to send MQTT story: %s", e, exc_info=True)

        def on_story(payload: dict) -> None:
            try:
                asyncio.run_coroutine_threadsafe(on_story_async(payload), loop).result()
            except Exception as e:
                logger.error("[telegram] Failed to send MQTT story: %s", e, exc_info=True)

        self._mqtt_sub = mqtt.MQTTSubscriber(on_story, client_id="bbc-telegram-sub")
        self._mqtt_sub.start()

    def stop_subscriber(self) -> None:
        """Stop the MQTT subscriber."""
        if self._mqtt_sub:
            self._mqtt_sub.stop()

    async def stop_bot(self) -> None:
        """Stop the bot and MQTT subscriber."""
        self.stop_subscriber()
        if self._app:
            await self._app.stop()
            logger.info("[telegram] Bot stopped")

    # ── Sent-stories tracking (channel-level, inherited from PlatformAdapter) ──
