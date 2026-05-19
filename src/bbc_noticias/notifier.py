"""
Cron job — fetches BBC Mundo stories and sends them to Discord + Telegram.

Runs periodically (e.g. every 2 hours). Fetches new stories from RSS,
selects the best one via LLM, sends the full story to both platforms.

The actual posting is handled here (not via button handlers like Discord).
Button handlers remain available for on-demand requests via the bot containers.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

from .story_service import fetch_and_pick_story, simplify_story, get_story_payload
from .adapters.base import StoryPayload


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_story_text(payload: StoryPayload) -> str:
    """Format a story as readable plain text (for Telegram)."""
    return (
        f"📰 *{payload.headline}*\n\n"
        f"{payload.summary}\n\n"
        f"{payload.bullets}\n\n"
        f"🔗 {payload.url}"
    )


async def run() -> bool:
    """
    Main entry point for the cron job.
    Returns True if a story was successfully sent to any platform, False otherwise.
    """
    logger.info("[cron] Starting BBC cron job at %s", datetime.now(timezone.utc))

    # Full pipeline: fetch → select → fetch article → simplify → format
    try:
        payload = await get_story_payload(max_age_hours=3)
    except Exception as e:
        logger.error("[cron] get_story_payload failed: %s", e)
        return False

    if not payload:
        logger.info("[cron] No suitable story found in last 3 hours.")
        return False

    logger.info("[cron] Story ready: %s", payload.headline[:60])

    # ── Discord ───────────────────────────────────────────────────────────
    discord_sent = await _send_discord(payload)

    # ── Telegram ─────────────────────────────────────────────────────────
    telegram_sent = await _send_telegram(payload)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info(
        "[cron] Done — discord=%s telegram=%s",
        "✅" if discord_sent else "❌",
        "✅" if telegram_sent is True else ("❌" if telegram_sent is False else "N/A"),
    )

    return discord_sent or (telegram_sent is True)


async def _send_discord(payload: StoryPayload) -> bool:
    """Post story summary to Discord webhook (if configured)."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        return False

    try:
        import httpx  # lazy
        category_emoji = "📰"
        text = (
            f"{category_emoji} *Nueva historia de BBC Mundo*\n\n"
            f"{payload.headline}\n\n"
            f"{payload.summary}\n\n"
            f"{payload.bullets}\n\n"
            f"🔗 {payload.url}"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                webhook_url,
                json={"content": text[:2000]},  # Discord limit
            )
            resp.raise_for_status()
            logger.info("[cron] Discord webhook sent")
            return True
    except Exception as e:
        logger.warning("[cron] Discord webhook failed: %s", e)
        return False


async def _send_telegram(payload: StoryPayload) -> bool | None:
    """Send full story to Telegram channel/DM (if configured)."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    channel_id = os.getenv("TELEGRAM_CHANNEL_ID", "")

    if not bot_token:
        return None  # Not configured, skip silently

    # Prefer channel if set, otherwise use DM chat_id
    target = channel_id or chat_id
    if not target:
        logger.warning("[cron] TELEGRAM_BOT_TOKEN set but no TELEGRAM_CHAT_ID or TELEGRAM_CHANNEL_ID")
        return None

    try:
        from telegram import Bot

        bot = Bot(token=bot_token)
        text = _build_story_text(payload)
        msg = await bot.send_message(
            chat_id=int(target),
            text=text,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        logger.info("[cron] Telegram sent to %s, msg_id=%s", target, msg.message_id)
        return True
    except Exception as e:
        logger.warning("[cron] Telegram send failed: %s", e)
        return False


if __name__ == "__main__":
    success = asyncio.run(run())
    sys.exit(0 if success else 1)


# ── bot.py backward-compatibility shim ─────────────────────────────────────────

def send_article(title: str, original_url: str, simplified_text: str, pub_date: str) -> dict:
    """
    Sync wrapper for backward compatibility with bot.py.
    Posts via webhook to Discord only (Telegram is async-only).
    Returns {"discord": bool, "telegram": bool}.
    """
    logger.info("[send_article] Posting: %s", title)
    result = {"discord": False, "telegram": False}

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if webhook_url:
        try:
            import httpx  # lazy
            payload = {"content": f"**{title}**\n{simplified_text[:1800]}\n\n🔗 {original_url}"}
            resp = httpx.post(webhook_url, json=payload, timeout=10.0)
            resp.raise_for_status()
            result["discord"] = True
            logger.info("  Discord: ✅")
        except Exception as e:
            logger.warning("  Discord: ❌ (%s)", e)

    logger.info("[send_article] Done — discord=%s telegram=%s", result["discord"], result["telegram"])
    return result