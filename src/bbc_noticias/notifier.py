"""
Cron job — fetches BBC Mundo stories and publishes them to the shared queue.

Runs periodically (e.g. every 2 hours). Fetches new stories from RSS,
selects the best one via LLM, writes to /app/shared/queue.json.

Discord and Telegram bots subscribe to the queue independently.

Run once:
    python -m src.bbc_noticias.notifier

With cron: the entrypoint runs cron daemon which fires:
    python -m src.bbc_noticias.notifier publish
"""

import asyncio
import dataclasses
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from .adapters.base import StoryPayload
from .story_service import get_story_payload
from . import pubsub

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Cron entrypoint ─────────────────────────────────────────────────────────────


async def run() -> bool:
    """
    Main entry point for the cron job.
    Fetches story → publishes to queue → returns success status.
    """
    logger.info("[cron] Starting BBC cron job at %s", datetime.now(UTC))

    try:
        payload = await get_story_payload(max_age_hours=3)
    except Exception as e:
        logger.error("[cron] get_story_payload failed: %s", e, exc_info=True)
        return False

    if not payload:
        logger.info("[cron] No suitable story found in last 3 hours.")
        return False

    logger.info("[cron] Story ready: %s", payload.headline[:60])

    # Publish to both platforms (each bot consumes independently)
    pubsub.write_to_queue(dataclasses.asdict(payload), "both")
    return True


def main() -> None:
    """Sync wrapper for direct invocation."""
    success = asyncio.run(run())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


# ── Legacy compatibility shims (used by bot.py) ────────────────────────────────


def _build_story_text(payload: StoryPayload) -> str:
    """Format a story as readable plain text (for Telegram)."""
    return f"📰 *{payload.headline}*\n\n{payload.summary}\n\n{payload.bullets}\n\n🔗 {payload.url}"


def send_article(
    title: str, original_url: str, simplified_text: str, pub_date: str | None = None
) -> dict:
    """
    Sync wrapper for sending via Discord webhook.
    Uses _build_story_text to format the message properly.
    Returns {"discord": bool, "telegram": None}.
    """
    logger.info("[send_article] Posting: %s", title)
    result = {"discord": False, "telegram": None}

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if webhook_url:
        try:
            import httpx  # lazy

            payload = {"content": simplified_text[:2000]}
            resp = httpx.post(webhook_url, json=payload, timeout=10.0)
            resp.raise_for_status()
            result["discord"] = True
            logger.info("  Discord: ✅")
        except Exception as e:
            logger.warning("  Discord: ❌ (%s)", e)

    logger.info(
        "[send_article] Done — discord=%s telegram=%s", result["discord"], result["telegram"]
    )
    return result