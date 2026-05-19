"""
Cron job — fetches BBC Mundo stories and enqueues them for the Discord bot.

Runs periodically (e.g. every 2 hours). Fetches new stories from RSS,
selects the best one via LLM, sends to the Discord webhook, and enqueues
it so the Discord button handler can pick it up.

The actual posting to Discord (forum + thread) is handled by
discord_bot.py when a user clicks the "Nueva historia" button.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

from . import queue as _queue
from .story_service import fetch_and_pick_story, simplify_story

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")


async def run() -> bool:
    """
    Main entry point for the cron job.
    Returns True if a story was successfully enqueued, False otherwise.
    """
    logger.info("[cron] Starting BBC cron job at %s", datetime.now(timezone.utc))

    # Fetch + select + simplify — all platform-agnostic
    try:
        story = await fetch_and_pick_story(max_age_hours=3)
    except Exception as e:
        logger.error("[cron] Failed to fetch/pick story: %s", e)
        return False

    if not story:
        logger.info("[cron] No suitable story found in last 3 hours.")
        return False

    logger.info("[cron] Selected story: %s", story.get("title", "?"))

    # Send to Discord webhook (summary notification, not the full post)
    if WEBHOOK_URL:
        await _send_webhook(story)

    # Enqueue so the Discord button handler can pick it up
    _queue.enqueue_story(story)
    logger.info("[cron] Story enqueued: %s", story.get("title", "?"))
    return True


async def _send_webhook(story: dict) -> None:
    """Post a short summary to the Discord webhook (optional notification)."""
    import httpx  # lazy — only needed when WEBHOOK_URL is set

    if not WEBHOOK_URL:
        return

    category_emoji = {
        "science": "🔬", "technology": "💻", "business": "💼",
        "health": "🏥", "entertainment": "🎬", "sports": "⚽",
        "environment": "🌍", "politics": "🏛️", "world": "🌍",
    }.get(story.get("category", "").lower(), "📰")

    payload = {
        "content": (
            f"{category_emoji} **Nueva historia de BBC Mundo**\n"
            f"[{story.get('title', 'Sin título')}]({story.get('link', '')})\n"
            f"_Haz click en el botón 'Nueva historia' en #historias para verla completa._"
        )
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(WEBHOOK_URL, json=payload)
            resp.raise_for_status()
            logger.info("[cron] Webhook sent, status=%s", resp.status_code)
    except Exception as e:
        logger.warning("[cron] Webhook failed: %s — continuing anyway", e)


# ── bot.py compatibility (one-shot sender) ────────────────────────────────────

def send_article(title: str, original_url: str, simplified_text: str, pub_date: str) -> dict:
    """
    Post a simplified article to Discord and Telegram (backward compat for bot.py).
    Returns {"discord": bool, "telegram": bool}.
    """
    logger.info("[send_article] Posting: %s", title)
    result = {"discord": False, "telegram": False}

    if WEBHOOK_URL:
        try:
            import httpx  # lazy
            payload = {
                "content": f"**{title}**\n{simplified_text[:1800]}\n\n🔗 {original_url}"
            }
            resp = httpx.post(WEBHOOK_URL, json=payload, timeout=10.0)
            resp.raise_for_status()
            result["discord"] = True
            logger.info("  Discord: ✅")
        except Exception as e:
            logger.warning("  Discord: ❌ (%s)", e)

    logger.info("[send_article] Done — discord=%s telegram=%s", result["discord"], result["telegram"])
    return result


if __name__ == "__main__":
    success = asyncio.run(run())
    sys.exit(0 if success else 1)
