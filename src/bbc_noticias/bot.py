"""
BBC Noticias Bot — main entrypoint (sync).

Fetches BBC Mundo RSS → selects most relevant story for Dorian via OpenRouter →
fetches & simplifies the article → sends to Discord and/or Telegram.

Run once (how it works with cron):
    python -m src.bbc_noticias.bot

For cron: schedule this script to run once a day via your system's cron,
docker-compose cron, or OpenClaw's built-in cron.
"""

import dataclasses
import logging

from dotenv import load_dotenv

load_dotenv()

from src.bbc_noticias.adapters.base import StoryPayload  # noqa: E402
from src.bbc_noticias.config import load  # noqa: E402
from src.bbc_noticias.notifier import _build_story_text, send_article  # noqa: E402
from src.bbc_noticias.rss import fetch_stories  # noqa: E402
from src.bbc_noticias.scraper import fetch_article  # noqa: E402
from src.bbc_noticias.selector import select_best_story  # noqa: E402
from src.bbc_noticias.sent_stories import filter_unsent  # noqa: E402
from src.bbc_noticias.simplifier import simplify  # noqa: E402


def run() -> bool:
    cfg = load()
    log = logging.getLogger("bot")
    log.info("BBC Noticias Bot — model: %s, dry_run: %s", cfg.openrouter_model, cfg.dry_run)

    # 1. Fetch RSS
    log.info("[1/4] Fetching RSS feeds...")
    stories = fetch_stories(max_age_hours=cfg.max_age_hours)
    log.info("  Found %d stories from the last %dh.", len(stories), cfg.max_age_hours)
    if not stories:
        log.warning("  No recent stories found. Exiting.")
        return False

    # Limit for the selection prompt
    stories = stories[: cfg.max_stories_for_selection]

    # Filter out stories we've already sent (unless we're in dry-run mode)
    unsent_links = set(filter_unsent([s["link"] for s in stories]))
    stories = [s for s in stories if s["link"] in unsent_links]
    if not stories:
        print("  All recent stories have already been sent. Exiting.")
        return False

    # 2. Select best story
    log.info("[2/4] Selecting most relevant story...")
    from src.bbc_noticias.llm import LLM

    try:
        llm = LLM()
    except ValueError as e:
        log.error("  LLM init failed: %s", e)
        return False

    best = select_best_story(stories, llm)
    if not best:
        log.warning("  Could not select a story. Exiting.")
        return False
    log.info("  Selected: %s | %s", best["title"], best["link"])

    # 3. Fetch & simplify article
    log.info("[3/4] Fetching article and simplifying text...")
    article_text = fetch_article(best["link"])
    if not article_text:
        log.error("  Could not fetch article text. Exiting.")
        return False
    log.info("  Fetched %d characters.", len(article_text))

    article_dict = {
        "title": best["title"],
        "text": article_text,
        "url": best["link"],
    }
    result = simplify(article_dict, llm)
    log.info(
        "  Simplified (summary=%d chars, bullets=%d chars)",
        len(result["summary"]),
        len(result["bullets"]),
    )

    emoji = "📰"
    headline = f"{emoji} **{best['title']}**"
    payload = StoryPayload(
        headline=headline,
        summary=result["summary"],
        bullets=result["bullets"],
        url=best["link"],
        topic_title=best["title"],
    )

    if cfg.dry_run:
        log.info("[dry-run] Would have sent — title: %s | link: %s", best["title"], best["link"])
        log.debug("  Summary preview: %s...", result["summary"][:500])
        return True

    # 4. Notify
    log.info("[4/4] Sending article to channels...")

    # Discord — use _build_story_text for properly formatted message
    discord_ok = False
    if cfg.discord_webhook_url:
        discord_ok = send_article(
            title=best["title"],
            original_url=best["link"],
            simplified_text=_build_story_text(payload),
        )["discord"]
        log.info("  Discord: %s", "✅" if discord_ok else "❌")

    # Mark as sent so button-handler bots don't re-send it
    if discord_ok:
        try:
            from src.bbc_noticias.queue import enqueue_story

            enqueue_story(dataclasses.asdict(payload))
        except Exception as e:
            log.warning("  Queue: ❌ (%s)", e)

    return discord_ok


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    run()
