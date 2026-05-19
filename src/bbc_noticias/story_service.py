"""
Story service — platform-agnostic business logic.

Pipeline: fetch RSS → filter unsent → select best → fetch article → simplify → format
"""

import asyncio
import logging
from typing import Optional

from . import rss
from . import scraper
from . import simplifier
from . import selector
from . import llm
from .queue_service import queue_service
from .adapters.base import StoryPayload


logger = logging.getLogger(__name__)


# Module-level LLM instance (lazy-initialized)
_llm: Optional[llm.LLM] = None


def _get_llm() -> llm.LLM:
    global _llm
    if _llm is None:
        _llm = llm.LLM()
    return _llm


def _format_headline(story: dict) -> str:
    """Build the emoji + bold-title headline from a story dict."""
    category = story.get("category", "").lower()
    emoji = {
        "science":        "🔬",
        "technology":     "💻",
        "business":       "💼",
        "health":         "🏥",
        "entertainment":  "🎬",
        "sports":         "⚽",
        "environment":   "🌍",
        "politics":       "🏛️",
        "world":          "🌍",
    }.get(category, "📰")
    return f"{emoji} **{story['title']}**"


async def fetch_and_pick_story(max_age_hours: int = 48) -> Optional[dict]:
    """
    Fetch recent stories, filter unsent, select best with LLM.
    Returns a single story dict or None.
    """
    loop = asyncio.get_running_loop()
    stories = await loop.run_in_executor(None, rss.fetch_stories, max_age_hours)
    if not stories:
        logger.warning("[story] No stories fetched from RSS")
        return None

    logger.info("[story] Fetched %d stories from RSS", len(stories))

    # Filter to unsent URLs
    urls = [s["link"] for s in stories]
    unsent_urls = await loop.run_in_executor(None, queue_service.filter_unsent, urls)
    unsent_links = set(unsent_urls)
    stories = [s for s in stories if s["link"] in unsent_links]
    logger.info("[story] %d unsent after filter", len(stories))

    if not stories:
        return None

    # LLM selects the best
    selected = await loop.run_in_executor(None, selector.select_best_story, stories, _get_llm())
    if not selected:
        logger.warning("[story] LLM could not select a story")
        return None

    logger.info("[story] LLM selected: %s", selected.get("title", "?"))
    return selected


async def simplify_story(story: dict) -> StoryPayload:
    """
    Fetch article, simplify text, format for posting.
    Returns a StoryPayload ready for any platform adapter.
    """
    loop = asyncio.get_running_loop()

    # Fetch article in background
    article_fetched = await loop.run_in_executor(
        None, scraper.fetch_article, story["link"]
    )

    # Build input dict for simplifier
    article_dict = {
        "title":       story.get("title", ""),
        "description": story.get("description", ""),
        "content":     article_fetched,
        "url":         story["link"],
        "category":    story.get("category", ""),
    }

    # Simplify and format in background
    simplified = await loop.run_in_executor(None, simplifier.simplify, article_dict)
    headline   = _format_headline(story)

    return StoryPayload(
        headline   = headline,
        summary    = simplified["summary"],
        bullets    = simplified["bullets"],
        url        = story["link"],
        topic_title= story["title"],
    )


async def get_story_payload(max_age_hours: int = 48) -> Optional[StoryPayload]:
    """
    Full pipeline: fetch → select → simplify → format.
    Returns a StoryPayload ready to pass to any PlatformAdapter.send_story().
    """
    story = await fetch_and_pick_story(max_age_hours)
    if not story:
        return None
    return await simplify_story(story)
