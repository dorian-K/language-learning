"""
Story service — full pipeline: fetch RSS → select best → fetch article → simplify → format.

Pipeline: fetch RSS → filter unsent → select best → fetch article → simplify → format
"""

import asyncio
import logging

from .adapters.base import StoryPayload
from .llm import LLM
from .prompts import DORIAN_PROFILE, STORY_SELECTION_PROMPT
from .rss import fetch_stories
from .scraper import fetch_article
from .sent_stories import filter_unsent
from .simplifier import simplify

logger = logging.getLogger(__name__)


def _call_simplify(article_dict: dict, llm: LLM) -> dict | None:
    """
    Wrapper that catches TypeError (None content from LLM) and returns None
    instead of propagating. This lets the caller retry or handle gracefully.
    """
    try:
        return simplify(article_dict, llm)
    except TypeError as e:
        if "None" in str(e):
            logger.warning("[simplifier] LLM returned None on simplify, will retry: %s", e)
            return None
        raise


def _format_headline(story: dict) -> str:
    """Add emoji + bold title for Discord/Telegram display."""
    emoji = "📰"
    title = story.get("title", "").strip()
    return f"{emoji} **{title}**"


async def _select_best_story(stories: list[dict], llm: LLM) -> dict | None:
    """Ask LLM which story is most relevant for Dorian."""
    if not stories:
        return None

    story_lines = []
    for i, s in enumerate(stories, 1):
        story_lines.append(
            f"[{i}] {s['title']}\n"
            f"    Fuente: {s['source']} | Fecha: {s['pub_date']}\n"
            f"    {s['description'][:300]}"
        )

    story_list = "\n\n".join(story_lines)
    prompt = STORY_SELECTION_PROMPT.format(profile=DORIAN_PROFILE, story_list=story_list)

    selected_title = llm.complete(
        system="You are a helpful news curation assistant.",
        user=prompt,
        temperature=0.3,
    )

    for s in stories:
        if s["title"].strip() == selected_title.strip():
            return s
        if selected_title.strip().lower() in s["title"].strip().lower():
            return s

    logger.warning(
        "[story_service] Could not match title '%s', falling back to first story.", selected_title
    )
    return stories[0]


async def simplify_story(story: dict) -> StoryPayload:
    """
    Fetch article, simplify text, format for posting.
    Returns a StoryPayload ready to be sent to adapters.
    """
    loop = asyncio.get_event_loop()
    article_text = await loop.run_in_executor(None, fetch_article, story["link"])
    if not article_text:
        raise ValueError(f"Could not fetch article: {story['link']}")

    llm = LLM()

    article_dict = {
        "title": story.get("title", ""),
        "text": article_text,
        "url": story.get("link", ""),
    }

    simplified = await loop.run_in_executor(None, _call_simplify, article_dict, llm)
    retries = 2
    for attempt in range(1, retries + 1):
        if simplified is not None:
            break
        logger.warning(
            "[simplify_story] Simplification returned None (attempt %s/%s), retrying: %s",
            attempt, retries, story["link"],
        )
        simplified = await loop.run_in_executor(None, _call_simplify, article_dict, llm)

    if simplified is None:
        raise TypeError(f"Simplification failed after {retries} attempts for: {story['link']}")

    headline = _format_headline(story)

    return StoryPayload(
        headline=headline,
        summary=simplified["summary"],
        bullets=simplified["bullets"],
        text=simplified["text"],
        url=story["link"],
        topic_title=story["title"],
    )


async def get_story_payload(max_age_hours: int = 48) -> StoryPayload | None:
    """
    Full pipeline: fetch → select → simplify → format.
    Returns a StoryPayload or None if no suitable story found.
    """
    llm = LLM()

    # 1. Fetch RSS
    stories = await asyncio.get_event_loop().run_in_executor(None, fetch_stories, max_age_hours)
    if not stories:
        logger.warning("[story_service] No stories found in RSS feeds.")
        return None

    # 2. Filter already-sent stories
    unsent_links = set(filter_unsent([s["link"] for s in stories]))
    stories = [s for s in stories if s["link"] in unsent_links]
    if not stories:
        logger.info("[story_service] All stories already sent.")
        return None

    # 3. Select best story
    best = await _select_best_story(stories, llm)
    if not best:
        logger.warning("[story_service] Could not select a story.")
        return None

    logger.info("[story_service] Selected: %s", best["title"])

    # 4. Simplify and format
    return await simplify_story(best)
