"""
Story selector — asks OpenRouter which of today's BBC Mundo stories
is most relevant for Dorian.
"""

import logging

from .llm import LLM
from .prompts import DORIAN_PROFILE, STORY_SELECTION_PROMPT

logger = logging.getLogger(__name__)


def select_best_story(stories: list[dict], llm: LLM) -> dict | None:
    """
    Given a list of story dicts (title, link, description, pub_date, source),
    ask OpenRouter which one is most relevant for Dorian and return it.
    """
    if not stories:
        return None

    story_lines = []
    for idx, s in enumerate(stories, 1):
        story_lines.append(
            f"[{idx}] {s['title']}\n"
            f"    Fuente: {s['source']} | Fecha: {s['pub_date']}\n"
            f"    {s['description'][:300]}"
        )

    story_list = "\n\n".join(story_lines)

    logger.debug("Story list:\n%s", story_list)

    prompt = STORY_SELECTION_PROMPT.format(profile=DORIAN_PROFILE, story_list=story_list)
    selected_title = llm.complete(
        system="You are a helpful news curation assistant.",
        user=prompt,
        temperature=0.3,
    )

    # Match back to a story by title (exact or partial)
    for s in stories:
        if s["title"].strip() == selected_title.strip():
            return s
        # Partial match
        if selected_title.strip().lower() in s["title"].strip().lower():
            return s

    # Fallback: return first
    logger.warning(
        "[selector] Could not match title '%s', falling back to first story.",
        selected_title,
        exc_info=True,
    )
    return stories[0]
