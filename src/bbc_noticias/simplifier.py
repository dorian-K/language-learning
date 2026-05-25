"""
Text simplifier — takes raw Spanish article text and returns a B1-adapted version.
Calls OpenRouter with SIMPLIFY_PROMPT: simplifies sentence structures
and adds English translations for difficult words.
"""

import json
import logging
import re

from .llm import LLM
from .prompts import DORIAN_PROFILE, SIMPLIFY_PROMPT, VOCAB_HARD_LIST

logger = logging.getLogger(__name__)


def simplify(article_dict: dict, llm: LLM) -> dict:
    """
    Given an article dict with 'text', 'url', and 'title' keys, call OpenRouter
    to produce a B1-adapted version with:
      - simpler sentence structures
      - English translations in ||(text)|| format for difficult words
      - a summary and bullet points

    Returns a dict with keys: summary, bullets, text
    """
    article_text = article_dict.get("text", "")

    # Truncate if too long (OpenRouter has context limits and high costs)
    max_chars = 20000
    if len(article_text) > max_chars:
        article_text = article_text[:max_chars] + "\n[... texto truncado ...]"

    prompt = SIMPLIFY_PROMPT.format(
        profile=DORIAN_PROFILE,
        hard_words=VOCAB_HARD_LIST,
        article_text=article_text,
    )

    computed_max_tokens = min(60000, max(16000, 8000 + len(article_text) // 2))
    logger.info("[simplifier] max_tokens=%s for article text len=%s", computed_max_tokens, len(article_text))

    raw = llm.complete(
        system=(
            "You are a Spanish language tutor. Always respond with ONLY valid JSON "
            "matching the required schema. Never add explanations, preambles, "
            "or anything outside the JSON object."
        ),
        user=prompt,
        temperature=0.6,
        max_tokens=computed_max_tokens,
    )

    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("[simplifier] LLM output was not valid JSON: %s\nRaw: %s", e, raw[:500])
        raise

