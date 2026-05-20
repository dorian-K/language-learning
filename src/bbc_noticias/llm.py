"""
OpenRouter LLM client — wraps OpenAI-compatible API for BBC Noticias bot.
Uses openrouter/auto (or configured model) via the OpenAI SDK.
"""

import logging
import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class LLM:
    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file or set the environment variable."
            )
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE,
            http_client=httpx.Client(
                timeout=httpx.Timeout(180.0, connect=10.0),
            ),
        )
        self.model = OPENROUTER_MODEL

    def complete(
        self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 4000
    ) -> str:
        """Send a chat completion and return the text response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=httpx.Timeout(120.0),
            extra_body={"reasoning": {"effort": "low"}},
        )
        content = response.choices[0].message.content
        if content is None:
            # Dump full response structure for debugging
            logger.error(
                "[llm] LLM returned None content. Full response: %s | Choice: %s | Finish reason: %s",
                response,
                response.choices[0],
                response.choices[0].finish_reason,
            )
            refusal = getattr(response.choices[0].message, "refusal", None)
            raise TypeError(f"LLM returned None (refusal: {refusal})")
        if not isinstance(content, str):
            logger.error("[llm] LLM returned non-string type: %s", type(content))
            raise TypeError(f"LLM returned {type(content).__name__}, expected str")
        return content.strip()

    def complete_json(self, system: str, user: str, temperature: float = 0.3) -> dict:
        """Same as complete() but parses the response as JSON."""
        import json
        import re

        text = self.complete(system, user, temperature)
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("[llm] LLM output was not valid JSON: %s\nRaw: %s", e, text[:500])
            raise
