import json
import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing! Please add it to your .env file.")

# GLM-5.2 via OpenRouter's Exacto routing preset (":exacto" picks the most accuracy-optimized
# provider endpoint for the model). Overridable via ANKI_LLM_MODEL — deliberately NOT OPENROUTER_MODEL,
# which the bbc_noticias bot uses for a different model (avoids one .env value hijacking both).
MODEL = os.getenv("ANKI_LLM_MODEL", "z-ai/glm-5.2:exacto")

# Initialize the OpenRouter client using the OpenAI SDK
client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",  # This tells the SDK to talk to OpenRouter, not OpenAI
)
# ==========================================


def extract_json_from_text(text):
    """
    Reasoning models (like DeepSeek-R1) sometimes ignore the 'no markdown' rule
    and wrap their output in ```json ... ``` blocks. This helper function
    robustly finds and extracts the JSON array regardless of how it's formatted.
    """
    match = re.search(r"```(?:json)?\s*(\[\s*\{.*?\}\s*\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # If no markdown block is found, strip whitespace and hope it's raw JSON
    return text.strip()


def invoke_llm(messages, print_reasoning=False, want_json=True):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        # Enable the model's reasoning tokens via OpenRouter's unified reasoning param.
        extra_body={"reasoning": {"enabled": True}},
    )

    # The final JSON response
    raw_output = response.choices[0].message.content

    # Note: reasoning_content is not available in openai>=2.x with deepseek-reasoner
    # Reasoning is instead delivered via a separate stream/event — see openai docs
    if print_reasoning:
        print("Reasoning not available via message.reasoning_content in openai>=2.x")

    # Clean and parse the text into an actual Python Dictionary
    clean_json_str = extract_json_from_text(raw_output)
    try:
        vocab_data = json.loads(clean_json_str)
    except json.JSONDecodeError as e:
        logger.error("[llm] extract_vocab JSON parse error: %s | raw: %s", e, clean_json_str[:200])
        raise

    return vocab_data
