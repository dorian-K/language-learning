import json
import logging
import os
import re

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is missing! Please add it to your .env file.")

# Initialize the DeepSeek Client using the OpenAI SDK
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",  # This tells the SDK to talk to DeepSeek, not OpenAI
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
        model="deepseek-v4-pro",
        messages=messages,
        response_format={"type": "json_object"},
        # Temperature is ignored by deepseek-reasoner (it enforces its own logical temperature)
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
