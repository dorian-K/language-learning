from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re

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


def invoke_llm(messages, print_reasoning=False, json=True):
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        response_format={"type": "json_object"},
        # Temperature is ignored by deepseek-reasoner (it enforces its own logical temperature)
    )

    # The final JSON response
    raw_output = response.choices[0].message.content

    # (Optional: If you want to see the model's internal "thoughts", you can access them via:)
    reasoning_process = response.choices[0].message.reasoning_content
    if print_reasoning:
        print(f"Model's reasoning process:\n{reasoning_process}\n")

    # Clean and parse the text into an actual Python Dictionary
    clean_json_str = extract_json_from_text(raw_output)
    vocab_data = json.loads(clean_json_str)

    return vocab_data
