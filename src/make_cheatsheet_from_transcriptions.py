import os
import glob
import json
import re

import time
import random

from openai import OpenAI
from dotenv import load_dotenv
import better_exchook

better_exchook.setup_all()

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = os.path.join(
    os.path.dirname(__file__), "../transcriptions/lt"
)  # Place your audio files here
OUTPUT_FOLDER = os.path.join(
    os.path.dirname(__file__), "../cheatsheet/lt5"
)  # Transcriptions will be saved here
OUTPUT_COMPACTED_FOLDER = os.path.join(
    os.path.dirname(__file__), "../cheatsheet/lt5_compacted"
)  # Compacted cheatsheets will be saved here
PROMPT_FILE = os.path.join(
    os.path.dirname(__file__), "cheatsheet_prompt.txt"
)  # File containing the instructions


API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing! Please add it to your .env file.")

# Initialize the DeepSeek Client using the OpenAI SDK
client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# MODEL = "moonshotai/kimi-k2.5"
MODEL = "google/gemini-3.1-flash-lite-preview"
COMPACTING_MODEL = "google/gemini-3-flash-preview"


def process_single_transcript(filepath, previous_cheatsheet, extraction_rules):

    filename = os.path.basename(filepath)
    base_name = os.path.splitext(filename)[0]
    output_filepath = os.path.join(OUTPUT_FOLDER, f"{base_name}_cheat.md")

    # Skip if already processed (allows you to stop/resume the script without paying twice)
    if os.path.exists(output_filepath):
        print(f"Skipping {filename} (Already processed).")

        return output_filepath

    print(f"Processing {filename}...")

    with open(filepath, "r", encoding="utf-8") as f:
        transcript = f.read()

    if previous_cheatsheet is None:
        previous_cheatsheet_content = (
            "<no previous cheatsheet - this is the first transcript>"
        )
    else:
        with open(previous_cheatsheet, "r", encoding="utf-8") as f:
            previous_cheatsheet_content = f.read()

    # Build the Prompt using the "Sandwich Method" discussed earlier
    user_message = (
        "I am going to provide a transcript of two speakers learning Spanish. "
        "Read it carefully. I will give you extraction instructions after the transcript.\n\n"
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        "The cheatsheet from the previous transcript is below. \n\n"
        f"<cheatsheet>\n{previous_cheatsheet_content}\n</cheatsheet>\n\n"
        f"Now, based on the transcript above (from {filename}) and the current cheatsheet content, strictly follow these instructions:\n\n"
        f"{extraction_rules}"
        "Now output the new cheatsheet in markdown format, following the exact structure and formatting rules provided. Do not add any explanations or extra text, just the markdown content of the cheatsheet."
    )

    # Call LLM
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert linguistics AI and Spanish teacher. You output markdown. Make no mistakes.",
            },
            {"role": "user", "content": user_message},
        ],
        extra_body={
            "reasoning": {"enabled": False},
            "chat_template_kwargs": {"thinking": False},
        },
        stream=False,
        max_tokens=100000,
    )
    raw_output = response.choices[0].message.content

    print(f"finish reason: {response.choices[0].finish_reason}")
    assert response.choices[0].finish_reason == "stop", (
        f"Model output was cut off {response.choices[0].finish_reason}. Consider increasing max_tokens or checking for errors."
    )

    # Save the formatted JSON to the output folder
    with open(output_filepath, "w", encoding="utf-8") as out_f:
        out_f.write(raw_output)

    print(f"Successfully extracted {len(raw_output)} characters for {filename}.")

    if len(raw_output) > 100000:
        raise ValueError(
            f"Output is too long! This might indicate the model is not following instructions properly. Check the output file for {filename} to see if it looks correct."
        )
    if False:  # print reasoning
        reasoning_process = response.choices[0].message.reasoning_content
        print(f"Model's reasoning process for {filename}:\n{reasoning_process}\n")

    return output_filepath


def compact_cheatsheets(files, out_filepath):
    # here we take 10 cheatsheets, and have llms compact them into 1, following the same rules as before. This is to create a "master cheatsheet" that summarizes all the previous ones, which can be used as the "previous_cheatsheet" context for future transcripts, allowing the model to have a more comprehensive understanding of all past content without having to read through every single previous cheatsheet in detail.
    combined_cheatsheet_content = ""
    for f in files:
        with open(f, "r", encoding="utf-8") as in_f:
            combined_cheatsheet_content += (
                f"<{os.path.basename(f)}>\n" + in_f.read() + "\n</>\n\n"
            )

    combined_cheatsheet_content += """
Now, based on the combined content of these 10 cheatsheets, create a single, compact cheatsheet that summarizes all the key information. Follow the same formatting and structure rules as before, but focus on condensing the information while retaining all important details. The output should be in markdown format. Do not add any explanations or extra text, just the markdown content of the compacted cheatsheet.
    """

    response = client.chat.completions.create(
        model=COMPACTING_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert linguistics AI and Spanish teacher. You output markdown. Make no mistakes.",
            },
            {"role": "user", "content": combined_cheatsheet_content},
        ],
        extra_body={
            "reasoning": {"enabled": True},
            "chat_template_kwargs": {"thinking": True},
        },
        stream=False,
        max_tokens=200000,
    )
    assert response.choices[0].finish_reason == "stop", (
        f"Model output was cut off {response.choices[0].finish_reason}. Consider increasing max_tokens or checking for errors."
    )
    compacted_cheatsheet = response.choices[0].message.content

    # Save the compacted cheatsheet
    with open(out_filepath, "w", encoding="utf-8") as out_f:
        out_f.write(compacted_cheatsheet)


def process_transcripts():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_COMPACTED_FOLDER, exist_ok=True)

    # Read the master prompt instructions
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(
            f"Could not find {PROMPT_FILE}. Please create it and paste the prompt rules inside."
        )

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        extraction_rules = f.read()

    # Find all .txt transcriptions
    txt_files = glob.glob(os.path.join(INPUT_FOLDER, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in {INPUT_FOLDER}.")
        return

    print(f"Found {len(txt_files)} transcripts. Starting llm Extraction...")

    txt_files = [t for t in txt_files if os.path.basename(t)]  # Skip hidden files
    prev_cheat = None
    i = 0
    for filepath in sorted(txt_files):
        i += 1
        prev_cheat = process_single_transcript(filepath, prev_cheat, extraction_rules)
        # if i > 10:
        #   break

    print("\nAll transcripts processed!")

    # now we compact
    for i in range(9):
        print(f"Compacting cheatsheets batch {i + 1}...")
        batch_files = list(sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "*.md"))))[
            i * 10 : (i + 1) * 10
        ]
        if not batch_files:
            print(f"No more cheatsheets to compact in batch {i + 1}.")
            break
        out_filepath = os.path.join(OUTPUT_COMPACTED_FOLDER, f"compacted_{i + 1}.md")
        compact_cheatsheets(batch_files, out_filepath)
        print(f"Batch {i + 1} compacted into {out_filepath}.")


if __name__ == "__main__":
    process_transcripts()
