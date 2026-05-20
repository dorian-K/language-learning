import glob
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import invoke_llm

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = os.path.join(
    os.path.dirname(__file__), "../transcriptions/lt"
)  # Place your audio files here
OUTPUT_FOLDER = os.path.join(
    os.path.dirname(__file__), "../vocab/lt"
)  # Transcriptions will be saved here

PROMPT_FILE = os.path.join(
    os.path.dirname(__file__), "vocab_extract_prompt.txt"
)  # File containing the instructions

MAX_CONCURRENT_CALLS = 100  # Adjust based on your system's capabilities and API rate limits

# Load environment variables


def process_single_transcript(filepath, extraction_rules):
    # wait a random amount
    time.sleep(
        0.5 * MAX_CONCURRENT_CALLS * random.random()
    )  # Random delay between 1 and 3 seconds to avoid hitting rate limits

    filename = os.path.basename(filepath)
    base_name = os.path.splitext(filename)[0]
    output_filepath = os.path.join(OUTPUT_FOLDER, f"{base_name}_vocab.json")

    # Skip if already processed (allows you to stop/resume the script without paying twice)
    if os.path.exists(output_filepath):
        print(f"Skipping {filename} (Already processed).")
        return

    print(f"Processing {filename}...")

    with open(filepath, encoding="utf-8") as f:
        transcript = f.read()

    # Build the Prompt using the "Sandwich Method" discussed earlier
    user_message = (
        "I am going to provide a transcript of two speakers learning Spanish. "
        "Read it carefully. I will give you extraction instructions after the transcript.\n\n"
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        "Now, based on the transcript above, strictly follow these instructions:\n\n"
        f"{extraction_rules}"
    )

    raw_output = ""
    try:
        # Call DeepSeek-R1 (deepseek-reasoner)
        vocab_data = invoke_llm(
            [
                {
                    "role": "system",
                    "content": "You are an expert linguistics AI and Spanish teacher. You output strict JSON arrays.",
                },
                {"role": "user", "content": user_message},
            ]
        )

        # Save the formatted JSON to the output folder
        with open(output_filepath, "w", encoding="utf-8") as out_f:
            json.dump(vocab_data, out_f, indent=4, ensure_ascii=False)

        print(f"Successfully extracted {len(vocab_data)} words for {filename}.")

        # print(f"DeepSeek's reasoning process for {filename}:\n{reasoning_process}\n")

    except json.JSONDecodeError:
        print(
            f"JSON Error on {filename}: The model output invalid JSON format. Check the raw text."
        )
        # Saves the broken text so you can see what went wrong
        with open(output_filepath.replace(".json", "_ERROR.txt"), "w", encoding="utf-8") as err_f:
            err_f.write(raw_output)

    except Exception as e:
        print(f"API Error on {filename}: {e}")


def process_transcripts():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Read the master prompt instructions
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(
            f"Could not find {PROMPT_FILE}. Please create it and paste the prompt rules inside."
        )

    with open(PROMPT_FILE, encoding="utf-8") as f:
        extraction_rules = f.read()

    # Find all .txt transcriptions
    txt_files = glob.glob(os.path.join(INPUT_FOLDER, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in {INPUT_FOLDER}.")
        return

    print(f"Found {len(txt_files)} transcripts. Starting DeepSeek R1 Extraction...")

    txt_files = [
        t for t in txt_files if os.path.basename(t) not in ["track01.txt"]
    ]  # Skip hidden files
    # for filepath in sorted(txt_files):
    #   process_single_transcript(filepath, extraction_rules)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CALLS) as executor:
        # Submit all tasks to the executor
        future_to_file = {
            executor.submit(process_single_transcript, filepath, extraction_rules): filepath
            for filepath in sorted(txt_files)
        }

        # As each thread completes, print its result
        for future in as_completed(future_to_file):
            result = future.result()
            print(result)

    print("\nAll transcripts processed!")


if __name__ == "__main__":
    process_transcripts()
