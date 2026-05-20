import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from extract_from_anki import MAX_CONCURRENT_CALLS, b64_encode, process_note

INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "../vocab/lt/")
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "../anki/lt/")

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "extract_from_vocab_repackage_prompt.txt")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(
            f"Could not find {PROMPT_FILE}. Please create it and paste the prompt rules inside."
        )

    with open(PROMPT_FILE, encoding="utf-8") as f:
        extraction_rules = f.read()

    file_contents = []
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(INPUT_FOLDER, filename)
        with open(filepath, encoding="utf-8") as f:
            file_contents.append((filename, f.read()))

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CALLS) as executor:
        # Submit all tasks to the executor

        future_to_file = {
            executor.submit(
                process_note,
                b64_encode(filename),
                contents,
                extraction_rules,
                output_filepath=os.path.join(OUTPUT_FOLDER, f"{b64_encode(filename)}.json"),
            ): filename
            for filename, contents in file_contents
        }

        # As each thread completes, print its result
        for future in as_completed(future_to_file):
            result = future.result()
            # print(result)
