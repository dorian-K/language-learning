import base64
import json
import os
import random
import shutil
import sqlite3
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import genanki

from llm import invoke_llm

INPUT = os.path.join(
    os.path.dirname(__file__), "../anki/Refold ES1K.apkg"
)  # Place your audio files here
OUTPUT_FOLDER = os.path.join(
    os.path.dirname(__file__), "../anki/", os.path.basename(INPUT)[:-5]
)  # Transcriptions will be saved here
PROMPT_FILE = os.path.join(
    os.path.dirname(__file__), "extract_from_anki_repackage_prompt.txt"
)  # File containing the instructions

MAX_CONCURRENT_CALLS = 200


def load_apkg_to_genanki(apkg_path):
    """
    Extracts an .apkg file and reconstructs it into genanki objects.
    Returns: a list of genanki.Deck objects, and a list of media file paths.
    """
    extract_dir = tempfile.mkdtemp()

    # 1. Unzip the .apkg file
    with zipfile.ZipFile(apkg_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    if os.path.exists(os.path.join(extract_dir, "collection.anki21b")):
        raise ValueError(
            "This .apkg file appears to be from a newer Anki version (2.1.21+). Export the deck in compatibility mode and try again"
        )

    # 2. Locate the database
    db_path = os.path.join(extract_dir, "collection.anki21")
    if not os.path.exists(db_path):
        db_path = os.path.join(extract_dir, "collection.anki2")

    if not os.path.exists(db_path):
        raise FileNotFoundError("Could not find SQLite database in the .apkg file.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 3. Read Decks and Models from the 'col' table
    cursor.execute("SELECT decks, models FROM col LIMIT 1")
    col_data = cursor.fetchone()
    decks_json = json.loads(col_data[0])
    models_json = json.loads(col_data[1])

    # --- Reconstruct genanki.Model objects ---
    genanki_models = {}
    for model_id_str, m_data in models_json.items():
        # Anki stores fields and templates with lots of meta-data.
        # We extract just what genanki needs.
        fields = [{"name": f["name"]} for f in m_data.get("flds", [])]
        templates = [
            {"name": t["name"], "qfmt": t["qfmt"], "afmt": t["afmt"]}
            for t in m_data.get("tmpls", [])
        ]

        model = genanki.Model(
            model_id=int(m_data["id"]),
            name=m_data["name"],
            fields=fields,
            templates=templates,
            css=m_data.get("css", ""),
        )
        genanki_models[int(m_data["id"])] = model
        print(f"Loaded model: {model} with ID {model.model_id}")

    # --- Reconstruct genanki.Deck objects ---
    genanki_decks = {}
    for deck_id_str, d_data in decks_json.items():
        deck_id = int(d_data["id"])
        # Anki auto-generates a Default deck (id: 1) which we usually ignore
        # unless it has cards. We'll load them all just in case.
        deck = genanki.Deck(deck_id=deck_id, name=d_data["name"])
        genanki_decks[deck_id] = deck
        print(f"Loaded deck: {deck.name} with ID {deck.deck_id}")
    # --- Reconstruct genanki.Note objects ---
    # We join notes with cards to find out which deck the note belongs to.
    cursor.execute("""
        SELECT n.id, n.guid, n.mid, n.flds, n.tags, c.did 
        FROM notes n 
        JOIN cards c ON n.id = c.nid 
        GROUP BY n.id
    """)

    for row in cursor.fetchall():
        note_id, guid, mid, raw_flds, tags_raw, did = row

        # In Anki's DB, fields are separated by \x1f (Unit Separator)
        fields_list = raw_flds.split("\x1f")

        # Tags are space-separated, usually padded with spaces: ' tag1 tag2 '
        tags_list = [t for t in tags_raw.strip().split(" ") if t]

        note = genanki.Note(
            model=genanki_models[mid], fields=fields_list, tags=tags_list, guid=guid
        )
        # print(f"Loaded note {note.fields} with tags {note.tags} into deck ID {did}")
        # print()

        # Add the note to its corresponding deck
        if did in genanki_decks:
            genanki_decks[did].add_note(note)
        else:
            raise ValueError(f"Deck ID {did} not found for note ID {note_id}")

    conn.close()

    # --- Reconstruct Media Files ---
    # The 'media' file in the root is a JSON dict mapping integer strings to filenames
    media_json_path = os.path.join(extract_dir, "media")
    media_files = []

    if os.path.exists(media_json_path):
        with open(media_json_path, encoding="utf-8") as f:
            media_map = json.load(f)

        # Rename the integer files (e.g., '0', '1') back to their original extensions
        for numeric_id, original_filename in media_map.items():
            old_path = os.path.join(extract_dir, numeric_id)
            new_path = os.path.join(extract_dir, original_filename)
            if os.path.exists(old_path):
                shutil.move(old_path, new_path)
                media_files.append(new_path)

    # Clean up decks that have no notes in them (like the auto-generated Default deck)
    active_decks = [deck for deck in genanki_decks.values() if len(deck.notes) > 0]

    return active_decks, media_files


def note_to_llm_str(note):
    """
    Converts a genanki.Note object into a string format suitable for LLM input.
    This is a simplified example and may need to be expanded based on the actual note structure and model requirements.
    """
    model = note.model
    deny_fields = ["Index", "word_audio", "sentence_audio"]
    fields_str = "\n".join(
        [
            f"{field['name']}: {value}"
            for field, value in zip(model.fields, note.fields)
            if field["name"] not in deny_fields
        ]
    )
    assert len(note.tags) == 0, (
        "This function does not currently handle tags. Please remove tags from the note or extend this function to include them."
    )

    return fields_str


def b64_encode(s):
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("utf-8")


def process_note(note_id, note_str, extraction_rules, output_filepath):
    if os.path.exists(output_filepath):
        print(f"Note {note_id} already processed, skipping.")
        return
    time.sleep(0.1 * MAX_CONCURRENT_CALLS * random.random())

    print(f"Processing note with GUID: {note_id}")
    user_message = f"{extraction_rules}\n\nHere is the note data:\n{note_str}\n\nExtract the relevant information into a JSON format as per the instructions."
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
            ],
            print_reasoning=(MAX_CONCURRENT_CALLS <= 2),
        )  # Only print reasoning if we're doing a single call for easier debugging

        # Save the formatted JSON to the output folder
        with open(output_filepath, "w", encoding="utf-8") as out_f:
            json.dump(vocab_data, out_f, indent=4, ensure_ascii=False)

        print(f"Successfully extracted {len(vocab_data)} words for {note_id}.json.")

        # print(f"DeepSeek's reasoning process for {filename}:\n{reasoning_process}\n")

    except json.JSONDecodeError:
        print(f"JSON Error on {note_id}: The model output invalid JSON format. Check the raw text.")
        # Saves the broken text so you can see what went wrong
        with open(output_filepath.replace(".json", "_ERROR.txt"), "w", encoding="utf-8") as err_f:
            err_f.write(raw_output)

    except Exception as e:
        print(f"API Error on {note_id}: {e}")
        print(f"Its filename: {output_filepath}")
        return False

    return True


if __name__ == "__main__":
    decks, media_files = load_apkg_to_genanki(INPUT)

    print(f"Extracted {len(decks)} decks and {len(media_files)} media files from {INPUT}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(
            f"Could not find {PROMPT_FILE}. Please create it and paste the prompt rules inside."
        )

    with open(PROMPT_FILE, encoding="utf-8") as f:
        extraction_rules = f.read()

    # Example: Convert notes to LLM input format
    #
    #    print(f"Deck: {deck.name}")
    #    for note in deck.notes[5:]:
    #         process_note(note, extraction_rules)
    #        break
    for deck in decks:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CALLS) as executor:
            # Submit all tasks to the executor
            future_to_file = {
                executor.submit(
                    process_note,
                    b64_encode(note.guid),
                    note_to_llm_str(note),
                    extraction_rules,
                    output_filepath=os.path.join(OUTPUT_FOLDER, f"{b64_encode(note.guid)}.json"),
                ): note
                for note in deck.notes
            }

            # As each thread completes, print its result
            for future in as_completed(future_to_file):
                result = future.result()
                # print(result)
