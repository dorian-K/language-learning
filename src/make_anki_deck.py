import json
import os
import random

import genanki

# Define Paths
if False:
    INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "../anki/Refold ES1K")
else:
    INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "../anki/lt")

OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "../anki")

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 1. Custom CSS adapted from Refold ES1K with additions for dual languages
CSS = """
.card {
     font-family: sans-serif;
     font-size: 16px;
     text-align: left;
     color: rgb(48, 32, 111);
     background-color: rgb(251, 250, 254);
}
.card.nightMode {
     color: rgba(255, 255, 255, 0.85);
     background-color: rgb(11, 7, 22);
}
.flashcard {
     display: block;
     padding: 24px;
     min-height: 200px;
     max-width: 500px;
     margin: 0 auto;
     border-radius: 10px;
     background-color: rgb(255, 255, 255);
     box-shadow: 0 5px 10px -5px rgba(133, 102, 255, 0.4);
}
.nightMode .flashcard {
     background-color: rgb(19, 12, 34);
     box-shadow: 0 5px 10px -5px rgba(0, 0, 0, 0.75);
}
.word {
     color: rgb(75, 50, 174);
     font-size: 28px;
     font-weight: 700;
     margin-bottom: 16px;
     line-height: 1.3;
}
.nightMode .word {
     color: rgb(133, 102, 255);
}
.definition {
     font-size: 22px;
     margin-bottom: 16px;
     line-height: 1.3;
}
.example-sentence {
     margin-top: 16px;
     font-style: italic;
     line-height: 1.4;
}
.sentence-translation {
     margin-top: 12px;
     font-size: 16px;
     line-height: 1.4;
}
hr {
     border: 0;
     border-bottom: 1px solid rgb(216, 208, 249);
     margin: 24px 0;
}
.nightMode hr {
     border-color: rgb(48, 32, 111);
}
.lang-label {
     color: rgb(101, 68, 233);
     font-size: 0.75em;
     font-weight: bold;
     text-transform: uppercase;
     margin-right: 4px;
}
.nightMode .lang-label {
     color: rgb(153, 128, 255);
}
"""


# 2. Define the Universal Model for our dual-direction setup
MODEL_ID = random.Random("Symmetrical_ES_EN_DE_Vocab").randrange(
    1 << 30, 1 << 31
)  # Reusing a stable base ID
unified_model = genanki.Model(
    MODEL_ID,
    "Symmetrical_ES_EN_DE_Vocab",
    fields=[
        {"name": "Front_Word"},
        {"name": "Front_Sentence"},
        {"name": "Back_Word"},
        {"name": "Back_Sentence"},
    ],
    templates=[
        {
            "name": "Vocabulary Card",
            "qfmt": """
        <div class="flashcard">
            <div class="word">{{Front_Word}}</div>
            <div class="example-sentence">{{Front_Sentence}}</div>
        </div>
        """,
            "afmt": """
        <div class="flashcard">
            <div class="word">{{Front_Word}}</div>
            <div class="example-sentence">{{Front_Sentence}}</div>

            <hr id="answer" />

            <div class="definition">{{Back_Word}}</div>
            <div class="sentence-translation">{{Back_Sentence}}</div>
        </div>
        """,
        }
    ],
    css=CSS,
)

# Dictionary to hold the dynamic decks
decks_by_level = {}


def get_deck_for_level(level):
    """Creates or fetches a deck for a specific language level."""
    deck_name = os.path.basename(INPUT_FOLDER).replace("_", " ").title()
    if level not in decks_by_level:
        # Seed the random generator with the level name to ensure stable deck IDs across runs
        deck_id = random.Random(f"{deck_name}_{level}").randrange(1 << 30, 1 << 31)
        deck_name = f"{deck_name}::level{level}"
        decks_by_level[level] = genanki.Deck(deck_id, deck_name)
    return decks_by_level[level]


def format_list(item_list):
    """Helper to join lists of words cleanly."""
    return ", ".join(item_list) if isinstance(item_list, list) else str(item_list)


def process_json_files():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder {INPUT_FOLDER} not found.")
        return

    entries = {}
    num_skipped = 0
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(INPUT_FOLDER, filename)

        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            # Iterate through the two objects in the array
            for index, card in enumerate(data):
                key = f"{card.get('direction')}|{card.get('cue_spanish', '')}|{card.get('cue_en', '')}|{card.get('cue_de', '')}"
                if key in entries:
                    num_skipped += 1
                    # handle duplicate
                    # all data types in both should be the same
                    for field in entries[key]:
                        assert type(entries[key][field]) is type(card.get(field)), (
                            f"Data type mismatch for field '{field}' in file {filename}"
                        )

                    # arrays: append unique items. but keep current order stable
                    # sentence strings: add newline and append if different
                    for field in entries[key]:
                        if isinstance(entries[key][field], list) and isinstance(
                            card.get(field), list
                        ):
                            existing_set = set(entries[key][field])
                            new_items = [
                                item for item in card.get(field, []) if item not in existing_set
                            ]
                            entries[key][field].extend(new_items)
                        elif isinstance(entries[key][field], str) and isinstance(
                            card.get(field), str
                        ):
                            if entries[key][field].lower() == card.get(field).lower():
                                continue
                            elif "sentence" in field or "notes" in field:
                                entries[key][field] += "\n" + card.get(field)
                            elif "level" in field:
                                pass  # ignore...
                            else:
                                print(
                                    f"Warning: Conflicting non-sentence field '{field}' for key '{key}' in file {filename}. Keeping original value."
                                )
                        else:
                            raise ValueError(
                                f"Unsupported data type for field '{field}' in file {filename}. Expected both to be lists or both to be strings."
                            )
                else:
                    entries[key] = card

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    for key, card in entries.items():
        direction = card.get("direction")
        level = card.get("mandatory_level", "Uncategorized")
        # Assign to corresponding deck
        deck = get_deck_for_level(level)

        # Determine fields based on direction
        if direction in ["spanish_to_target", "spanish_sentence_to_target"]:
            front_word = card.get("cue_spanish", "")
            front_sentence = f'"{card.get("example_sentence_es", "")}"'

            back_word = (
                f"<span class='lang-label'>EN:</span> {format_list(card.get('target_en', []))}<br>"
                f"<span class='lang-label'>DE:</span> {format_list(card.get('target_de', []))}"
            )
            back_sentence = (
                f"<span class='lang-label'>EN:</span> {card.get('example_sentence_en', '')}<br>"
                f"<span class='lang-label'>DE:</span> {card.get('example_sentence_de', '')}"
            )

        elif direction in ["target_to_spanish", "target_sentence_to_spanish"]:
            front_word = (
                f"<span class='lang-label'>EN:</span> {card.get('cue_en', '')}<br>"
                f"<span class='lang-label'>DE:</span> {card.get('cue_de', '')}"
            )
            front_sentence = (
                f"<span class='lang-label'>EN:</span> \"{card.get('example_sentence_en', '')}\"<br>"
                f"<span class='lang-label'>DE:</span> \"{card.get('example_sentence_de', '')}\""
            )

            back_word = (
                f"<span class='lang-label'>ES:</span> {format_list(card.get('target_es', []))}"
            )
            back_sentence = (
                f"<span class='lang-label'>ES:</span> {card.get('example_sentence_es', '')}"
            )
        else:
            print(f"Warning: Unrecognized direction '{direction}',skipping card.")
            continue  # Skip if unrecognized format

        # Generate a stable GUID from the filename and the card index
        note_guid = genanki.guid_for(key)

        note = genanki.Note(
            model=unified_model,
            fields=[front_word, front_sentence, back_word, back_sentence],
            guid=note_guid,
        )

        deck.add_note(note)

    print(
        f"Finished processing files. Total unique cards added: {len(entries)}. Resolved {num_skipped} duplicates across files."
    )
    # 4. Export the decks
    print("\n--- Generating Anki Packages ---")
    # for level, deck in decks_by_level.items():
    output_file = os.path.join(OUTPUT_FOLDER, f"{os.path.basename(INPUT_FOLDER)}.apkg")
    genanki.Package(list(decks_by_level.values())).write_to_file(output_file)
    print(f"Successfully created: {output_file}")


if __name__ == "__main__":
    process_json_files()
