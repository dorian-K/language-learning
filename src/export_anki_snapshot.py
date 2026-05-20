"""
Export an Anki .apkg deck as JSON snapshot.
Use this after making manual edits in Anki to get the deck back into
a JSON format that can be fed into make_anki_deck.py.

Usage:
    python3 src/export_anki_snapshot.py <path/to/deck.apkg> [output_folder]

Output:
    One .json file per card in the deck, in the same format as
    extract_from_anki.py produces.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
from extract_from_anki import load_apkg_to_genanki, note_to_llm_str


def export_deck(apkg_path: str, output_folder: str = "anki/snapshot") -> int:
    """Export all cards from an .apkg file to JSON files."""
    decks, _ = load_apkg_to_genanki(apkg_path)
    os.makedirs(output_folder, exist_ok=True)

    card_count = 0
    for deck in decks:
        for note in deck.notes:
            # Parse all fields (same as note_to_llm_str but keep field names)
            model = note.model
            fields = {field["name"]: value for field, value in zip(model.fields, note.fields)}

            snapshot = {
                "guid": note.guid,
                "fields": fields,
                "tags": list(note.tags),
                # Dump as LLM string so the content is preserved and comparable
                "llm_str": note_to_llm_str(note),
            }

            out_path = os.path.join(output_folder, f"{note.guid}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4, ensure_ascii=False)
            card_count += 1

    print(f"Exported {card_count} cards to {output_folder}/")
    return card_count


if __name__ == "__main__":
    apkg_path = sys.argv[1] if len(sys.argv) > 1 else "anki/Refold ES1K.apkg"
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "anki/snapshot"
    export_deck(apkg_path, output_folder)
