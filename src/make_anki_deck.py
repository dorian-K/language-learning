import json
import os
import random

import genanki

VOCAB_SOURCE = os.getenv("VOCAB_SOURCE", "Refold ES1K")

INPUT_CONFIGS = [
    {
        "folder": os.path.join(os.path.dirname(__file__), f"../anki/{VOCAB_SOURCE}"),
        "model_type": "vocab",
        "deck_naming": "level",
    },
    {
        "folder": os.path.join(os.path.dirname(__file__), "../anki/irregular_verbs"),
        "model_type": "conjugation",
        "deck_naming": "flat",
        "deck_name": "Conjugations",
    },
]

OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "../anki")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VOCAB_CSS = """
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

CONJUGATION_CSS = """
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
.special-label {
     color: rgb(200, 80, 80);
     font-size: 0.85em;
     font-weight: bold;
     margin-bottom: 8px;
}
.nightMode .special-label {
     color: rgb(255, 150, 150);
}
"""

VOCAB_MODEL_ID = random.Random("Symmetrical_ES_EN_DE_Vocab").randrange(1 << 30, 1 << 31)
vocab_model = genanki.Model(
    VOCAB_MODEL_ID,
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
    css=VOCAB_CSS,
)

CONJUGATION_MODEL_ID = random.Random("Verb_Conjugation_Cards").randrange(1 << 30, 1 << 31)
conjugation_model = genanki.Model(
    CONJUGATION_MODEL_ID,
    "Verb_Conjugation_Cards",
    fields=[
        {"name": "Front_Word"},
        {"name": "Front_Sentence"},
        {"name": "Back_Word"},
        {"name": "Back_Sentence"},
        {"name": "Meta_Tags"},
    ],
    templates=[
        {
            "name": "Conjugation Forward",
            "qfmt": """
        <div class="flashcard">
            <div class="special-label">[Conjugation]</div>
            <div class="word">{{Front_Word}}</div>
            <div class="example-sentence">{{Front_Sentence}}</div>
        </div>
        """,
            "afmt": """
        <div class="flashcard">
            <div class="special-label">[Conjugation]</div>
            <div class="word">{{Front_Word}}</div>
            <div class="example-sentence">{{Front_Sentence}}</div>
            <hr id="answer" />
            <div class="definition">{{Back_Word}}</div>
            <div class="sentence-translation">{{Back_Sentence}}</div>
        </div>
        """,
        },
        {
            "name": "Conjugation Reverse",
            "qfmt": """
        <div class="flashcard">
            <div class="special-label">[Translation]</div>
            <div class="word">{{Front_Word}}</div>
            <div class="example-sentence">{{Front_Sentence}}</div>
        </div>
        """,
            "afmt": """
        <div class="flashcard">
            <div class="special-label">[Translation]</div>
            <div class="word">{{Front_Word}}</div>
            <div class="example-sentence">{{Front_Sentence}}</div>
            <hr id="answer" />
            <div class="definition">{{Back_Word}}</div>
            <div class="sentence-translation">{{Back_Sentence}}</div>
        </div>
        """,
        },
    ],
    css=CONJUGATION_CSS,
)

decks_by_name = {}


def get_deck_for_config(config, base_name):
    config_type = config["deck_naming"]
    if config_type == "level":
        full_name = f"Lt::level{base_name}"
    else:
        folder_name = os.path.basename(config["folder"])
        subdeck = config.get("deck_name", folder_name)
        full_name = f"Lt::{subdeck}"

    if full_name not in decks_by_name:
        deck_id = random.Random(full_name).randrange(1 << 30, 1 << 31)
        decks_by_name[full_name] = genanki.Deck(deck_id, full_name)
    return decks_by_name[full_name]


def format_list(item_list):
    return ", ".join(item_list) if isinstance(item_list, list) else str(item_list)


def process_vocab_card(card, deck):
    direction = card.get("direction")

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

        back_word = f"<span class='lang-label'>ES:</span> {format_list(card.get('target_es', []))}"
        back_sentence = f"<span class='lang-label'>ES:</span> {card.get('example_sentence_es', '')}"
    else:
        return None

    key = f"{direction}|{card.get('cue_spanish', '')}|{card.get('cue_en', '')}|{card.get('cue_de', '')}"
    note_guid = genanki.guid_for(key)
    note = genanki.Note(
        model=vocab_model,
        fields=[front_word, front_sentence, back_word, back_sentence],
        guid=note_guid,
    )
    deck.add_note(note)
    return note


def process_conjugation_card(card, deck):
    direction = card.get("direction")
    infinitive = card.get("infinitive", "")
    tense = card.get("tense", "")
    person = card.get("person", "")
    conjugated = card.get("conjugated_form", "")

    meta = f"{tense} | {person}"

    if direction == "conjugation_forward":
        front_word = f"{infinitive} ({person})"
        front_sentence = card.get("example_sentence_es", "")
        back_word = conjugated
        back_sentence = (
            f"<span class='lang-label'>EN:</span> {card.get('example_sentence_en', '')}<br>"
            f"<span class='lang-label'>DE:</span> {card.get('example_sentence_de', '')}"
        )
    elif direction == "conjugation_reverse":
        front_word = f"{conjugated} ({infinitive})"
        front_sentence = card.get("example_sentence_es", "")
        back_word = (
            f"<span class='lang-label'>EN:</span> {card.get('example_sentence_en', '')}<br>"
            f"<span class='lang-label'>DE:</span> {card.get('example_sentence_de', '')}"
        )
        back_sentence = ""
    else:
        return None

    key = f"{direction}|{infinitive}|{tense}|{person}"
    note_guid = genanki.guid_for(key)
    note = genanki.Note(
        model=conjugation_model,
        fields=[front_word, front_sentence, back_word, back_sentence, meta],
        guid=note_guid,
    )
    deck.add_note(note)
    return note


def process_json_files():
    total_cards = 0

    for config in INPUT_CONFIGS:
        folder = config["folder"]
        model_type = config["model_type"]
        process_func = process_vocab_card if model_type == "vocab" else process_conjugation_card

        if not os.path.exists(folder):
            print(f"Warning: Input folder {folder} not found.")
            continue

        entries = {}
        num_skipped = 0

        for filepath in [f for f in os.listdir(folder) if f.endswith('.json')]:
            filepath = os.path.join(folder, filepath)
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                for card in data:
                    if model_type == "vocab":
                        key = f"{card.get('direction')}|{card.get('cue_spanish', '')}|{card.get('cue_en', '')}|{card.get('cue_de', '')}"
                    else:
                        key = f"{card.get('direction')}|{card.get('infinitive', '')}|{card.get('tense', '')}|{card.get('person', '')}"

                    if key in entries:
                        num_skipped += 1
                        existing = entries[key]["card"]
                        for field in existing:
                            if isinstance(existing[field], list) and isinstance(card.get(field), list):
                                existing_set = set(existing[field])
                                new_items = [item for item in card.get(field, []) if item not in existing_set]
                                existing[field].extend(new_items)
                            elif isinstance(existing[field], str) and isinstance(card.get(field), str):
                                if existing[field].lower() == card.get(field).lower():
                                    continue
                                elif "sentence" in field or "notes" in field:
                                    existing[field] += "\n" + card.get(field)
                            else:
                                pass
                    else:
                        level = card.get("mandatory_level", "Uncategorized") if model_type == "vocab" else None
                        entries[key] = {"card": card, "level": level}

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

        if model_type == "vocab":
            by_level = {}
            for entry in entries.values():
                lvl = entry["level"]
                if lvl not in by_level:
                    by_level[lvl] = []
                by_level[lvl].append(entry["card"])
            for lvl, cards in by_level.items():
                deck = get_deck_for_config(config, lvl)
                for card in cards:
                    note = process_func(card, deck)
                    if note:
                        total_cards += 1
                print(f"  level{lvl}: {len(cards)} cards")
        else:
            deck = get_deck_for_config(config, "Conjugations")
            for entry in entries.values():
                card = entry["card"]
                note = process_func(card, deck)
                if note:
                    total_cards += 1

        print(f"Processed {folder}: {total_cards} cards, {num_skipped} duplicates skipped")

    print(f"\nTotal unique cards: {total_cards}")

    output_file = f"{VOCAB_SOURCE}.apkg"

    print("\n--- Generating Anki Packages ---")
    output_file = os.path.join(OUTPUT_FOLDER, output_file)
    genanki.Package(list(decks_by_name.values())).write_to_file(output_file)
    print(f"Successfully created: {output_file}")


if __name__ == "__main__":
    process_json_files()
