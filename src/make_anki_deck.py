import json
import os
import random

import genanki

from conjugation_table import build_conjugation_lookup, paradigm_key, render_conjugation_table
from spoken_sentence import spoken_sentence
from spoken_word import spoken_word
from tts import find_audio

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
        "deck_naming": "priority",
        "deck_name": "Conjugations",
    },
    {
        "folder": os.path.join(os.path.dirname(__file__), "../anki/regular_verbs"),
        "model_type": "conjugation",
        "deck_naming": "flat",
        "deck_name": "Regular Conjugations",
    },
    {
        "folder": os.path.join(os.path.dirname(__file__), "../anki/numbers"),
        "model_type": "numbers",
        "deck_naming": "flat",
        "deck_name": "Numbers",
    },
]

OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "../anki")

# The ~12 most frequent Spanish irregulars. Cards for these go into the High Priority conjugation
# subdeck; the long tail goes into Low Priority (see deck_naming="priority"). Split is
# organization-only — every card is still built, with full audio. Matched case-insensitively on the
# card's `infinitive` field (none of these carry accents, so str.lower() is enough).
HIGH_PRIORITY_IRREGULARS = {
    "ser",
    "estar",
    "ir",
    "haber",
    "tener",
    "hacer",
    "poder",
    "decir",
    "ver",
    "dar",
    "saber",
    "querer",
}


def conjugation_tier(card):
    """ "High Priority" if the card's verb is a top-frequency irregular, else "Low Priority"."""
    infinitive = (card.get("infinitive", "") or "").strip().lower()
    return "High Priority" if infinitive in HIGH_PRIORITY_IRREGULARS else "Low Priority"


def tense_display_name(raw_tense):
    """Normalized display name for a tense string (also the paradigm grouping key).

    Legacy data spells the same tense several ways ("condicional" vs "indicativo/condicional",
    "imperativo afirmativo" vs "imperativo/afirmativo"); TENSE_DESCRIPTIONS maps every variant to a
    single display name, so grouping the conjugation table by it keeps each paradigm whole while
    still separating genuinely-different tenses ("indicativo/presente" vs "subjuntivo/presente").
    """
    return TENSE_DESCRIPTIONS.get(raw_tense, (raw_tense.replace("_", " ").replace("/", " - "), ""))[
        0
    ]


# Audio clips, keyed by a content hash of the spoken text (see tts.audio_stem):
#   - number words     -> anki/numbers/media/   (generate_number_audio.py)
#   - example sentences -> anki/sentence_audio/  (generate_sentence_audio.py) — conjugation decks
#   - vocab headwords   -> anki/word_audio/      (generate_word_audio.py) — Refold ES1K vocab deck
# Each package bundles only the clips its own cards reference (accumulated per-package below),
# so a deck with no audio present ships silently, unchanged.
NUMBERS_MEDIA_DIR = os.path.join(os.path.dirname(__file__), "../anki/numbers/media")
SENTENCE_MEDIA_DIR = os.path.join(os.path.dirname(__file__), "../anki/sentence_audio")
WORD_MEDIA_DIR = os.path.join(os.path.dirname(__file__), "../anki/word_audio")


def sound_suffix(text, media_dir, media):
    """Return ``" [sound:<file>]"`` if a clip for ``text`` exists in ``media_dir``, else ``""``.

    Duolingo-style autoplay: Anki plays a card side's ``[sound:]`` tag automatically, so callers
    append this to the field that carries the Spanish text — it fires on show when Spanish is on
    the front, on flip when Spanish is on the back. The referenced file is added to ``media`` so
    the packaging step bundles exactly the clips this deck uses. No clip present ⇒ ``""`` (silent).
    """
    text = (text or "").strip()
    if not text:
        return ""
    audio = find_audio(media_dir, text)
    if not audio:
        return ""
    media.add(os.path.join(media_dir, audio))
    return f" [sound:{audio}]"


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
     font-size: 24px;
     font-style: italic;
     line-height: 1.5;
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
.tense-hint {
     margin-top: 16px;
     font-size: 0.9em;
     color: rgb(101, 68, 233);
     cursor: pointer;
}
.nightMode .tense-hint {
     color: rgb(153, 128, 255);
}
.tense-reveal {
     font-size: 1.1em;
     font-weight: bold;
     color: rgb(200, 80, 80);
     margin-bottom: 12px;
}
.nightMode .tense-reveal {
     color: rgb(255, 150, 150);
}
.conj-table {
     margin-top: 20px;
}
.conj-table .ct-title {
     font-size: 0.75em;
     text-transform: uppercase;
     letter-spacing: 0.05em;
     color: rgb(101, 68, 233);
     font-weight: bold;
     margin-bottom: 6px;
}
.nightMode .conj-table .ct-title {
     color: rgb(153, 128, 255);
}
.conj-table table {
     border-collapse: collapse;
     width: 100%;
     max-width: 320px;
}
.conj-table td {
     padding: 3px 12px;
     border-bottom: 1px solid rgb(230, 224, 250);
}
.nightMode .conj-table td {
     border-color: rgb(40, 28, 66);
}
.conj-table .ct-person {
     color: rgb(120, 110, 150);
     text-align: right;
     white-space: nowrap;
     font-size: 0.9em;
}
.nightMode .conj-table .ct-person {
     color: rgb(150, 140, 180);
}
.conj-table .ct-form {
     font-weight: 600;
}
.conj-table .ct-current {
     background-color: rgba(133, 102, 255, 0.15);
}
.conj-table .ct-current .ct-form {
     color: rgb(75, 50, 174);
}
.nightMode .conj-table .ct-current .ct-form {
     color: rgb(153, 128, 255);
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

TENSE_DESCRIPTIONS = {
    "presente": ("Presente", "Present tense - ongoing actions, habits, facts"),
    "presente (subjuntivo)": (
        "Presente de subjuntivo",
        "Present subjunctive - wishes, emotions, uncertainty",
    ),
    "pretérito indefinido": ("Pretérito indefinido", "Simple past - completed actions in the past"),
    "pretérito_imperfecto": (
        "Pretérito imperfecto",
        "Imperfect - ongoing/past habits, descriptions, ongoing background actions",
    ),
    "futuro": ("Futuro simple", "Simple future - future actions, predictions"),
    "indicativo/presente": ("Presente", "Present tense - ongoing actions, habits, facts"),
    "indicativo/pretérito_indefinido": (
        "Pretérito indefinido",
        "Simple past - completed actions in the past",
    ),
    "indicativo/pretérito_imperfecto": (
        "Pretérito imperfecto",
        "Imperfect - ongoing/past habits, descriptions",
    ),
    "indicativo/futuro": ("Futuro simple", "Simple future - future actions, predictions"),
    "indicativo/condicional": (
        "Condicional",
        "Conditional - hypothetical actions, polite requests",
    ),
    "condicional": ("Condicional", "Conditional - hypothetical actions, polite requests"),
    "subjuntivo/presente": (
        "Presente de subjuntivo",
        "Present subjunctive - wishes, emotions, uncertainty",
    ),
    "subjuntivo/imperfecto": (
        "Imperfecto de subjuntivo",
        "Imperfect subjunctive - hypothetical, past uncertainty",
    ),
    "subjuntivo imperfecto": (
        "Imperfecto de subjuntivo",
        "Imperfect subjunctive - hypothetical, past uncertainty",
    ),
    "subjuntivo_presente": (
        "Presente de subjuntivo",
        "Present subjunctive - wishes, emotions, uncertainty",
    ),
    "subjuntivo_imperfecto": (
        "Imperfecto de subjuntivo",
        "Imperfect subjunctive - hypothetical, past uncertainty",
    ),
    "imperativo afirmativo": (
        "Imperativo afirmativo",
        "Positive commands - telling someone to do something",
    ),
    "imperativo/afirmativo": (
        "Imperativo afirmativo",
        "Positive commands - telling someone to do something",
    ),
    "imperativo/negativo": (
        "Imperativo negativo",
        "Negative commands - telling someone NOT to do something",
    ),
    "imperfecto de subjuntivo": (
        "Imperfecto de subjuntivo",
        "Imperfect subjunctive - hypothetical, past uncertainty",
    ),
    "pretérito imperfecto": (
        "Pretérito imperfecto",
        "Imperfect - ongoing/past habits, descriptions",
    ),
    # Bare-mood variants some legacy cards use. Without these they hit the fallback and get a
    # differently-cased display name ("pretérito indefinido"), which splits the paradigm table in
    # two (see tense_display_name / build_conjugation_lookup).
    "pretérito_indefinido": ("Pretérito indefinido", "Simple past - completed actions in the past"),
    "subjuntivo presente": (
        "Presente de subjuntivo",
        "Present subjunctive - wishes, emotions, uncertainty",
    ),
}

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
        {"name": "Conjugation_Table"},
    ],
    templates=[
        {
            "name": "Conjugation Forward",
            "qfmt": """
        <div class="flashcard">
            <div class="special-label">[Conjugation]</div>
            <div class="word">{{Front_Word}}</div>
            <div class="example-sentence">{{Front_Sentence}}</div>
            <div class="tense-hint">{{hint:Meta_Tags}}</div>
        </div>
        """,
            "afmt": """
        <div class="flashcard">
            <div class="special-label">[Conjugation]</div>
            <div class="word">{{Front_Word}}</div>
            <div class="example-sentence">{{Front_Sentence}}</div>
            <hr id="answer" />
            <div class="tense-reveal">{{Meta_Tags}}</div>
            <div class="definition">{{Back_Word}}</div>
            <div class="sentence-translation">{{Back_Sentence}}</div>
            {{Conjugation_Table}}
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
            {{Conjugation_Table}}
        </div>
        """,
        },
    ],
    css=CONJUGATION_CSS,
)

NUMBERS_MODEL_ID = random.Random("ES_Numbers").randrange(1 << 30, 1 << 31)
numbers_model = genanki.Model(
    NUMBERS_MODEL_ID,
    "ES_Numbers",
    fields=[
        {"name": "Front_Word"},
        {"name": "Front_Sentence"},
        {"name": "Back_Word"},
        {"name": "Back_Sentence"},
        {"name": "Meta_Tags"},
    ],
    templates=[
        {
            "name": "Number Card",
            "qfmt": """
        <div class="flashcard">
            <div class="special-label">[Number]</div>
            <div class="word">{{Front_Word}}</div>
            <div class="example-sentence">{{Front_Sentence}}</div>
            <div class="tense-hint">{{hint:Meta_Tags}}</div>
        </div>
        """,
            "afmt": """
        <div class="flashcard">
            <div class="special-label">[Number]</div>
            <div class="word">{{Front_Word}}</div>
            <div class="example-sentence">{{Front_Sentence}}</div>
            <hr id="answer" />
            <div class="definition">{{Back_Word}}</div>
            <div class="sentence-translation">{{Back_Sentence}}</div>
            <div class="tense-reveal">{{Meta_Tags}}</div>
        </div>
        """,
        },
    ],
    css=CONJUGATION_CSS,
)


def get_deck_for_config(config, base_name, decks_by_name):
    config_type = config["deck_naming"]
    if config_type == "level":
        full_name = f"Lt::level{base_name}"
    elif config_type == "priority":
        # Nest the tier (base_name = "High Priority" / "Low Priority") under the parent deck.
        subdeck = config.get("deck_name", os.path.basename(config["folder"]))
        full_name = f"Lt::{subdeck}::{base_name}"
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


def process_vocab_card(card, deck, media):
    direction = card.get("direction")
    sentence_es = (card.get("example_sentence_es", "") or "").strip()
    # Audio speaks the Spanish headword (not the example sentence); attach it to whichever side
    # carries the Spanish word so it autoplays there (front for spanish→target, back otherwise).
    sound = sound_suffix(spoken_word(card), WORD_MEDIA_DIR, media)

    if direction in ["spanish_to_target", "spanish_sentence_to_target"]:
        front_word = f"{card.get('cue_spanish', '')}{sound}"
        front_sentence = f'"{sentence_es}"'

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
            f"<span class='lang-label'>ES:</span> {format_list(card.get('target_es', []))}{sound}"
        )
        back_sentence = f"<span class='lang-label'>ES:</span> {sentence_es}"
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


def process_conjugation_card(card, deck, media):
    direction = card.get("direction")
    infinitive = card.get("infinitive", "")
    raw_tense = card.get("tense", "")
    person = card.get("person", "")
    conjugated = card.get("conjugated_form", "")

    tense_info = TENSE_DESCRIPTIONS.get(
        raw_tense, (raw_tense.replace("_", " ").replace("/", " - "), "")
    )
    tense_name = tense_info[0]
    tense_desc = tense_info[1]
    meta = f"{tense_name}, {person}"

    sentence_es = (card.get("example_sentence_es", "") or "").strip()
    # spoken_sentence fills the [infinitive] blank on forward cards with conjugated_form, so the
    # audio speaks the real conjugated sentence (not "volver"). Reverse cards: no bracket, no-op.
    sound = sound_suffix(spoken_sentence(card), SENTENCE_MEDIA_DIR, media)

    # Full-paradigm table for the back. Only on the FORWARD (conjugation-practice) card — that's
    # where the learner just tried to recall this form, so the whole paradigm reinforces it. The
    # reverse card is translation practice, where the table is off-topic; showing it on both would
    # also duplicate the same ~500-byte table across every note for no benefit. `_conj_forms` is
    # injected in process_json_files from the sibling per-person cards; absent in unit tests that
    # build one card in isolation, in which case the table renders empty.
    table_html = ""

    if direction == "conjugation_forward":
        # Front keeps the [infinitive] blank silent; the corrected audio rides on the answer word
        # (Back_Word renders only in afmt) so it autoplays on flip, not before recall.
        # Show the target tense on the front word: conjugation practice supplies the tense as an
        # input, so the learner isn't left guessing it from context (inferring the tense from a
        # sentence is a translation skill — that's what the reverse card trains). The usage
        # description stays behind the click-hint (Meta_Tags) as an optional reminder.
        front_word = f"{infinitive} ({person}) · {tense_name}"
        front_sentence = sentence_es
        back_word = f"{conjugated}{sound}"
        back_sentence = (
            f"<span class='lang-label'>EN:</span> {card.get('example_sentence_en', '')}<br>"
            f"<span class='lang-label'>DE:</span> {card.get('example_sentence_de', '')}"
        )
        meta = tense_desc
        table_html = render_conjugation_table(card.get("_conj_forms") or {}, person, tense_name)
    elif direction == "conjugation_reverse":
        front_word = f"{conjugated} ({infinitive})"
        front_sentence = f"{sentence_es}{sound}"
        back_word = (
            f"<span class='lang-label'>EN:</span> {card.get('example_sentence_en', '')}<br>"
            f"<span class='lang-label'>DE:</span> {card.get('example_sentence_de', '')}"
        )
        back_sentence = ""
        meta = f"{tense_name}, {person}\n{tense_desc}"
    else:
        return None

    key = f"{direction}|{infinitive}|{raw_tense}|{person}"
    note_guid = genanki.guid_for(key)
    note = genanki.Note(
        model=conjugation_model,
        fields=[front_word, front_sentence, back_word, back_sentence, meta, table_html],
        guid=note_guid,
    )
    deck.add_note(note)
    return note


CATEGORY_LABELS = {
    "building_block": "Building block",
    "composition": "Composition",
    "gender_apocope": "Gender / apocope",
    "real_world": "Real-world",
    "ordinal": "Ordinal",
}


def process_numbers_card(card, deck, media):
    direction = card.get("direction")
    numeral = card.get("numeral", "")
    spanish = card.get("spanish", "")

    # Duolingo-style autoplay: attach one [sound:] to the Spanish side only. Anki auto-plays
    # audio on the shown side, so it fires on flip for numeral->es (Spanish on back) and on
    # show for es->numeral (Spanish on front). Only tag when the clip actually exists, so the
    # deck still builds silently before any audio has been generated.
    spanish_field = f"{spanish}{sound_suffix(spanish, NUMBERS_MEDIA_DIR, media)}"

    if direction == "numeral_to_es":
        front_word, back_word = numeral, spanish_field
    elif direction == "es_to_numeral":
        front_word, back_word = spanish_field, numeral
    else:
        return None

    example_es = card.get("example_es", "")
    front_sentence = f'"{example_es}"' if example_es else ""
    if card.get("example_en") or card.get("example_de"):
        back_sentence = (
            f"<span class='lang-label'>EN:</span> {card.get('example_en', '')}<br>"
            f"<span class='lang-label'>DE:</span> {card.get('example_de', '')}"
        )
    else:
        back_sentence = ""

    label = CATEGORY_LABELS.get(card.get("category", ""), card.get("category", ""))
    notes = card.get("notes", "")
    meta = f"{label}\n{notes}" if notes else label

    key = f"{direction}|{numeral}|{spanish}"
    note_guid = genanki.guid_for(key)
    note = genanki.Note(
        model=numbers_model,
        fields=[front_word, front_sentence, back_word, back_sentence, meta],
        guid=note_guid,
    )
    deck.add_note(note)
    return note


def process_json_files():
    packages = {}

    for config in INPUT_CONFIGS:
        folder = config["folder"]
        model_type = config["model_type"]
        process_func = {
            "vocab": process_vocab_card,
            "numbers": process_numbers_card,
        }.get(model_type, process_conjugation_card)

        if not os.path.exists(folder):
            print(f"Warning: Input folder {folder} not found.")
            continue

        deck_name = config.get("deck_name") or os.path.basename(folder)
        if deck_name not in packages:
            packages[deck_name] = {"decks": {}, "config": config, "media": set()}
        media = packages[deck_name]["media"]

        entries = {}
        num_skipped = 0

        for filepath in [f for f in os.listdir(folder) if f.endswith(".json")]:
            filepath = os.path.join(folder, filepath)
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                for card in data:
                    if model_type == "vocab":
                        key = f"{card.get('direction')}|{card.get('cue_spanish', '')}|{card.get('cue_en', '')}|{card.get('cue_de', '')}"
                    elif model_type == "numbers":
                        key = f"{card.get('direction')}|{card.get('numeral', '')}|{card.get('spanish', '')}"
                    else:
                        key = f"{card.get('direction')}|{card.get('infinitive', '')}|{card.get('tense', '')}|{card.get('person', '')}"

                    if key in entries:
                        num_skipped += 1
                        existing = entries[key]["card"]
                        for field in existing:
                            if isinstance(existing[field], list) and isinstance(
                                card.get(field), list
                            ):
                                existing_set = set(existing[field])
                                new_items = [
                                    item for item in card.get(field, []) if item not in existing_set
                                ]
                                existing[field].extend(new_items)
                            elif isinstance(existing[field], str) and isinstance(
                                card.get(field), str
                            ):
                                if existing[field].lower() == card.get(field).lower():
                                    continue
                                elif "sentence" in field or "notes" in field:
                                    existing[field] += "\n" + card.get(field)
                            else:
                                pass
                    else:
                        level = (
                            card.get("mandatory_level", "Uncategorized")
                            if model_type == "vocab"
                            else None
                        )
                        entries[key] = {"card": card, "level": level}

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

        # Conjugation decks: reconstruct each verb+tense's full paradigm from its sibling per-person
        # cards and attach it to every card, so the back can render the whole table (see
        # conjugation_table.py). Vocab/numbers have no paradigm and are left untouched.
        if model_type == "conjugation":
            lookup = build_conjugation_lookup(
                (entry["card"] for entry in entries.values()), tense_key=tense_display_name
            )
            for entry in entries.values():
                c = entry["card"]
                c["_conj_forms"] = lookup.get(
                    paradigm_key(c.get("infinitive"), c.get("tense"), tense_display_name), {}
                )

        folder_card_count = 0
        if model_type == "vocab":
            by_level = {}
            for entry in entries.values():
                lvl = entry["level"]
                if lvl not in by_level:
                    by_level[lvl] = []
                by_level[lvl].append(entry["card"])
            for lvl, cards in by_level.items():
                deck = get_deck_for_config(config, lvl, packages[deck_name]["decks"])
                for card in cards:
                    note = process_func(card, deck, media)
                    if note:
                        folder_card_count += 1
                print(f"  level{lvl}: {len(cards)} cards")
        elif config["deck_naming"] == "priority":
            by_tier = {}
            for entry in entries.values():
                tier = conjugation_tier(entry["card"])
                by_tier.setdefault(tier, []).append(entry["card"])
            for tier, cards in by_tier.items():
                deck = get_deck_for_config(config, tier, packages[deck_name]["decks"])
                for card in cards:
                    note = process_func(card, deck, media)
                    if note:
                        folder_card_count += 1
                print(f"  {tier}: {len(cards)} cards")
        else:
            deck = get_deck_for_config(config, "Conjugations", packages[deck_name]["decks"])
            for entry in entries.values():
                card = entry["card"]
                note = process_func(card, deck, media)
                if note:
                    folder_card_count += 1

        print(f"Processed {folder}: {folder_card_count} cards, {num_skipped} duplicates skipped")

    print("\n--- Generating Anki Packages ---")
    for deck_name, package_data in packages.items():
        output_file = os.path.join(OUTPUT_FOLDER, f"{deck_name}.apkg")
        # Each package bundles only the audio its own cards referenced (empty ⇒ silent deck).
        media_files = sorted(package_data["media"])
        package = genanki.Package(list(package_data["decks"].values()), media_files=media_files)
        package.write_to_file(output_file)
        suffix = f" (+{len(media_files)} audio)" if media_files else ""
        print(f"Successfully created: {output_file}{suffix}")


if __name__ == "__main__":
    process_json_files()
