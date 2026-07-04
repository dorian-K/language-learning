"""Generate the Spanish-numbers Anki deck data (deterministic, no LLM).

Numbers are a closed, structural system: a handful of memorised atoms plus a few
composition/gender rules covers every value. So instead of brute-forcing 1..1,000,000
we emit a curated set that teaches the *building blocks* and the *rules*, spelling each
value exactly with ``spanish_numbers`` (see that module for why we don't use an LLM here).

Each number becomes two cards — ``numeral_to_es`` (production) and ``es_to_numeral``
(recognition) — written as a 2-object JSON array into ``anki/numbers/`` (one file per
item, idempotent-friendly, matching the conjugation-folder convention). ``make_anki_deck.py``
reads that folder into the flat ``Lt::Numbers`` deck.

Run as ``python src/generate_numbers.py`` from the repo root (top-level import style).
"""

import json
import os
import re

from spanish_numbers import apocope, cardinal, feminine, format_numeral, ordinal

OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "../anki/numbers")


def _slug(text: str) -> str:
    """Filesystem-safe slug for a card key (keeps it human-readable)."""
    return re.sub(r"[^0-9a-zA-Zñáéíóú]+", "_", text.lower()).strip("_")


def _card(direction, numeral, spanish, category, example_es, example_en, example_de, notes):
    return {
        "direction": direction,
        "numeral": numeral,
        "spanish": spanish,
        "category": category,
        "example_es": example_es,
        "example_en": example_en,
        "example_de": example_de,
        "notes": notes,
    }


def make_pair(numeral, spanish, category, example_es="", example_en="", example_de="", notes=""):
    """Return (file_key, [forward_card, reverse_card]) for one number."""
    key = f"{category}_{_slug(numeral)}"
    cards = [
        _card(
            "numeral_to_es",
            numeral,
            spanish,
            category,
            example_es,
            example_en,
            example_de,
            notes,
        ),
        _card(
            "es_to_numeral",
            numeral,
            spanish,
            category,
            example_es,
            example_en,
            example_de,
            notes,
        ),
    ]
    return key, cards


def build_cards():
    """Build every (file_key, cards) pair across the five categories."""
    pairs = []

    # --- Building blocks: the atoms you must memorise -------------------------
    building = list(range(0, 30))  # 0-15 unique, 16-19 & 21-29 written-together
    building += [30, 40, 50, 60, 70, 80, 90]  # tens
    building += [100, 200, 300, 400, 500, 600, 700, 800, 900]  # hundreds (incl. irregulars)
    building += [1000, 1000000, 1000000000]  # mil, millón, mil millones
    building_set = set(building)
    for n in building:
        pairs.append(make_pair(format_numeral(n), cardinal(n), "building_block"))

    # --- Composition examples: exercise the joining rules --------------------
    composition = [
        21,
        31,
        42,
        47,
        68,
        99,
        101,
        115,
        128,
        256,
        777,
        999,
        1234,
        2015,
        21000,
        100000,
        999999,
        2500000,
    ]
    for n in composition:
        if n in building_set:
            continue  # avoid duplicating an atom
        pairs.append(make_pair(format_numeral(n), cardinal(n), "composition"))

    # --- Gender & apocope: the tricky grammar of numbers ---------------------
    # (numeral-with-noun, spanish-agreeing-form, notes, es/en/de example sentence)
    gender = [
        (
            "1 libro",
            apocope(cardinal(1)) + " libro",
            "Apócope: uno → un ante sustantivo masculino.",
            "Tengo un libro.",
            "I have one book.",
            "Ich habe ein Buch.",
        ),
        (
            "21 libros",
            apocope(cardinal(21)) + " libros",
            "Apócope: veintiuno → veintiún ante sustantivo masculino.",
            "Hay veintiún libros.",
            "There are twenty-one books.",
            "Es gibt einundzwanzig Bücher.",
        ),
        (
            "21 casas",
            feminine(cardinal(21)) + " casas",
            "Femenino: veintiuno → veintiuna.",
            "Hay veintiuna casas.",
            "There are twenty-one houses.",
            "Es gibt einundzwanzig Häuser.",
        ),
        (
            "31 días",
            apocope(cardinal(31)) + " días",
            "Apócope: … y uno → … y un.",
            "El mes tiene treinta y un días.",
            "The month has thirty-one days.",
            "Der Monat hat einunddreißig Tage.",
        ),
        (
            "100 euros",
            cardinal(100) + " euros",
            "Ciento se apocopa a cien ante un sustantivo (y ante mil).",
            "Cuesta cien euros.",
            "It costs one hundred euros.",
            "Es kostet hundert Euro.",
        ),
        (
            "200 casas",
            feminine(cardinal(200)) + " casas",
            "Femenino: doscientos → doscientas.",
            "Hay doscientas casas.",
            "There are two hundred houses.",
            "Es gibt zweihundert Häuser.",
        ),
        (
            "500 personas",
            feminine(cardinal(500)) + " personas",
            "Femenino: quinientos → quinientas.",
            "Vinieron quinientas personas.",
            "Five hundred people came.",
            "Fünfhundert Personen kamen.",
        ),
        (
            "1.000.000 de personas",
            cardinal(1000000) + " de personas",
            "Un millón lleva 'de' ante el sustantivo.",
            "Un millón de personas.",
            "One million people.",
            "Eine Million Menschen.",
        ),
    ]
    for numeral, spanish, notes, es, en, de in gender:
        pairs.append(make_pair(numeral, spanish, "gender_apocope", es, en, de, notes))

    # --- Real-world: years and prices ----------------------------------------
    for year in [1492, 1808, 1936, 1995, 2024]:
        pairs.append(make_pair(str(year), cardinal(year), "real_world", notes="Año."))

    prices = [
        (3, 50, "3,50 €"),
        (19, 99, "19,99 €"),
        (1, 25, "1,25 €"),
    ]
    for euros, cents, numeral in prices:
        spanish = f"{cardinal(euros)} con {cardinal(cents)}"
        pairs.append(
            make_pair(numeral, spanish, "real_world", notes="Precio: la coma se lee 'con'.")
        )

    # --- Ordinals -------------------------------------------------------------
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100]:
        notes = ""
        if n in (1, 3):
            notes = f"Apócope ante sustantivo masculino: {ordinal(n, apocopate=True)}."
        pairs.append(make_pair(f"{n}.º", ordinal(n), "ordinal", notes=notes))

    return pairs


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pairs = build_cards()
    for key, cards in pairs:
        path = os.path.join(OUTPUT_FOLDER, f"{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=4, ensure_ascii=False)
    print(f"Wrote {len(pairs)} number files ({len(pairs) * 2} cards) to {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
