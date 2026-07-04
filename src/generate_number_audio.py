"""Generate peninsular-Spanish audio clips for the Numbers deck (local, offline).

Reads the number cards in ``anki/numbers/`` and synthesizes one audio clip per **unique**
``spanish`` string (both card directions of a number share a clip) into ``anki/numbers/media/``.
``make_anki_deck.py`` then bundles those clips into ``Numbers.apkg`` and auto-plays them
Duolingo-style. Only the Numbers deck gets audio.

The spelled-out ``spanish`` field is fed to TTS verbatim — digits are already words
(``"cuarenta y siete"``, ``"tres con cincuenta"``), so there is no numeral-reading ambiguity.

**Idempotent:** clips already present are skipped, so re-running does nothing (no re-synthesis,
no wasted GPU time). Backend/voice are chosen via env vars (see ``tts.py``); default is Piper on
CPU. For the H100 route use ``slurm/generate_number_audio.slurm`` (``TTS_BACKEND=xtts``).

Run from the repo root (top-level import style):  ``python src/generate_number_audio.py``
"""

from __future__ import annotations  # Python 3.9 (some HPC clusters) — keep annotations lazy

import argparse
import json
import os

from tts import find_audio, synthesize

NUMBERS_FOLDER = os.path.join(os.path.dirname(__file__), "../anki/numbers")
MEDIA_FOLDER = os.path.join(NUMBERS_FOLDER, "media")

# When ``--limit`` is used (smoke test), front-load a varied spread so a handful of clips still
# exercises the tricky cases: an atom, a compound, an irregular hundred, gender/apocope, a price,
# a year, and an ordinal. Any of these not in the card set are simply skipped.
SMOKE_PRIORITY = [
    "cuarenta y siete",
    "quinientos",
    "novecientos noventa y nueve mil novecientos noventa y nueve",
    "veintiún libros",
    "veintiuna casas",
    "cien euros",
    "tres con cincuenta",
    "mil novecientos noventa y cinco",
    "primero",
]


def collect_spanish_strings() -> list[str]:
    """Return the sorted set of unique, non-empty ``spanish`` strings across all number cards."""
    strings: set[str] = set()
    for name in os.listdir(NUMBERS_FOLDER):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(NUMBERS_FOLDER, name), encoding="utf-8") as f:
            for card in json.load(f):
                spanish = (card.get("spanish") or "").strip()
                if spanish:
                    strings.add(spanish)
    ordered = sorted(strings)
    # Priority phrases first (for --limit smoke tests), then the rest.
    head = [p for p in SMOKE_PRIORITY if p in strings]
    return head + [s for s in ordered if s not in head]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only synthesize the first N phrases (a varied spread) — for a quick smoke test. "
        "Idempotent, so a later full run just fills in the rest.",
    )
    args = parser.parse_args()

    os.makedirs(MEDIA_FOLDER, exist_ok=True)
    strings = collect_spanish_strings()
    if args.limit is not None:
        strings = strings[: args.limit]
        print(f"SMOKE TEST: limiting to {len(strings)} phrases")

    generated = skipped = 0
    for text in strings:
        if find_audio(MEDIA_FOLDER, text):
            skipped += 1
            continue
        basename = synthesize(text, MEDIA_FOLDER)
        generated += 1
        print(f"  {text}  ->  {basename}")

    print(
        f"\nDone: {generated} generated, {skipped} already present "
        f"({len(strings)} phrases requested) in {MEDIA_FOLDER}"
    )


if __name__ == "__main__":
    main()
