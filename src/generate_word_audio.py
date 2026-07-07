"""Generate peninsular-Spanish audio for the vocab headwords (Refold ES1K deck).

The vocab deck plays audio of the Spanish **word** (e.g. "levantar", "la manzana"), not the
example sentence. This script synthesizes one clip per **unique** headword into ``anki/word_audio/``
using the same offline TTS as the other decks (``tts.py``). ``make_anki_deck.py`` then bundles the
clips its cards reference and auto-plays them Duolingo-style (on the side that shows the Spanish
word: front for spanish→target, back for target→spanish).

The spoken text comes from ``spoken_word(card)`` — the cue for spanish→target cards and the primary
``target_es`` for target→spanish cards — so both directions of the same item share one clip.

Same design as ``generate_sentence_audio.py``:
- **Idempotent** — clips already present are skipped (content-hash filenames).
- **Backend/voice** chosen via env vars (see ``tts.py``); default is Piper on CPU. For the H100
  route use ``slurm/generate_word_audio.slurm`` (``TTS_BACKEND=xtts``).
- **Standalone** — imports only ``tts`` + ``spoken_word`` (no genanki), so it runs on the cluster
  with just the ``tts`` extra installed.

The card JSON is gitignored (it lives on your laptop), so **rsync ``anki/`` up to the cluster
first**, run this there, then rsync ``anki/word_audio/`` back down.

Run from the repo root (top-level import style):  ``python src/generate_word_audio.py``
"""

from __future__ import annotations  # Python 3.9 (some HPC clusters) — keep annotations lazy

import argparse
import json
import os

from spoken_word import spoken_word
from tts import find_audio, synthesize

# Must match make_anki_deck.py: same env var + default so both agree on which vocab folder to use.
VOCAB_SOURCE = os.getenv("VOCAB_SOURCE", "Refold ES1K")
ANKI_DIR = os.path.join(os.path.dirname(__file__), "../anki")

# Only the vocab deck carries spoken headwords (numbers/conjugations have their own audio).
WORD_FOLDER = os.path.join(ANKI_DIR, VOCAB_SOURCE)

# One shared, content-hash-keyed pool; make_anki_deck.py reads the same dir.
MEDIA_FOLDER = os.path.join(ANKI_DIR, "word_audio")


def collect_words() -> list[str]:
    """Return the sorted set of unique, non-empty Spanish headwords across all vocab cards."""
    words: set[str] = set()
    if not os.path.isdir(WORD_FOLDER):
        return []
    for name in os.listdir(WORD_FOLDER):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(WORD_FOLDER, name), encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  skipping {name}: {e}")
                continue
        for card in data:
            word = spoken_word(card)
            if word:
                words.add(word)
    return sorted(words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only synthesize the first N words — for a quick smoke test. Idempotent, so a "
        "later full run just fills in the rest.",
    )
    args = parser.parse_args()

    os.makedirs(MEDIA_FOLDER, exist_ok=True)
    words = collect_words()
    if not words:
        print(
            f"No Spanish headwords found in {WORD_FOLDER}.\n"
            "Did you rsync the anki/ card JSON up to the cluster? (It is gitignored.)"
        )
        return
    if args.limit is not None:
        words = words[: args.limit]
        print(f"SMOKE TEST: limiting to {len(words)} words")

    generated = skipped = 0
    by_voice: dict[str, int] = {}
    for text in words:
        if find_audio(MEDIA_FOLDER, text):
            skipped += 1
            continue
        basename, voice = synthesize(text, MEDIA_FOLDER)
        generated += 1
        # Log the speaker/voice per clip so a degenerate output can be traced back to its voice.
        print(f"  [{voice}]  {text}  ->  {basename}")
        by_voice[voice or "?"] = by_voice.get(voice or "?", 0) + 1

    print(
        f"\nDone: {generated} generated, {skipped} already present "
        f"({len(words)} words requested) in {MEDIA_FOLDER}"
    )
    if by_voice:
        print("Clips per voice:")
        for voice, count in sorted(by_voice.items()):
            print(f"  {count:>4}  {voice}")


if __name__ == "__main__":
    main()
