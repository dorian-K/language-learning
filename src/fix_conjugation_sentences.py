"""Repair malformed conjugation example sentences in the generated JSON, in place.

Scans anki/irregular_verbs and anki/regular_verbs, normalizes forward blanks to a bare
``[infinitive]``, fills unfilled reverse brackets, and reports any card whose verb is missing
from the sentence entirely (unrepairable — regenerate it). Idempotent: re-running changes nothing
once clean.

Usage:
    python src/fix_conjugation_sentences.py            # repair in place, report
    DRY_RUN=true python src/fix_conjugation_sentences.py   # report only, write nothing
    DELETE_UNREPAIRABLE=true python src/fix_conjugation_sentences.py  # rm files needing regen
"""

import glob
import json
import os

from conjugation_validation import repair_conjugation_card, validate_conjugation_card

FOLDERS = [
    os.path.join(os.path.dirname(__file__), "../anki/irregular_verbs"),
    os.path.join(os.path.dirname(__file__), "../anki/regular_verbs"),
]


def main():
    dry_run = os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes")
    delete_unrepairable = os.getenv("DELETE_UNREPAIRABLE", "false").lower() in ("true", "1", "yes")

    repaired = 0
    files_changed = 0
    unrepairable = []  # (filepath, card summary)

    for folder in FOLDERS:
        for filepath in sorted(glob.glob(os.path.join(folder, "*.json"))):
            with open(filepath, encoding="utf-8") as f:
                cards = json.load(f)

            file_changed = False
            file_unrepairable = False
            for card in cards:
                changed, cannot = repair_conjugation_card(card)
                if changed:
                    repaired += 1
                    file_changed = True
                if cannot or validate_conjugation_card(card):
                    file_unrepairable = True
                    unrepairable.append(
                        (
                            filepath,
                            f"{card.get('direction')} | {card.get('infinitive')} | "
                            f"{card.get('conjugated_form')} | {card.get('example_sentence_es')!r}",
                        )
                    )

            if file_changed and not dry_run:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(cards, f, indent=4, ensure_ascii=False)
                files_changed += 1
            elif file_changed:
                files_changed += 1

            if file_unrepairable and delete_unrepairable and not dry_run:
                os.remove(filepath)
                print(f"Deleted (needs regen): {os.path.basename(filepath)}")

    print(
        f"\nRepaired {repaired} card(s) across {files_changed} file(s)"
        f"{' (dry run — nothing written)' if dry_run else ''}."
    )
    if unrepairable:
        print(f"\n{len(unrepairable)} card(s) still invalid (verb missing — regenerate):")
        for fp, summary in unrepairable:
            print(f"  {os.path.basename(fp)}: {summary}")
        if not delete_unrepairable:
            print(
                "\nRe-run with DELETE_UNREPAIRABLE=true to remove those files, then "
                "`python src/generate_verb_conjugations.py` to regenerate them."
            )
    else:
        print("No unrepairable cards remain.")


if __name__ == "__main__":
    main()
