"""Fold manual Anki edits back into the source JSON (Refold ES1K vocab deck).

Problem: you build a deck with ``make_anki_deck.py``, import it into Anki, then hand-edit a
handful of cards there (fix a translation, tweak a sentence). Re-importing a freshly built deck
would clobber those edits. This tool pulls the edits back into the source JSON so the next build
keeps them.

Why a two-step diff/apply with a patch file:
- The source JSON drifts over time (sentences get regenerated, duplicates merged, translations
  changed), so a raw content comparison of *every* card against a fresh export is dominated by
  pipeline noise, not your edits. Comparing content alone can't tell "I edited this" from "the
  generator changed this".
- Anki stamps every note with a modification time (``notes.mod``). Your hand-edits show up as a
  recent cluster distinct from the bulk import, so we use mod-time to isolate the *candidate*
  edited notes, then a content diff (with HTML normalized) to show exactly what changed.
- The patch is a small, reviewable JSON file: rsync it around, prune anything you didn't mean to
  change, commit it, then ``apply`` it to the source JSON (with an automatic backup).

For perfectly clean future syncs you can take a ``snapshot`` right after each build+import; then
``diff --baseline snapshot.json`` compares your export against that exact baseline instead of the
(possibly drifted) current source, isolating edits with no mod-time guessing.

Workflow:
    # optional, for clean future syncs — run right after you build + import a deck:
    python src/sync_anki_edits.py snapshot --out anki/_baseline_refold.json

    # after hand-editing in Anki, export the deck to .apkg, then:
    python src/sync_anki_edits.py diff "anki_import/All__Refold Es1K.apkg" --out patch.json
    #   ... review patch.json, delete any entries you don't want ...
    python src/sync_anki_edits.py apply patch.json

Only the Refold ES1K vocab model (``Symmetrical_ES_EN_DE_Vocab``) is supported; conjugation decks
would need their own parser (tense-name reversal etc.) and are not handled yet.

Run from the repo root (top-level import style): ``python src/sync_anki_edits.py ...``
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from collections import Counter

import genanki

DEFAULT_SOURCE = os.path.join(os.path.dirname(__file__), "../anki/Refold ES1K")
VOCAB_MODEL_FIELDS = ["Front_Word", "Front_Sentence", "Back_Word", "Back_Sentence"]

# Structured card fields synced per direction. `direction` and the derived level fields
# (earliest_level / mandatory_level) are intentionally excluded — they aren't on the Anki card and
# `spanish_sentence_to_target` vs `spanish_to_target` is a harmless alias, not an edit.
SYNC_FIELDS = {
    "spanish_to_target": [
        "cue_spanish",
        "target_en",
        "target_de",
        "example_sentence_es",
        "example_sentence_en",
        "example_sentence_de",
    ],
    "target_to_spanish": [
        "cue_en",
        "cue_de",
        "target_es",
        "example_sentence_es",
        "example_sentence_en",
        "example_sentence_de",
    ],
}

_SPAN = re.compile(
    r"<span class='lang-label'>(EN|DE|ES):</span>\s*(.*?)(?=<span class='lang-label'>|$)",
    re.DOTALL,
)


def canon(text: str) -> str:
    """Normalize a card field's HTML so cosmetic differences don't read as edits.

    Anki rewrites attribute quotes (``class="x"`` ↔ ``class='x'``) and ``<br>`` vs newlines on
    some operations; audio tags may or may not be present depending on when the deck was built.
    Canonicalizing these lets a genuine content edit stand out.
    """
    text = text or ""
    text = re.sub(r"\s*\[sound:[^\]]+\]", "", text)  # drop audio tags
    text = text.replace('class="lang-label"', "class='lang-label'")
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = text.replace("&nbsp;", " ")
    return text.strip()


def _spans(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for lang, val in _SPAN.findall(text):
        out[lang] = val.strip().strip('"').strip()
    return out


def _split_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_vocab_fields(raw_fields: dict[str, str]) -> dict:
    """Parse a note's raw Anki fields back into structured source fields.

    Round-trips cleanly against ``make_anki_deck.process_vocab_card`` (verified in tests), so any
    difference from the source card is a real content change, not a parser artifact.
    """
    fw = canon(raw_fields.get("Front_Word", ""))
    fs = canon(raw_fields.get("Front_Sentence", ""))
    bw = canon(raw_fields.get("Back_Word", ""))
    bs = canon(raw_fields.get("Back_Sentence", ""))

    if bw.startswith("<span class='lang-label'>ES:"):  # target_to_spanish
        sfw, sfs, sbs = _spans(fw), _spans(fs), _spans(bs)
        return {
            "direction": "target_to_spanish",
            "cue_en": sfw.get("EN", ""),
            "cue_de": sfw.get("DE", ""),
            "target_es": _split_list(_spans(bw).get("ES", "")),
            "example_sentence_en": sfs.get("EN", ""),
            "example_sentence_de": sfs.get("DE", ""),
            "example_sentence_es": sbs.get("ES", ""),
        }
    # spanish_to_target
    sbw, sbs = _spans(bw), _spans(bs)
    return {
        "direction": "spanish_to_target",
        "cue_spanish": fw.strip().strip('"').strip(),
        "target_en": _split_list(sbw.get("EN", "")),
        "target_de": _split_list(sbw.get("DE", "")),
        "example_sentence_es": fs.strip().strip('"').strip(),
        "example_sentence_en": sbs.get("EN", ""),
        "example_sentence_de": sbs.get("DE", ""),
    }


def _norm(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value)
    return (value or "").strip() if isinstance(value, str) else str(value or "")


def source_guid(card: dict) -> str:
    """The deterministic genanki GUID make_anki_deck assigns to this vocab card."""
    key = (
        f"{card.get('direction', '')}|{card.get('cue_spanish', '')}"
        f"|{card.get('cue_en', '')}|{card.get('cue_de', '')}"
    )
    return genanki.guid_for(key)


def load_source(source_dir: str) -> dict[str, dict]:
    """Map GUID -> {file, card} for every vocab card in the source folder."""
    index: dict[str, dict] = {}
    for fname in sorted(os.listdir(source_dir)):
        if not fname.endswith(".json") or fname.startswith("_"):
            continue
        with open(os.path.join(source_dir, fname), encoding="utf-8") as f:
            for card in json.load(f):
                if card.get("direction") not in (
                    "spanish_to_target",
                    "spanish_sentence_to_target",
                    "target_to_spanish",
                    "target_sentence_to_spanish",
                ):
                    continue
                index[source_guid(card)] = {"file": fname, "card": card}
    return index


def load_apkg_notes(apkg_path: str) -> list[dict]:
    """Return [{guid, mod, fields:{name:value}}] for the vocab notes in an .apkg."""
    tmp = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(apkg_path) as z:
            z.extractall(tmp)
        db = os.path.join(tmp, "collection.anki21")
        if not os.path.exists(db):
            db = os.path.join(tmp, "collection.anki2")
        if not os.path.exists(db):
            raise SystemExit(
                f"{apkg_path}: no collection.anki21/anki2 found. Newer Anki may use the zstd "
                "'.anki21b' format — re-export with 'Support older Anki versions' enabled."
            )
        conn = sqlite3.connect(db)
        try:
            models = json.loads(conn.execute("SELECT models FROM col LIMIT 1").fetchone()[0])
            model_by_id = {
                int(mid): {"name": m["name"], "fields": [f["name"] for f in m["flds"]]}
                for mid, m in models.items()
            }
            notes = []
            for guid, mid, mod, flds in conn.execute("SELECT guid, mid, mod, flds FROM notes"):
                model = model_by_id.get(mid, {"name": "?", "fields": []})
                if model["fields"] != VOCAB_MODEL_FIELDS:
                    raise SystemExit(
                        f"Unsupported note model {model['name']!r} with fields {model['fields']}. "
                        "Only the Refold ES1K vocab model is supported."
                    )
                values = flds.split("\x1f")
                notes.append(
                    {
                        "guid": guid,
                        "mod": mod,
                        "fields": dict(zip(model["fields"], values, strict=True)),
                    }
                )
            return notes
        finally:
            conn.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _mod_day(mod: int) -> str:
    return datetime.datetime.fromtimestamp(mod, datetime.UTC).strftime("%Y-%m-%d")


def _import_day(notes: list[dict]) -> str:
    """The day the bulk import happened = the day the most notes share a mod timestamp."""
    return Counter(_mod_day(n["mod"]) for n in notes).most_common(1)[0][0]


def card_changes(old: dict, new_parsed: dict) -> dict:
    """Field -> {old, new} for the synced fields that differ. `old` is the source card."""
    fields = SYNC_FIELDS.get(new_parsed["direction"], [])
    changes = {}
    for field in fields:
        ov, nv = old.get(field), new_parsed.get(field)
        if _norm(ov) != _norm(nv):
            changes[field] = {"old": ov, "new": nv}
    return changes


def cmd_diff(args) -> None:
    source = load_source(args.source)
    notes = load_apkg_notes(args.apkg)
    print(f"Source cards: {len(source)}  |  Anki notes: {len(notes)}")

    baseline = None
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            baseline = json.load(f)  # guid -> structured fields
        print(f"Baseline: {len(baseline)} cards ({args.baseline})")

    if args.all or baseline:
        candidates = notes
        cutoff_desc = "all notes"
    else:
        cutoff = args.since or _import_day(notes)
        candidates = [n for n in notes if _mod_day(n["mod"]) > cutoff]
        cutoff_desc = f"notes modified after {cutoff}"
    print(f"Considering {cutoff_desc}: {len(candidates)} candidate notes\n")

    patch, unmatched, skipped_multiline = [], 0, 0
    for note in candidates:
        if note["guid"] not in source:
            unmatched += 1
            continue
        parsed = parse_vocab_fields(note["fields"])
        entry = source[note["guid"]]
        # Compare against the baseline snapshot if given, else the (possibly drifted) source card.
        old = baseline.get(note["guid"], entry["card"]) if baseline else entry["card"]
        changes = card_changes(old, parsed)
        if not changes:
            continue
        # A newline in a "new" value means concatenated example sentences — an artifact of the
        # build-time duplicate merge, not a hand edit. --skip-multiline drops those.
        if args.skip_multiline and any("\n" in str(ch["new"]) for ch in changes.values()):
            skipped_multiline += 1
            continue
        patch.append(
            {
                "guid": note["guid"],
                "file": entry["file"],
                "direction": parsed["direction"],
                "mod_date": _mod_day(note["mod"]),
                "changes": changes,
            }
        )

    patch.sort(key=lambda e: (e["mod_date"], e["file"]))
    extra = f", {skipped_multiline} multi-sentence drift skipped" if args.skip_multiline else ""
    print(
        f"=== {len(patch)} card(s) changed  "
        f"({unmatched} candidate(s) had no source match{extra}) ===\n"
    )
    for e in patch:
        print(f"{e['mod_date']}  {e['file']}  [{e['direction']}]")
        for field, ch in e["changes"].items():
            print(f"    {field}:")
            print(f"        - {_norm(ch['old'])!r}")
            print(f"        + {_norm(ch['new'])!r}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(patch, f, indent=2, ensure_ascii=False)
        print(f"\nWrote patch: {args.out}  ({len(patch)} entries)")
        print("Review/prune it, then:  python src/sync_anki_edits.py apply", args.out)
    else:
        print("\n(no --out given; run again with --out patch.json to save a patch)")


def cmd_apply(args) -> None:
    with open(args.patch, encoding="utf-8") as f:
        patch = json.load(f)
    if not patch:
        print("Patch is empty — nothing to apply.")
        return

    # Group by file so each file is read/written once.
    by_file: dict[str, list[dict]] = {}
    for e in patch:
        by_file.setdefault(e["file"], []).append(e)

    backup = args.source.rstrip("/") + "_backup_pre_edit_sync"
    if not args.no_backup:
        if os.path.exists(backup):
            shutil.rmtree(backup)
        shutil.copytree(args.source, backup)
        print(f"Backup: {backup}/")

    applied = conflicts = missing = 0
    for fname, entries in by_file.items():
        fpath = os.path.join(args.source, fname)
        if not os.path.exists(fpath):
            print(f"  ! file gone, skipping: {fname}")
            missing += len(entries)
            continue
        with open(fpath, encoding="utf-8") as f:
            cards = json.load(f)
        by_guid = {source_guid(c): c for c in cards}
        touched = False
        for e in entries:
            card = by_guid.get(e["guid"])
            if card is None:
                print(f"  ! card gone ({e['guid']}) in {fname}; source may have drifted — skipping")
                missing += 1
                continue
            for field, ch in e["changes"].items():
                if _norm(card.get(field)) != _norm(ch["old"]):
                    print(
                        f"  ~ conflict in {fname} [{field}]: source is now "
                        f"{_norm(card.get(field))!r}, patch expected {_norm(ch['old'])!r}"
                    )
                    conflicts += 1
                    if not args.force:
                        continue
                card[field] = ch["new"]
                touched = True
                applied += 1
        if touched:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(cards, f, indent=4, ensure_ascii=False)

    print(f"\nApplied {applied} field change(s). Conflicts: {conflicts}, missing: {missing}.")
    if conflicts and not args.force:
        print("Conflicting fields were left untouched. Re-run with --force to overwrite them.")


def cmd_snapshot(args) -> None:
    """Dump the current source render (GUID -> structured fields) as a baseline for future diffs."""
    source = load_source(args.source)
    snap = {guid: entry["card"] for guid, entry in source.items()}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False)
    print(f"Wrote baseline snapshot: {args.out}  ({len(snap)} cards)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diff", help="diff an edited .apkg against source, write a patch")
    d.add_argument("apkg", help="path to the edited .apkg exported from Anki")
    d.add_argument("--source", default=DEFAULT_SOURCE, help="source JSON folder")
    d.add_argument("--out", help="write the patch to this JSON file")
    d.add_argument("--since", metavar="YYYY-MM-DD", help="only notes modified after this day")
    d.add_argument("--all", action="store_true", help="ignore mod-time; consider every note")
    d.add_argument("--baseline", help="baseline snapshot JSON to diff against (see 'snapshot')")
    d.add_argument(
        "--skip-multiline",
        action="store_true",
        help="drop entries whose new value has newline-joined sentences (build-merge drift, "
        "not hand edits)",
    )
    d.set_defaults(func=cmd_diff)

    a = sub.add_parser("apply", help="apply a patch to the source JSON")
    a.add_argument("patch", help="patch JSON produced by 'diff'")
    a.add_argument("--source", default=DEFAULT_SOURCE, help="source JSON folder")
    a.add_argument("--no-backup", action="store_true", help="skip the source backup")
    a.add_argument("--force", action="store_true", help="overwrite fields even on conflict")
    a.set_defaults(func=cmd_apply)

    s = sub.add_parser("snapshot", help="save a baseline snapshot of the current source")
    s.add_argument("--source", default=DEFAULT_SOURCE, help="source JSON folder")
    s.add_argument("--out", required=True, help="output baseline JSON path")
    s.set_defaults(func=cmd_snapshot)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
