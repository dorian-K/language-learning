"""
Sync manual Anki edits back to source JSON files.

1. Export your edited .apkg from Anki
2. Run: python -m src.sync_anki_changes <deck_name> [--apply]
3. Review the diff, then apply with --apply

Decks: Refold_ES1K (vocab), irregular_verbs (conjugations)
"""

import base64
import hashlib
import json
import os
import re
import sys
import zipfile
import tempfile
import sqlite3
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))


def safe_filename(guid: str) -> str:
    return hashlib.sha256(guid.encode()).hexdigest()[:16]


def parse_lang_spans(text: str) -> dict[str, str]:
    result = {}
    pattern = r"<span class='lang-label'>(EN|DE|ES):</span>(.*?)(?=<br\s*/?>|$)"
    for m in re.finditer(pattern, text, re.DOTALL):
        lang = m.group(1)
        content = m.group(2).strip()
        content = re.sub(r"<br\s*/?>", "\n", content)
        content = re.sub(r"<[^>]+>", "", content).strip()
        result[lang] = content
    return result


def strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip().strip('"').strip("'")


def parse_vocab_card(fields: dict) -> dict | None:
    fw = fields.get("Front_Word", "")
    fs = fields.get("Front_Sentence", "")
    bw = fields.get("Back_Word", "")
    bs = fields.get("Back_Sentence", "")

    spans_fw = parse_lang_spans(fw)
    spans_bs = parse_lang_spans(bs)

    front_plain = strip_html(fw)
    back_plain = strip_html(bw)

    if "ES:" in bw or "ES:" in fw or spans_fw.get("ES") or spans_bs.get("ES"):
        direction = "target_to_spanish"
        cue_en = spans_fw.get("EN") or front_plain
        cue_de = spans_fw.get("DE") or ""
        target_es_raw = spans_fw.get("ES") or spans_bs.get("ES") or back_plain
        target_es = [x.strip() for x in re.split(r",\s*", target_es_raw) if x.strip()]

        example_sentence_en = spans_bs.get("EN") or strip_html(bs)
        example_sentence_de = spans_bs.get("DE") or ""
        example_sentence_es = spans_bs.get("ES") or strip_html(fs)

        return {
            "direction": direction,
            "cue_en": cue_en,
            "cue_de": cue_de,
            "target_es": target_es,
            "example_sentence_en": example_sentence_en,
            "example_sentence_de": example_sentence_de,
            "example_sentence_es": example_sentence_es,
        }
    else:
        direction = "spanish_to_target"
        cue_spanish = front_plain

        target_en_raw = spans_fw.get("EN") or spans_bs.get("EN") or back_plain
        target_de_raw = spans_fw.get("DE") or spans_bs.get("DE") or back_plain
        target_en = [x.strip() for x in re.split(r",\s*", target_en_raw) if x.strip()]
        target_de = [x.strip() for x in re.split(r",\s*", target_de_raw) if x.strip()]

        example_sentence_es = strip_html(fs)
        example_sentence_en = spans_bs.get("EN") or ""
        example_sentence_de = spans_bs.get("DE") or ""

        return {
            "direction": direction,
            "cue_spanish": cue_spanish,
            "target_en": target_en,
            "target_de": target_de,
            "example_sentence_es": example_sentence_es,
            "example_sentence_en": example_sentence_en,
            "example_sentence_de": example_sentence_de,
        }


def parse_conjugation_card(fields: dict) -> dict | None:
    fw = fields.get("Front_Word", "")
    fs = fields.get("Front_Sentence", "")
    bw = fields.get("Back_Word", "")
    bs = fields.get("Back_Sentence", "")
    meta = fields.get("Meta_Tags", "")

    spans_bs = parse_lang_spans(bs)
    spans_bw = parse_lang_spans(bw)

    if "[Conjugation]" in fw:
        direction = "conjugation_forward"
        m = re.match(r"(.+?)\s*\(([^)]+)\)\s*$", fw.replace("[Conjugation]", "").strip())
        if not m:
            return None
        infinitive = m.group(1).strip()
        person = m.group(2).strip()
        conjugated = strip_html(bw)
        example_sentence_es = strip_html(fs)
        example_sentence_en = spans_bw.get("EN") or spans_bs.get("EN") or ""
        example_sentence_de = spans_bw.get("DE") or spans_bs.get("DE") or ""
    elif "[Translation]" in fw:
        direction = "conjugation_reverse"
        m = re.match(r"(.+?)\s*\(([^)]+)\)\s*$", fw.replace("[Translation]", "").strip())
        if not m:
            return None
        conjugated = m.group(1).strip()
        infinitive = m.group(2).strip()
        example_sentence_es = strip_html(fs)
        example_sentence_en = spans_bw.get("EN") or spans_bs.get("EN") or ""
        example_sentence_de = spans_bw.get("DE") or spans_bs.get("DE") or ""
    else:
        return None

    if "|" in meta:
        tense = meta.split("|")[0].strip()
    else:
        tense = meta.strip()

    return {
        "direction": direction,
        "infinitive": infinitive,
        "tense": tense,
        "person": person,
        "conjugated_form": conjugated,
        "example_sentence_es": example_sentence_es,
        "example_sentence_en": example_sentence_en,
        "example_sentence_de": example_sentence_de,
    }


def vocab_key(card: dict) -> str:
    d = card.get("direction", "")
    if d == "spanish_to_target":
        return f"spanish_to_target|{card.get('cue_spanish', '')}"
    elif d == "target_to_spanish":
        return f"target_to_spanish|{card.get('cue_en', '')}|{card.get('cue_de', '')}"
    return f"{d}|{card.get('cue_spanish', '')}"


def conjugation_key(card: dict) -> str:
    return f"{card.get('direction')}|{card.get('infinitive')}|{card.get('tense')}|{card.get('person')}"


def card_text_diff(a: dict, b: dict, model_type: str) -> list[str]:
    diffs = []
    ignore = ["direction", "infinitive", "tense", "person", "earliest_level", "mandatory_level"]
    if model_type == "vocab":
        ignore.append("cue_spanish")
        ignore.append("cue_en")
        ignore.append("cue_de")
        ignore.append("target_es")
        ignore.append("target_en")
        ignore.append("target_de")
    for k in ignore:
        a.pop(k, None)
        b.pop(k, None)
    for k in set(a.keys()) | set(b.keys()):
        av = a.get(k, "")
        bv = b.get(k, "")
        if isinstance(av, list):
            av = ", ".join(av)
        if isinstance(bv, list):
            bv = ", ".join(bv)
        if av != bv:
            diffs.append(f"  {k}: {av!r} → {bv!r}")
    return diffs


def load_apkg(apkg_path: str, output_folder: str):
    extract_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(apkg_path, "r") as z:
        z.extractall(extract_dir)

    db_path = os.path.join(extract_dir, "collection.anki21")
    if not os.path.exists(db_path):
        db_path = os.path.join(extract_dir, "collection.anki2")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT models FROM col LIMIT 1")
    models_json = json.loads(cursor.fetchone()[0])
    model_map = {}
    for mid_str, m_data in models_json.items():
        fields = [f["name"] for f in m_data.get("flds", [])]
        model_map[int(mid_str)] = {"name": m_data["name"], "fields": fields}

    cursor.execute("SELECT n.guid, n.mid, n.flds FROM notes n")
    notes = []
    for row in cursor.fetchall():
        guid, mid, raw_flds = row
        fields_list = raw_flds.split("\x1f")
        model = model_map.get(mid, {"name": "Unknown", "fields": []})
        named = {fname: fval for fname, fval in zip(model["fields"], fields_list)}
        notes.append({"guid": guid, "model_name": model["name"], "fields": named})

    conn.close()
    shutil.rmtree(extract_dir)

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    guid_map = {}
    for note in notes:
        sf = safe_filename(note["guid"])
        guid_map[sf] = note["guid"]
        with open(os.path.join(output_folder, f"{sf}.json"), "w", encoding="utf-8") as f:
            json.dump(note, f, indent=4, ensure_ascii=False)

    with open(os.path.join(output_folder, "_guid_map.json"), "w") as f:
        json.dump(guid_map, f)

    print(f"Exported {len(notes)} cards to {output_folder}/")
    return guid_map


def sync(source_folder: str, snapshot_folder: str, model_type: str, apply: bool):
    source_by_file = {}
    source_by_key = {}

    for fname in os.listdir(source_folder):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(source_folder, fname), encoding="utf-8") as f:
            data = json.load(f)
        for card in data:
            key_func = vocab_key if model_type == "vocab" else conjugation_key
            key = key_func(card)
            source_by_key[key] = card
            source_by_file[fname] = (key, card)

    print(f"Loaded {len(source_by_key)} source cards")

    with open(os.path.join(snapshot_folder, "_guid_map.json")) as f:
        guid_map = json.load(f)

    changed = []
    unmatched = 0

    for sf, guid in guid_map.items():
        with open(os.path.join(snapshot_folder, f"{sf}.json"), encoding="utf-8") as f:
            snapshot = json.load(f)

        fields = snapshot["fields"]
        parse_func = parse_vocab_card if model_type == "vocab" else parse_conjugation_card
        parsed = parse_func(fields)
        if not parsed:
            print(f"  Could not parse {guid}")
            continue

        key_func = vocab_key if model_type == "vocab" else conjugation_key
        key = key_func(parsed)

        if key not in source_by_key:
            unmatched += 1
            continue

        original = source_by_key[key]
        diffs = card_text_diff(dict(original), dict(parsed), model_type)
        if diffs:
            changed.append({"key": key, "original": original, "snapshot": parsed, "diffs": diffs})

    print(f"Changed: {len(changed)}, Unmatched: {unmatched}")

    if not changed:
        print("\nNo changes detected.")
        return

    print(f"\n=== Diff ({len(changed)} cards changed) ===")
    for item in changed[:50]:
        print(f"\n{item['key']}:")
        for d in item["diffs"]:
            print(d)
    if len(changed) > 50:
        print(f"\n... and {len(changed) - 50} more changed cards")

    if apply:
        backup = source_folder + "_backup"
        if os.path.exists(backup):
            print(f"Removing old backup: {backup}/")
            shutil.rmtree(backup)
        print(f"Creating backup: {backup}/")
        shutil.copytree(source_folder, backup)

        changed_keys = {item["key"] for item in changed}
        updated_count = 0

        for fname in os.listdir(source_folder):
            if not fname.endswith(".json") or fname.startswith("_"):
                continue
            fpath = os.path.join(source_folder, fname)
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            made_change = False
            for card in data:
                key_func = vocab_key if model_type == "vocab" else conjugation_key
                if key_func(card) in changed_keys:
                    new_card = next(c for c in changed if c["key"] == key_func(card))["snapshot"]
                    for k, v in new_card.items():
                        if k not in ("earliest_level", "mandatory_level"):
                            card[k] = v
                    made_change = True
            if made_change:
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                updated_count += 1

        print(f"\nUpdated {updated_count} source files.")
        print(f"Backup saved at: {backup}/")
    else:
        print("\nRun with --apply to write changes to source files.")
        print("NOTE: earliest_level and mandatory_level will NOT be overwritten (source-only fields).")


def main():
    apply = "--apply" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if len(args) < 1:
        print(__doc__)
        sys.exit(1)

    deck_arg = args[0].replace("_", " ")

    if "irregular" in deck_arg.lower():
        apkg_path = "anki/irregular_verbs.apkg"
        source_folder = "anki/irregular_verbs"
        snapshot_folder = "anki/snapshot_irregular_verbs"
        model_type = "conjugation"
    elif "Refold" in deck_arg or "ES1K" in deck_arg:
        apkg_path = "anki/Refold ES1K.apkg"
        source_folder = "anki/Refold ES1K"
        snapshot_folder = "anki/snapshot_vocab"
        model_type = "vocab"
    else:
        print(f"Unknown deck: {deck_arg}")
        sys.exit(1)

    if not os.path.exists(apkg_path):
        print(f"APKG not found: {apkg_path}")
        sys.exit(1)

    print(f"Syncing: {deck_arg}")
    print(f"APKG: {apkg_path}")
    print(f"Source: {source_folder}")
    print()

    load_apkg(apkg_path, snapshot_folder)
    sync(source_folder, snapshot_folder, model_type, apply)


if __name__ == "__main__":
    main()
