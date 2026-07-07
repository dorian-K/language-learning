import json

import genanki
import pytest

import make_anki_deck as mad
import sync_anki_edits as sync

FIELDS = ["Front_Word", "Front_Sentence", "Back_Word", "Back_Sentence"]


def _render(card):
    """Render a source card to its Anki field dict via the production make_anki_deck code."""
    deck = genanki.Deck(1, "tmp")
    note = mad.process_vocab_card(card, deck, set())
    # Guard: our GUID formula must stay in lockstep with make_anki_deck's.
    assert sync.source_guid(card) == note.guid
    return dict(zip(FIELDS, note.fields, strict=True))


S2T = {
    "direction": "spanish_to_target",
    "cue_spanish": "levantar",
    "target_en": ["to lift", "to raise"],
    "target_de": ["heben"],
    "example_sentence_es": "¿Puedes levantar esta caja?",
    "example_sentence_en": "Can you lift this box?",
    "example_sentence_de": "Kannst du diese Kiste heben?",
}
T2S = {
    "direction": "target_to_spanish",
    "cue_en": "to lift",
    "cue_de": "heben",
    "target_es": ["levantar", "alzar"],
    "example_sentence_es": "¿Puedes levantar esta caja?",
    "example_sentence_en": "Can you lift this box?",
    "example_sentence_de": "Kannst du diese Kiste heben?",
}


@pytest.mark.parametrize("card", [S2T, T2S])
def test_parse_roundtrips_source(card):
    parsed = sync.parse_vocab_fields(_render(card))
    assert parsed["direction"] == card["direction"]
    for field in sync.SYNC_FIELDS[card["direction"]]:
        assert sync._norm(parsed[field]) == sync._norm(card[field]), field


def test_no_changes_when_identical():
    parsed = sync.parse_vocab_fields(_render(S2T))
    assert sync.card_changes(S2T, parsed) == {}


def test_detects_translation_edit():
    edited = dict(S2T, target_en=["to lift", "to raise", "to pick up"])
    parsed = sync.parse_vocab_fields(_render(edited))
    changes = sync.card_changes(S2T, parsed)
    assert set(changes) == {"target_en"}
    assert changes["target_en"]["new"] == ["to lift", "to raise", "to pick up"]


def test_detects_sentence_edit_t2s():
    edited = dict(T2S, example_sentence_es="¿Puedes alzar esta caja?")
    parsed = sync.parse_vocab_fields(_render(edited))
    changes = sync.card_changes(T2S, parsed)
    assert set(changes) == {"example_sentence_es"}


def test_canon_ignores_quote_and_br_noise():
    a = "<span class='lang-label'>EN:</span> hi<br>there"
    b = '<span class="lang-label">EN:</span> hi\nthere'
    assert sync.canon(a) == sync.canon(b)


def test_canon_strips_sound_tags():
    assert sync.canon('"hola" [sound:es_abc.mp3]') == '"hola"'


def test_direction_alias_is_not_an_edit():
    # spanish_sentence_to_target renders identically to spanish_to_target — no spurious change.
    aliased = dict(S2T, direction="spanish_sentence_to_target")
    parsed = sync.parse_vocab_fields(_render(aliased))
    assert sync.card_changes(aliased, parsed) == {}


def test_apply_updates_source_and_backs_up(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.json").write_text(json.dumps([S2T]), encoding="utf-8")

    patch = [
        {
            "guid": sync.source_guid(S2T),
            "file": "a.json",
            "direction": "spanish_to_target",
            "mod_date": "2026-04-01",
            "changes": {"target_en": {"old": S2T["target_en"], "new": ["to lift", "to hoist"]}},
        }
    ]
    patch_file = tmp_path / "patch.json"
    patch_file.write_text(json.dumps(patch), encoding="utf-8")

    args = type(
        "A", (), {"patch": str(patch_file), "source": str(src), "no_backup": False, "force": False}
    )()
    sync.cmd_apply(args)

    updated = json.loads((src / "a.json").read_text(encoding="utf-8"))
    assert updated[0]["target_en"] == ["to lift", "to hoist"]
    # backup preserves the original
    backup = tmp_path / "src_backup_pre_edit_sync" / "a.json"
    assert json.loads(backup.read_text(encoding="utf-8"))[0]["target_en"] == S2T["target_en"]


def test_apply_skips_conflict_without_force(tmp_path, capsys):
    src = tmp_path / "src"
    src.mkdir()
    drifted = dict(S2T, target_en=["something else entirely"])
    (src / "a.json").write_text(json.dumps([drifted]), encoding="utf-8")

    patch = [
        {
            "guid": sync.source_guid(S2T),
            "file": "a.json",
            "direction": "spanish_to_target",
            "mod_date": "2026-04-01",
            "changes": {"target_en": {"old": S2T["target_en"], "new": ["to hoist"]}},
        }
    ]
    patch_file = tmp_path / "patch.json"
    patch_file.write_text(json.dumps(patch), encoding="utf-8")

    args = type(
        "A", (), {"patch": str(patch_file), "source": str(src), "no_backup": True, "force": False}
    )()
    sync.cmd_apply(args)

    # conflict → left untouched
    updated = json.loads((src / "a.json").read_text(encoding="utf-8"))
    assert updated[0]["target_en"] == ["something else entirely"]
    assert "conflict" in capsys.readouterr().out.lower()
