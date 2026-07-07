"""make_anki_deck attaches vocab audio to the Spanish WORD, not the example sentence."""

import genanki

import make_anki_deck as mad
import tts

FIELDS = ["Front_Word", "Front_Sentence", "Back_Word", "Back_Sentence"]


def _render(card):
    deck = genanki.Deck(1, "tmp")
    note = mad.process_vocab_card(card, deck, set())
    return dict(zip(FIELDS, note.fields, strict=True))


def _stub_word_clip(word, monkeypatch):
    """Pretend a word clip exists so sound_suffix emits a [sound:] tag for `word` only."""
    basename = tts.audio_basename(word)

    def fake_find(media_dir, text):
        return basename if media_dir == mad.WORD_MEDIA_DIR and text.strip() == word else None

    monkeypatch.setattr(mad, "find_audio", fake_find)
    return f"[sound:{basename}]"


def test_spanish_to_target_audio_on_front_word(monkeypatch):
    tag = _stub_word_clip("levantar", monkeypatch)
    card = {
        "direction": "spanish_to_target",
        "cue_spanish": "levantar",
        "target_en": ["to lift"],
        "target_de": ["heben"],
        "example_sentence_es": "¿Puedes levantar esta caja?",
        "example_sentence_en": "Can you lift this box?",
        "example_sentence_de": "Kannst du diese Kiste heben?",
    }
    f = _render(card)
    assert tag in f["Front_Word"]  # word carries the audio
    assert "[sound:" not in f["Front_Sentence"]  # sentence no longer does
    assert "[sound:" not in f["Back_Word"]
    assert "[sound:" not in f["Back_Sentence"]


def test_target_to_spanish_audio_on_back_word(monkeypatch):
    tag = _stub_word_clip("levantar", monkeypatch)
    card = {
        "direction": "target_to_spanish",
        "cue_en": "to lift",
        "cue_de": "heben",
        "target_es": ["levantar", "alzar"],
        "example_sentence_es": "¿Puedes levantar esta caja?",
        "example_sentence_en": "Can you lift this box?",
        "example_sentence_de": "Kannst du diese Kiste heben?",
    }
    f = _render(card)
    assert tag in f["Back_Word"]  # answer word (shown on flip) carries the audio
    assert "[sound:" not in f["Back_Sentence"]
    assert "[sound:" not in f["Front_Word"]
    assert "[sound:" not in f["Front_Sentence"]


def test_no_clip_means_silent(monkeypatch):
    monkeypatch.setattr(mad, "find_audio", lambda media_dir, text: None)
    card = {
        "direction": "spanish_to_target",
        "cue_spanish": "levantar",
        "target_en": ["to lift"],
        "target_de": ["heben"],
        "example_sentence_es": "¿Puedes levantar esta caja?",
    }
    f = _render(card)
    assert all("[sound:" not in v for v in f.values())
