"""Forward conjugation cards show the target tense on the front; reverse cards do not."""

import genanki

import make_anki_deck as mad

FIELDS = ["Front_Word", "Front_Sentence", "Back_Word", "Back_Sentence", "Meta_Tags"]


def _render(card):
    deck = genanki.Deck(1, "tmp")
    note = mad.process_conjugation_card(card, deck, set())
    return dict(zip(FIELDS, note.fields, strict=True))


def test_forward_shows_tense_on_front_word():
    card = {
        "direction": "conjugation_forward",
        "infinitive": "comer",
        "tense": "indicativo/pretérito_imperfecto",
        "person": "yo",
        "conjugated_form": "comía",
        "example_sentence_es": "Mientras yo [comer], hacía viento.",
        "example_sentence_en": "While I was eating, it was windy.",
        "example_sentence_de": "Während ich aß, war es windig.",
    }
    f = _render(card)
    # The learner sees the verb, the person, and the target tense — no guessing.
    assert "comer" in f["Front_Word"]
    assert "yo" in f["Front_Word"]
    assert "Pretérito imperfecto" in f["Front_Word"]
    # The blank stays on the front sentence; the answer is on the back.
    assert "[comer]" in f["Front_Sentence"]
    assert f["Back_Word"].startswith("comía")


def test_reverse_does_not_show_tense_on_front():
    card = {
        "direction": "conjugation_reverse",
        "infinitive": "comer",
        "tense": "indicativo/pretérito_imperfecto",
        "person": "yo",
        "conjugated_form": "comía",
        "example_sentence_es": "Mientras yo comía, hacía viento.",
        "example_sentence_en": "While I was eating, it was windy.",
        "example_sentence_de": "Während ich aß, war es windig.",
    }
    f = _render(card)
    # Reverse is translation practice: the conjugated form is shown, tense is not front-loaded.
    assert "Pretérito imperfecto" not in f["Front_Word"]
    assert "Pretérito imperfecto" not in f["Front_Sentence"]
    # Tense reveal still available on the back.
    assert "Pretérito imperfecto" in f["Meta_Tags"]
