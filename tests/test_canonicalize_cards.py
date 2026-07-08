"""Generator forces canonical tense/person/infinitive so GUIDs stay stable across runs."""

import os

# generate_verb_conjugations imports llm, which validates the key at import time.
_had_key = "OPENROUTER_API_KEY" in os.environ
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from generate_verb_conjugations import canonicalize_cards  # noqa: E402

if not _had_key:
    os.environ.pop("OPENROUTER_API_KEY", None)


def test_overwrites_inconsistent_tense_and_infinitive():
    # The LLM stripped the category prefix and capitalized the verb — both must be corrected.
    cards = [
        {
            "direction": "conjugation_forward",
            "infinitive": "Ser",
            "tense": "futuro",
            "person": "yo",
        },
        {"direction": "conjugation_reverse", "infinitive": "ser", "tense": "indicativo/futuro"},
    ]
    out = canonicalize_cards(cards, "Ser", "indicativo", "futuro", "él/ella/usted")
    for card in out:
        assert card["infinitive"] == "ser"
        assert card["tense"] == "indicativo/futuro"
        assert card["person"] == "él/ella/usted"


def test_canonical_tense_matches_deck_guid_key_format():
    # Format must match make_anki_deck's key: "category/name" (slash, underscore preserved).
    cards = [{"direction": "conjugation_forward"}]
    out = canonicalize_cards(cards, "comer", "indicativo", "pretérito_indefinido", "yo")
    assert out[0]["tense"] == "indicativo/pretérito_indefinido"


def test_ignores_non_dict_entries():
    cards = ["junk", {"direction": "conjugation_forward"}]
    out = canonicalize_cards(cards, "vivir", "subjuntivo", "presente", "tú")
    assert out[1]["tense"] == "subjuntivo/presente"
    assert out[0] == "junk"
