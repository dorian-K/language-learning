"""_as_card_list normalizes the various JSON shapes GLM/OpenRouter return into a card list."""

import os

# llm.py validates the key at import time; set a dummy so the module imports without network, then
# remove it again so we don't leak it into other tests (e.g. the bbc_noticias no-key test).
_had_key = "OPENROUTER_API_KEY" in os.environ
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from llm import _as_card_list  # noqa: E402

if not _had_key:
    os.environ.pop("OPENROUTER_API_KEY", None)


def test_passthrough_list():
    cards = [{"direction": "conjugation_forward"}, {"direction": "conjugation_reverse"}]
    assert _as_card_list(cards) is cards


def test_unwrap_cards_wrapper():
    fwd = {"direction": "conjugation_forward"}
    rev = {"direction": "conjugation_reverse"}
    assert _as_card_list({"cards": [fwd, rev]}) == [fwd, rev]


def test_unwrap_generic_first_list_of_dicts():
    # Unknown wrapper key still works via the list-of-dicts fallback.
    fwd = {"direction": "conjugation_forward"}
    assert _as_card_list({"result_set": [fwd]}) == [fwd]


def test_wrap_bare_single_object():
    # The exact failure mode: model returned one card object instead of the 2-card array.
    card = {"direction": "conjugation_forward", "conjugated_form": "hablamos"}
    assert _as_card_list(card) == [card]


def test_non_container_returns_empty():
    assert _as_card_list("nonsense") == []
    assert _as_card_list(None) == []
