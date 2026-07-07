from spoken_sentence import spoken_sentence


def test_forward_substitutes_conjugated_form():
    card = {
        "direction": "conjugation_forward",
        "conjugated_form": "volváis",
        "example_sentence_es": "No [volver] al poo, el calor es peligroso.",
    }
    assert spoken_sentence(card) == "No volváis al poo, el calor es peligroso."


def test_forward_bracket_with_hint_form():
    card = {
        "direction": "conjugation_forward",
        "conjugated_form": "soy",
        "example_sentence_es": "Yo [ser, infinitive] muy cansado.",
    }
    assert spoken_sentence(card) == "Yo soy muy cansado."


def test_forward_substitutes_only_first_bracket():
    card = {
        "direction": "conjugation_forward",
        "conjugated_form": "soy",
        "example_sentence_es": "[ser] o no [ser].",
    }
    # Only the first blank is the target verb; a second bracket (rare) is left as-is.
    assert spoken_sentence(card) == "soy o no [ser]."


def test_reverse_sentence_unchanged():
    card = {
        "direction": "conjugation_reverse",
        "conjugated_form": "volváis",
        "example_sentence_es": "No volváis al poo, el calor es peligroso.",
    }
    assert spoken_sentence(card) == "No volváis al poo, el calor es peligroso."


def test_vocab_card_without_conjugated_form_unchanged():
    card = {"example_sentence_es": "Me gusta comer manzanas."}
    assert spoken_sentence(card) == "Me gusta comer manzanas."


def test_conjugated_form_present_but_no_bracket_unchanged():
    card = {"conjugated_form": "soy", "example_sentence_es": "Ya está conjugado aquí."}
    assert spoken_sentence(card) == "Ya está conjugado aquí."


def test_missing_field_returns_empty():
    assert spoken_sentence({}) == ""
    assert spoken_sentence({"example_sentence_es": None}) == ""
    assert spoken_sentence({"example_sentence_es": "  "}) == ""
