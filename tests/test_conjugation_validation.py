"""Validation + repair of malformed conjugation example sentences."""

from conjugation_validation import repair_conjugation_card, validate_conjugation_card


def _fwd(sentence, infinitive="comer", conjugated="como"):
    return {
        "direction": "conjugation_forward",
        "infinitive": infinitive,
        "conjugated_form": conjugated,
        "example_sentence_es": sentence,
    }


def _rev(sentence, infinitive="comer", conjugated="como"):
    return {
        "direction": "conjugation_reverse",
        "infinitive": infinitive,
        "conjugated_form": conjugated,
        "example_sentence_es": sentence,
    }


# ---- validation ----


def test_valid_forward_and_reverse_have_no_issues():
    assert validate_conjugation_card(_fwd("Yo [comer] pan.")) == []
    assert validate_conjugation_card(_rev("Yo como pan.")) == []


def test_forward_missing_blank_is_flagged():
    assert "forward sentence has no [infinitive] blank" in validate_conjugation_card(
        _fwd("Yo como pan.")
    )


def test_forward_bracket_with_extra_text_is_flagged():
    issues = validate_conjugation_card(
        _fwd("Tú [hablar, imperfecto] mucho.", infinitive="hablar", conjugated="hablabas")
    )
    assert "forward blank does not contain the bare infinitive" in issues


def test_reverse_with_unfilled_bracket_is_flagged():
    assert "reverse sentence still has an unfilled [bracket]" in validate_conjugation_card(
        _rev("Yo [comer] pan.")
    )


def test_reverse_missing_verb_is_flagged():
    assert "reverse sentence does not contain the conjugated form" in validate_conjugation_card(
        _rev("Yo bebo agua.")
    )


# ---- repair ----


def test_repair_forward_normalizes_extra_bracket_text():
    card = _fwd("Tú [hablar, imperfecto] mucho.", infinitive="hablar", conjugated="hablabas")
    changed, unrepairable = repair_conjugation_card(card)
    assert changed and not unrepairable
    assert card["example_sentence_es"] == "Tú [hablar] mucho."
    assert validate_conjugation_card(card) == []


def test_repair_forward_brackets_a_leaked_conjugated_form():
    card = _fwd("Yo como pan.")
    changed, unrepairable = repair_conjugation_card(card)
    assert changed and not unrepairable
    assert card["example_sentence_es"] == "Yo [comer] pan."


def test_repair_forward_unrepairable_when_verb_absent():
    card = _fwd("Yo bebo agua.")
    changed, unrepairable = repair_conjugation_card(card)
    assert not changed and unrepairable


def test_repair_reverse_fills_unfilled_bracket():
    card = _rev("Yo [comer] pan.", conjugated="como")
    changed, unrepairable = repair_conjugation_card(card)
    assert changed and not unrepairable
    assert card["example_sentence_es"] == "Yo como pan."
    assert validate_conjugation_card(card) == []


def test_repair_valid_cards_are_left_unchanged():
    for card in (_fwd("Yo [comer] pan."), _rev("Yo como pan.")):
        before = card["example_sentence_es"]
        changed, unrepairable = repair_conjugation_card(card)
        assert not changed and not unrepairable
        assert card["example_sentence_es"] == before


def test_repair_lowercases_blank_and_leaves_correct_lowercase_alone():
    # A legacy card can store the infinitive capitalized ("Dormir"); the blank must stay lowercase
    # and an already-correct "[dormir]" must not be gratuitously re-capitalized.
    card = _fwd("Tú [dormir] bien.", infinitive="Dormir", conjugated="duermes")
    changed, _ = repair_conjugation_card(card)
    assert not changed
    assert card["example_sentence_es"] == "Tú [dormir] bien."
    # A capitalized blank at sentence start is normalized down to lowercase.
    card2 = _fwd("[Dormir] bien.", infinitive="Dormir", conjugated="duerme")
    repair_conjugation_card(card2)
    assert card2["example_sentence_es"] == "[dormir] bien."


def test_repair_is_idempotent():
    card = _fwd("Tú [hablar, imperfecto] mucho.", infinitive="hablar", conjugated="hablabas")
    repair_conjugation_card(card)
    changed_again, _ = repair_conjugation_card(card)
    assert not changed_again
