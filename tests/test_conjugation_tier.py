from make_anki_deck import HIGH_PRIORITY_IRREGULARS, conjugation_tier


def test_high_priority_verb():
    assert conjugation_tier({"infinitive": "ser"}) == "High Priority"
    assert conjugation_tier({"infinitive": "querer"}) == "High Priority"


def test_high_priority_is_case_insensitive():
    assert conjugation_tier({"infinitive": "Ser"}) == "High Priority"
    assert conjugation_tier({"infinitive": "  HABER "}) == "High Priority"


def test_low_priority_verb():
    assert conjugation_tier({"infinitive": "volver"}) == "Low Priority"
    assert conjugation_tier({"infinitive": "recordar"}) == "Low Priority"


def test_missing_or_empty_infinitive_is_low_priority():
    assert conjugation_tier({}) == "Low Priority"
    assert conjugation_tier({"infinitive": None}) == "Low Priority"
    assert conjugation_tier({"infinitive": ""}) == "Low Priority"


def test_high_priority_set_has_twelve_verbs():
    # Guards against accidental edits to the curated frequency list.
    assert len(HIGH_PRIORITY_IRREGULARS) == 12
