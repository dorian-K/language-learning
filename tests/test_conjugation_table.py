"""Full-paradigm conjugation table assembled from the per-person cards."""

from conjugation_table import (
    build_conjugation_lookup,
    canonical_person,
    memory_hint,
    render_conjugation_table,
)


def _card(infinitive, tense, person, conjugated, direction="conjugation_forward"):
    return {
        "direction": direction,
        "infinitive": infinitive,
        "tense": tense,
        "person": person,
        "conjugated_form": conjugated,
    }


def test_lookup_groups_by_verb_and_tense():
    cards = [
        _card("hablar", "indicativo/presente", "yo", "hablo"),
        _card("hablar", "indicativo/presente", "tú", "hablas"),
        _card("hablar", "indicativo/futuro", "yo", "hablaré"),
    ]
    lookup = build_conjugation_lookup(cards)
    assert lookup[("hablar", "indicativo/presente")] == {"yo": "hablo", "tú": "hablas"}
    assert lookup[("hablar", "indicativo/futuro")] == {"yo": "hablaré"}


def test_lookup_dedupes_forward_and_reverse():
    # Both directions carry the same conjugated_form; the paradigm keys on person, not direction.
    cards = [
        _card("comer", "indicativo/presente", "yo", "como", "conjugation_forward"),
        _card("comer", "indicativo/presente", "yo", "como", "conjugation_reverse"),
    ]
    lookup = build_conjugation_lookup(cards)
    assert lookup[("comer", "indicativo/presente")] == {"yo": "como"}


def test_lookup_skips_incomplete_cards():
    cards = [_card("ser", "indicativo/presente", "yo", ""), {"infinitive": "ser"}]
    assert build_conjugation_lookup(cards) == {}


def test_canonical_person_maps_abbreviations():
    assert canonical_person("él") == "él/ella/usted"
    assert canonical_person("usted") == "él/ella/usted"
    assert canonical_person("nosotros") == "nosotros/nosotras"
    assert canonical_person("vosotras") == "vosotros/vosotras"
    assert canonical_person("ellos") == "ellos/ellas/ustedes"
    assert canonical_person("yo") == "yo"


def test_lookup_merges_abbreviated_person_into_canonical_row():
    # A legacy card stored person as "él"; it must land in the "él/ella/usted" row, not vanish.
    cards = [
        _card("caer", "indicativo/presente", "yo", "caigo"),
        _card("caer", "indicativo/presente", "él", "cae"),
    ]
    lookup = build_conjugation_lookup(cards)
    forms = lookup[("caer", "indicativo/presente")]
    assert forms["él/ella/usted"] == "cae"
    # Rendering with the abbreviated current person still highlights the right row.
    html = render_conjugation_table(forms, "él", "Presente")
    assert "<tr class='ct-current'><td class='ct-person'>él/ella/Ud.</td>" in html


def test_render_orders_persons_and_highlights_current():
    forms = {
        "ellos/ellas/ustedes": "hablan",
        "yo": "hablo",
        "tú": "hablas",
    }
    html = render_conjugation_table(forms, "tú", "Presente")
    assert "Presente" in html
    # Canonical order: yo before tú before ellos regardless of dict insertion order.
    assert html.index("hablo") < html.index("hablas") < html.index("hablan")
    # The current person's row is the highlighted one.
    assert "<tr class='ct-current'><td class='ct-person'>tú</td>" in html


def test_render_imperative_only_shows_its_two_persons():
    forms = {"tú": "come", "vosotros/vosotras": "comed"}
    html = render_conjugation_table(forms, "tú", "Imperativo afirmativo")
    assert "come" in html and "comed" in html
    # No empty rows for the persons the imperative lacks.
    assert html.count("<tr") == 2


def test_render_empty_forms_returns_empty_string():
    assert render_conjugation_table({}, "yo", "Presente") == ""


def test_render_includes_hint_when_provided():
    html = render_conjugation_table({"yo": "hablo"}, "yo", "Presente", hint="Regular -ar: -o, -as…")
    assert "ct-hint" in html and "Regular -ar" in html


def test_memory_hint_is_class_specific_for_present():
    assert "-ar" in memory_hint("hablar", "Presente")
    assert "-er" in memory_hint("comer", "Presente")
    assert "-ir" in memory_hint("vivir", "Presente")


def test_memory_hint_reflexive_infinitive_resolves_class():
    # dormirse -> -ir class
    assert memory_hint("dormirse", "Presente") == memory_hint("dormir", "Presente")


def test_memory_hint_future_is_class_independent_and_universal():
    assert memory_hint("hablar", "Futuro simple") == memory_hint("comer", "Futuro simple")
    assert "every verb" in memory_hint("ser", "Futuro simple")


def test_memory_hint_unknown_tense_is_empty():
    assert memory_hint("hablar", "Some Unknown Tense") == ""


def test_memory_hint_regular_only_rules_withheld_for_irregular_verbs():
    # A "regular -er endings" note on ser would contradict soy/eres/es — withhold it.
    assert memory_hint("ser", "Presente", is_regular=False) == ""
    assert memory_hint("ir", "Pretérito indefinido", is_regular=False) == ""
    # But the same present rule shows for a regular verb.
    assert memory_hint("comer", "Presente", is_regular=True) != ""
    # Universal rules still show even for irregular verbs.
    assert memory_hint("tener", "Presente de subjuntivo", is_regular=False) != ""
    assert memory_hint("ser", "Futuro simple", is_regular=False) != ""


def test_memory_hint_imperfect_withheld_for_ser_ir_ver():
    for verb in ("ser", "ir", "ver"):
        assert memory_hint(verb, "Pretérito imperfecto", is_regular=False) == ""
    # Regular for everyone else, even other irregular verbs.
    assert memory_hint("tener", "Pretérito imperfecto", is_regular=False) != ""
