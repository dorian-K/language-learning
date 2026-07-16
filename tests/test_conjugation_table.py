"""Full-paradigm conjugation table assembled from the per-person cards."""

from conjugation_table import build_conjugation_lookup, render_conjugation_table


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
