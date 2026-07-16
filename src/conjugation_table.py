"""Build the full-paradigm conjugation table shown on the back of conjugation cards.

Each generated card is a single verb+tense+person and stores its answer in ``conjugated_form``.
Collecting every person's form for a given (verb, tense) reconstructs the full paradigm — done
deterministically from data we already have, so there are no extra LLM calls and the table is
guaranteed consistent with the card answers themselves.
"""

# Canonical grammatical-person order (matches generate_verb_conjugations.PERSONS) paired with the
# short label shown in the table. The imperative mood only has tú/vosotros forms in our data, so
# its table renders just those two rows — correct for that mood, not a missing gap.
PERSON_ORDER = [
    ("yo", "yo"),
    ("tú", "tú"),
    ("él/ella/usted", "él/ella/Ud."),
    ("nosotros/nosotras", "nosotros"),
    ("vosotros/vosotras", "vosotros"),
    ("ellos/ellas/ustedes", "ellos/Uds."),
]


def paradigm_key(infinitive, tense, tense_key=None):
    """The ``(infinitive, tense)`` key used to group cards into one paradigm.

    Infinitive is lower-cased so "Ser"/"ser" group together. ``tense_key`` (default identity)
    normalizes the tense string — legacy data spells the same tense several ways
    ("condicional" vs "indicativo/condicional", "imperativo afirmativo" vs "imperativo/afirmativo"),
    so callers pass a normalizer (e.g. mapping to the display name) to keep those paradigms whole.
    """
    tense_key = tense_key or (lambda t: t)
    return ((infinitive or "").strip().lower(), tense_key((tense or "").strip()))


def build_conjugation_lookup(cards, tense_key=None):
    """Map ``paradigm_key -> {person: conjugated_form}`` from a flat list of cards.

    Forward and reverse cards share the same ``conjugated_form``; the first one seen for a person
    wins (they agree). Cards missing any required field are skipped.
    """
    lookup = {}
    for card in cards:
        person = (card.get("person") or "").strip()
        conjugated = (card.get("conjugated_form") or "").strip()
        key = paradigm_key(card.get("infinitive"), card.get("tense"), tense_key)
        if not (key[0] and key[1] and person and conjugated):
            continue
        lookup.setdefault(key, {}).setdefault(person, conjugated)
    return lookup


def render_conjugation_table(forms, current_person, tense_name=""):
    """HTML table of every known person's form for one (verb, tense); current row highlighted.

    ``forms`` is the ``{person: conjugated_form}`` dict for a single (infinitive, tense). Rows are
    emitted in canonical person order, skipping persons with no form (e.g. the imperative). Returns
    ``""`` when there is nothing to show.
    """
    if not forms:
        return ""
    rows = []
    for person, label in PERSON_ORDER:
        form = forms.get(person)
        if not form:
            continue
        cls = " class='ct-current'" if person == current_person else ""
        rows.append(
            f"<tr{cls}><td class='ct-person'>{label}</td><td class='ct-form'>{form}</td></tr>"
        )
    if not rows:
        return ""
    heading = f"<div class='ct-title'>{tense_name}</div>" if tense_name else ""
    return f"<div class='conj-table'>{heading}<table>{''.join(rows)}</table></div>"
