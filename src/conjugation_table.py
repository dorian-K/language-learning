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


# One-line memory rule per tense, keyed by the display name make_anki_deck derives from
# TENSE_DESCRIPTIONS. Present/preterite/imperfect endings depend on the -ar/-er/-ir class, so those
# are sub-keyed by class; future/conditional/subjunctive/imperative rules are class-independent.
# These are the reliable regular patterns (the table shows the actual forms, so irregular stems are
# still visible); future & conditional endings are identical for EVERY verb.
_HINTS = {
    "Presente": {
        "ar": "Regular -ar: -o, -as, -a, -amos, -áis, -an.",
        "er": "Regular -er: -o, -es, -e, -emos, -éis, -en.",
        "ir": "Regular -ir: -o, -es, -e, -imos, -ís, -en.",
    },
    "Pretérito indefinido": {
        "ar": "Regular -ar: -é, -aste, -ó, -amos, -asteis, -aron.",
        "er": "Regular -er/-ir: -í, -iste, -ió, -imos, -isteis, -ieron.",
        "ir": "Regular -er/-ir: -í, -iste, -ió, -imos, -isteis, -ieron.",
    },
    "Pretérito imperfecto": {
        "ar": "-ar: -aba, -abas, -aba, -ábamos, -abais, -aban (only ser/ir/ver are irregular).",
        "er": "-er/-ir: -ía, -ías, -ía, -íamos, -íais, -ían (only ser/ir/ver are irregular).",
        "ir": "-er/-ir: -ía, -ías, -ía, -íamos, -íais, -ían (only ser/ir/ver are irregular).",
    },
    "Futuro simple": (
        "Infinitive + -é, -ás, -á, -emos, -éis, -án — same endings for every verb; "
        "irregulars change only the stem."
    ),
    "Condicional": ("Infinitive + -ía, -ías, -ía, -íamos, -íais, -ían — same stem as the future."),
    "Presente de subjuntivo": (
        "From the yo-form, drop -o and swap the vowel: -ar→-e, -er/-ir→-a (hablo→hable, tengo→tenga)."
    ),
    "Imperfecto de subjuntivo": (
        "From the ellos-preterite, drop -ron and add -ra, -ras, -ra, -ramos, -rais, -ran "
        "(hablaron→hablara)."
    ),
    "Imperativo afirmativo": (
        "tú = the él/ella present form; vosotros = infinitive with -r → -d (hablad). "
        "Irregular tú: ten, ven, haz, di, sal, pon, sé, ve."
    ),
    "Imperativo negativo": "no + present subjunctive (no hables, no comas).",
}


def _verb_class(infinitive):
    """The -ar/-er/-ir conjugation class of an infinitive ('' if unknown). Handles reflexives."""
    inf = (infinitive or "").strip().lower()
    if inf.endswith("se") and len(inf) > 4:  # reflexive: dormirse -> dormir
        inf = inf[:-2]
    for cls in ("ar", "er", "ir"):
        if inf.endswith(cls):
            return cls
    return ""


# The present- and preterite-indicative ending rules only hold for regular verbs — strong
# irregulars (ser: soy/eres/es; ir: fui/fuiste; tener: tuve) break both the endings and the stem,
# so a "regular -er endings" note on ser would be actively misleading. Those hints are shown only on
# regular verbs. Every other rule (future/conditional endings, the subjunctive and imperative
# derivations) holds for irregulars too — they change only the stem — so they always show. The
# imperfect indicative is regular for all verbs EXCEPT these three, which we suppress by name.
_REGULAR_ONLY_HINTS = {"Presente", "Pretérito indefinido"}
_IMPERFECT_IRREGULARS = {"ser", "ir", "ver"}


def memory_hint(infinitive, tense_display, is_regular=True):
    """A one-line 'how to memorize this tense' rule, or '' if none applies or would mislead.

    ``is_regular`` says whether the verb follows the regular paradigm (the caller knows this from
    which deck the card came from); when False, the class-based ending rules that only hold for
    regular verbs are withheld so an irregular verb never shows a rule its own forms contradict.
    """
    hint = _HINTS.get(tense_display, "")
    if not hint:
        return ""
    if tense_display in _REGULAR_ONLY_HINTS and not is_regular:
        return ""
    base = (infinitive or "").strip().lower()
    if base.endswith("se") and len(base) > 4:
        base = base[:-2]
    if tense_display == "Pretérito imperfecto" and base in _IMPERFECT_IRREGULARS:
        return ""
    if isinstance(hint, dict):
        return hint.get(_verb_class(infinitive), "")
    return hint


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


def render_conjugation_table(forms, current_person, tense_name="", hint=""):
    """HTML table of every known person's form for one (verb, tense); current row highlighted.

    ``forms`` is the ``{person: conjugated_form}`` dict for a single (infinitive, tense). Rows are
    emitted in canonical person order, skipping persons with no form (e.g. the imperative). An
    optional ``hint`` (see memory_hint) is rendered as a one-line memory rule under the table.
    Returns ``""`` when there is nothing to show.
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
    hint_html = f"<div class='ct-hint'>💡 {hint}</div>" if hint else ""
    return f"<div class='conj-table'>{heading}<table>{''.join(rows)}</table>{hint_html}</div>"
