"""Validate and repair the example sentences on conjugation cards.

The LLM occasionally returns malformed conjugation sentences:
  * a FORWARD card whose ``example_sentence_es`` has no ``[infinitive]`` blank, or whose bracket
    holds extra text (``[hablar, imperfecto]``) instead of just the bare infinitive;
  * a REVERSE card that still carries an unfilled ``[bracket]`` instead of the conjugated form;
  * either card where the verb is missing from the sentence entirely (unrepairable — needs regen).

These helpers are pure/stdlib-only so both the generator (validate + retry) and a one-shot repair
script can share them. A forward card is *valid* when its sentence has exactly one blank containing
the bare infinitive; a reverse card is valid when it has no blank and contains the conjugated form.
"""

import re

_BRACKET = re.compile(r"\[[^\]]*\]")


def _norm(text):
    return (text or "").strip().lower()


def _contains_word(haystack, needle):
    """True if ``needle`` appears as a whole word in ``haystack`` (case-insensitive, accent-aware)."""
    needle = _norm(needle)
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", _norm(haystack)) is not None


def validate_conjugation_card(card):
    """Return a list of issue strings for one card; empty means valid.

    Only conjugation cards are checked; anything else returns ``[]``.
    """
    direction = card.get("direction")
    sentence = (card.get("example_sentence_es") or "").strip()
    infinitive = _norm(card.get("infinitive"))
    conjugated = _norm(card.get("conjugated_form"))
    brackets = _BRACKET.findall(sentence)
    issues = []

    if direction == "conjugation_forward":
        if not conjugated:
            issues.append("missing conjugated_form")
        if len(brackets) == 0:
            issues.append("forward sentence has no [infinitive] blank")
        elif len(brackets) > 1:
            issues.append("forward sentence has multiple brackets")
        elif _norm(brackets[0].strip("[]")) != infinitive:
            issues.append("forward blank does not contain the bare infinitive")
    elif direction == "conjugation_reverse":
        if not conjugated:
            issues.append("missing conjugated_form")
        if brackets:
            issues.append("reverse sentence still has an unfilled [bracket]")
        elif conjugated and not _contains_word(sentence, conjugated):
            issues.append("reverse sentence does not contain the conjugated form")
    return issues


def repair_conjugation_card(card):
    """Repair a card's ``example_sentence_es`` in place where possible.

    Returns ``(changed, unrepairable)``. ``unrepairable`` means the verb could not be located in
    the sentence at all, so the card should be regenerated rather than patched.
    """
    direction = card.get("direction")
    sentence = (card.get("example_sentence_es") or "").strip()
    infinitive = (card.get("infinitive") or "").strip()
    conjugated = (card.get("conjugated_form") or "").strip()
    if not sentence or not infinitive:
        return False, True

    if direction == "conjugation_forward":
        # Blank is always the lowercase infinitive (matches the prompt's "[ser]" convention and
        # avoids capitalizing an already-correct "[dormir]" when a legacy card's infinitive field
        # is stored as "Dormir").
        blank = f"[{infinitive.lower()}]"
        if _BRACKET.search(sentence):
            # Normalize every bracket to the bare infinitive (handles "[hablar, imperfecto]" and
            # the rare multi-bracket case). Idempotent when already correct.
            new = _BRACKET.sub(blank, sentence)
        elif conjugated and _contains_word(sentence, conjugated):
            # No blank, but the conjugated form is present as plain text — bracket it.
            new = re.sub(
                rf"(?<!\w){re.escape(conjugated)}(?!\w)",
                blank,
                sentence,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            return False, True
        changed = new != sentence
        if changed:
            card["example_sentence_es"] = new
        return changed, False

    if direction == "conjugation_reverse":
        if _BRACKET.search(sentence):
            if not conjugated:
                return False, True
            # Fill the blank with the conjugated form (same substitution spoken_sentence does).
            new = _BRACKET.sub(conjugated, sentence, count=1)
            # Drop any leftover brackets defensively.
            new = _BRACKET.sub(lambda m: m.group(0).strip("[]"), new)
            card["example_sentence_es"] = new
            return True, False
        # No bracket: valid as long as the conjugated form is present; otherwise unrepairable.
        if conjugated and not _contains_word(sentence, conjugated):
            return False, True
        return False, False

    return False, False
