"""Shared: reconstruct the Spanish sentence as it should be SPOKEN/synthesized.

Forward conjugation cards blank the verb as ``[infinitive]`` in ``example_sentence_es``
(e.g. ``"No [volver] al poo."``); the answer lives separately in ``conjugated_form``. Feeding
the raw field to TTS speaks the infinitive ("volver") instead of the conjugated form ("volváis").
This helper substitutes the conjugated form back into the ``[…]`` slot so the audio is correct.

Vocab and reverse conjugation cards have no bracketed blank, so this is a no-op for them
(returns ``example_sentence_es`` unchanged — their existing content-hash clips stay valid).

Pure stdlib (only ``re``) so ``generate_sentence_audio.py`` keeps its "imports only ``tts``,
no genanki" property and runs on the cluster with just the ``tts`` extra.
"""

from __future__ import annotations

import re

# Matches the first [ ... ] blank, e.g. "[volver]" or "[ser, infinitive]".
_BRACKET = re.compile(r"\[[^\]]*\]")


def spoken_sentence(card: dict) -> str:
    """Return the Spanish example sentence as it should be spoken (verb filled in)."""
    text = (card.get("example_sentence_es") or "").strip()
    conjugated = (card.get("conjugated_form") or "").strip()
    if conjugated and "[" in text:
        text = _BRACKET.sub(conjugated, text, count=1)
    return text
