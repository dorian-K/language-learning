"""Shared: the Spanish headword a vocab card should SPEAK (for word-level audio).

The Refold ES1K vocab deck plays audio of the Spanish **word**, not the example sentence. Both
directions of the same item should resolve to the same spoken text so they share one content-hash
clip:
- ``spanish_to_target``  → the cue (``cue_spanish``), e.g. "levantar" / "la manzana".
- ``target_to_spanish``  → the primary answer (``target_es[0]``), which is the same headword.

Pure stdlib so ``generate_word_audio.py`` stays standalone (imports only ``tts`` + this) and runs
on the cluster with just the ``tts`` extra.
"""

from __future__ import annotations


def spoken_word(card: dict) -> str:
    """Return the Spanish headword to synthesize/speak for this vocab card ("" if none)."""
    direction = card.get("direction", "")
    if direction in ("spanish_to_target", "spanish_sentence_to_target"):
        return (card.get("cue_spanish", "") or "").strip()
    if direction in ("target_to_spanish", "target_sentence_to_spanish"):
        target_es = card.get("target_es") or []
        if isinstance(target_es, list):
            return target_es[0].strip() if target_es else ""
        return str(target_es).strip()
    return ""
