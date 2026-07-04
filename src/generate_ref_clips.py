"""Generate Castilian (peninsular) reference clips for XTTS voice-cloning — tracked in git.

XTTS clones the *Spain accent* from these clips (see ``tts._xtts_refs`` / ``TTS_REF_DIR``). We
render a few neutral Spanish sentences with several **native es_ES Piper voices** — clean,
single-speaker, guaranteed peninsular — so the cluster's XTTS run gets a Spain accent WITH voice
variety, without needing externally-sourced audio. The sentences are deliberately rich in the
Castilian /θ/ (c before e/i, z: "cielo", "Zaragoza", "cinco", "gracias") to anchor the accent.

The clips are ``.wav`` (``*.mp3`` is gitignored, ``*.wav`` is not) so they are committed as the
canonical references. Swap in human recordings later if you want — just drop more ``.wav`` files
in ``ref/`` (or replace these) and regenerate the deck audio.

Run from the repo root (top-level import style):  ``python src/generate_ref_clips.py``
"""

from __future__ import annotations  # Python 3.9 (some HPC clusters) — keep annotations lazy

import os
import wave

from tts import _piper_voice

REF_DIR = os.path.join(os.path.dirname(__file__), "../ref")

# One native es_ES Piper voice per reference clip → distinct speakers → variety for XTTS to clone.
# Each pairs with a neutral sentence loaded with c/z so the peninsular /θ/ is well represented.
CLIPS = [
    (
        "es_ES-davefx-medium",
        "Buenos días, esta es una grabación de voz en español de España "
        "para practicar los números.",
    ),
    (
        "es_ES-carlfm-x_low",
        "El cielo de Zaragoza estaba despejado cuando cruzamos las cinco "
        "plazas del centro histórico.",
    ),
    (
        "es_ES-sharvard-medium",
        "Gracias por escuchar; hoy hace un día precioso y la ciudad parece "
        "tranquila y silenciosa.",
    ),
    (
        "es_ES-mls_10246-low",
        "Cincuenta y dos personas esperaban cerca de la estación mientras "
        "anunciaban el próximo tren.",
    ),
    (
        "es_ES-mls_9972-low",
        "Vivo en una casa azul junto a la plaza mayor de la vieja ciudad, "
        "cerca del quiosco de la esquina.",
    ),
]


def main() -> None:
    os.makedirs(REF_DIR, exist_ok=True)
    for voice, text in CLIPS:
        out = os.path.join(REF_DIR, f"{voice}.wav")
        pv = _piper_voice(voice)
        with wave.open(out, "wb") as wf:
            pv.synthesize_wav(text, wf)
        print(f"  {voice}  ->  {out}")
    print(f"\nDone: {len(CLIPS)} reference clips in {REF_DIR} (commit them — *.wav is tracked)")


if __name__ == "__main__":
    main()
