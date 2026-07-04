"""Audition every built-in XTTS-v2 speaker on a peninsular-Spanish diagnostic sentence.

Throwaway helper (NOT the deck pipeline). XTTS-v2 ships ~58 built-in "studio" speakers you
select by name (no reference clip). With ``language="es"`` their accent leans neutral / Latin
American — that's exactly why the deck normally voice-clones Castilian reference clips instead.
But some built-in speakers lean more Castilian than others, so this renders the SAME line for
every built-in speaker, letting you pick the ones that actually sound peninsular.

The line is deliberately loaded with the sounds that separate Castilian from Latin American
Spanish, so the difference is audible per speaker:
  - /θ/ ("th"): the c-before-e/i and z in  ejerCICios, leCCión, Zumo, CINco, cervEZas,
    plaZa, dieZ, haCEmos, graCIas   → Spain = "th", Latin America = "s".
  - vosotros + -áis/-éis endings:  "Vosotros habéis", "queréis"  → used in Spain, not LatAm.
  - the raspy jota /x/:  eJercicios, coJo, Gimnasia  → Spain = harsh guttural, LatAm = soft /h/.

Needs a GPU / large box — run on the cluster:
    sbatch slurm/generate_speaker_audition.slurm
Output: one clip per speaker at  speaker_audition/<speaker>.wav  — rsync back and audition.
"""

from __future__ import annotations

import os

PROMPT = (
    "Vale, chicos. ¿Vosotros habéis hecho ya los ejercicios de la lección? "
    "Si queréis, cojo el coche y en la plaza compramos un zumo, cinco cervezas "
    "y algo de cenar. A las diez hacemos gimnasia. Gracias."
)
OUT_DIR = "speaker_audition"


def main() -> int:
    import torch
    from TTS.api import TTS

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    speakers = sorted(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
    print(f"{len(speakers)} built-in speakers on {device}; rendering diagnostic line each")
    for i, spk in enumerate(speakers, 1):
        safe = spk.replace(" ", "_").replace("/", "_")
        dst = os.path.join(OUT_DIR, f"{safe}.wav")
        if os.path.exists(dst):  # idempotent — re-running skips finished clips
            print(f"  [{i:>2}/{len(speakers)}] skip {spk}")
            continue
        tts.tts_to_file(text=PROMPT, file_path=dst, language="es", speaker=spk)
        print(f"  [{i:>2}/{len(speakers)}] {spk} -> {dst}")

    print(f"\nDone: {len(speakers)} clips in {OUT_DIR}/ — rsync back and audition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
