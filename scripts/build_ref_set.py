"""Build the XTTS reference-clip set: normalize curated candidates into ``ref/``.

One-shot helper (not part of the deck pipeline). Takes the clips you auditioned in
``ref_candidates/`` (see ``fetch_ref_candidates.py``), loudness-normalizes each one, and
writes a uniform ``.wav`` into ``ref/`` — the folder ``TTS_REF_DIR`` points at, whose clips
pin the peninsular accent for XTTS voice cloning.

Why normalize: the sources sit at very different levels (VoxPopuli parliament broadcast vs
MLS audiobook), and XTTS conditions on the reference's loudness. We use **two-pass EBU R128**
``loudnorm`` in *linear* mode (measure, then apply a single linear gain + true-peak limiting),
which equalizes perceived level across clips without the dynamic pumping single-pass loudnorm
can introduce. Output is 22050 Hz mono 16-bit PCM — the convention the rest of the ref set uses.

Edit ``KEEP`` to change which candidates go in, then:  uv run python scripts/build_ref_set.py
"""

from __future__ import annotations

import json
import os
import subprocess

IN_DIR = "ref_candidates"
OUT_DIR = "ref"

# Loudness target (broadcast-ish; good, consistent level for voice-clone references).
TARGET_I = -16.0  # integrated loudness, LUFS
TARGET_TP = -1.5  # max true peak, dBTP
TARGET_LRA = 11.0  # loudness range
SAMPLE_RATE = 22050

# Curated keepers from the audition. Basenames in IN_DIR; output is always <stem>.wav in OUT_DIR.
KEEP = [
    "voxpopuli_es_female_spk125045.wav",
    "voxpopuli_es_female_spk125047.wav",
    "voxpopuli_es_female_spk24942.wav",
    "voxpopuli_es_female_spk28298.wav",
    "voxpopuli_es_female_spk4334.wav",
    "voxpopuli_es_female_spk96811.wav",
    "voxpopuli_es_male_spk125046.wav",
    "voxpopuli_es_male_spk4337.wav",
    "voxpopuli_es_male_spk96812.wav",
    "mls_es_spk11797.opus",
]


def _measure(src: str) -> dict[str, str]:
    """Pass 1: measure the clip's loudness, return loudnorm's JSON stats."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", src,
            "-af", f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, check=True,
    )
    # loudnorm prints the JSON block last on stderr; grab from the final '{' to its '}'.
    err = proc.stderr
    start = err.rindex("{")
    end = err.index("}", start) + 1
    return json.loads(err[start:end])


def _apply(src: str, dst: str, m: dict[str, str]) -> None:
    """Pass 2: apply linear normalization using the measured values → 22050 Hz mono 16-bit wav."""
    af = (
        f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}"
        f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
        f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
        f":offset={m['target_offset']}:linear=true:print_format=summary"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
            "-af", af, "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", dst,
        ],
        check=True,
    )


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    ok = 0
    for name in KEEP:
        src = os.path.join(IN_DIR, name)
        if not os.path.exists(src):
            print(f"  !! missing: {src}")
            continue
        stem = os.path.splitext(name)[0]
        dst = os.path.join(OUT_DIR, stem + ".wav")
        m = _measure(src)
        _apply(src, dst, m)
        kb = os.path.getsize(dst) >> 10
        print(f"  [{ok + 1:>2}] {name:<40} -> {dst}  ({kb} KB, in {float(m['input_i']):.1f} LUFS)")
        ok += 1
    print(f"\nNormalized {ok}/{len(KEEP)} clips into {OUT_DIR}/ at {TARGET_I} LUFS, {SAMPLE_RATE} Hz mono.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
