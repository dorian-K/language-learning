"""Fetch Castilian (peninsular) reference-clip *candidates* for XTTS voice cloning.

One-shot, throwaway helper — NOT part of the deck pipeline. Drops clips in a staging
dir (``ref_candidates/``) for the user to audition; keep the best, copy those into
``ref/`` (which ``TTS_REF_DIR`` points at), delete the rest.

Two sources, two extraction paths (each dodges a different failure mode of this small box):

  - **VoxPopuli es** (Spanish MEP speeches — peninsular, clean broadcast). Read from a
    LOCAL parquet shard at ``VOX_PARQUET``. The shard's normal row groups are ~670MB each
    and OOM a 2.7GB box when decompressed; the datasets-server rows API also refuses them
    (no page index). But the shard's *last* row group is tiny (~67MB, ~90 rows, dozens of
    distinct speakers), so we read only that one — cheap and varied.
  - **MLS spanish** (audiobook — cleaner, mixed accent). Fetched via the datasets-server
    ``/rows`` API, one small clip at a time. Consecutive rows are the same narrator, so we
    sample at spread offsets to get distinct speakers.

Usage:
    HF_TOKEN=hf_... VOX_PARQUET=/tmp/voxpopuli_es.parquet \\
        uv run --with requests --with pyarrow python scripts/fetch_ref_candidates.py

Then audition ref_candidates/*, and:
    mkdir -p ref && cp ref_candidates/<good>.wav ref/
    # convert non-wav:  ffmpeg -i in.opus -ar 22050 -ac 1 out.wav
"""

from __future__ import annotations

import os
import sys

import requests

OUT_DIR = "ref_candidates"
PER_SOURCE = 12
ROWS_URL = "https://datasets-server.huggingface.co/rows"

HDR = {"User-Agent": "ref-candidate-fetcher/3.0"}
_HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if _HF_TOKEN:
    HDR["Authorization"] = f"Bearer {_HF_TOKEN}"


def _ext_from_bytes(raw: bytes, path: str | None) -> str:
    if path:
        _, e = os.path.splitext(path)
        if e:
            return e.lower()
    if raw[:4] == b"RIFF":
        return ".wav"
    if raw[:4] == b"OggS":
        return ".ogg"
    if raw[:4] == b"fLaC":
        return ".flac"
    return ".bin"


# --- VoxPopuli: local parquet, small last row group ---------------------------------------


def fetch_voxpopuli() -> int:
    import pyarrow.parquet as pq

    path = os.getenv("VOX_PARQUET", "/tmp/voxpopuli_es.parquet")
    if not os.path.exists(path):
        print(f"  VOX_PARQUET not found at {path}; skipping voxpopuli")
        return 0
    pf = pq.ParquetFile(path)
    rg = pf.metadata.num_row_groups - 1  # smallest tail group — safe to decompress
    print(f"  reading local parquet row group {rg} ({path})")
    tbl = pf.read_row_group(rg, columns=["speaker_id", "gender", "audio"])
    d = tbl.to_pydict()
    seen: set[str] = set()
    n = 0
    for spk, gender, audio in zip(d["speaker_id"], d["gender"], d["audio"]):
        if n >= PER_SOURCE:
            break
        key = str(spk)
        if key in seen:
            continue
        raw = audio.get("bytes") if isinstance(audio, dict) else None
        if not raw:
            continue
        seen.add(key)
        ext = _ext_from_bytes(raw, audio.get("path") if isinstance(audio, dict) else None)
        dest = os.path.join(OUT_DIR, f"voxpopuli_es_{gender or 'x'}_spk{key}{ext}")
        with open(dest, "wb") as f:
            f.write(raw)
        n += 1
        print(f"    [{n}] {dest} ({len(raw) >> 10} KB)")
    return n


# --- MLS: datasets-server rows API, sampled across offsets --------------------------------

MLS = dict(dataset="facebook/multilingual_librispeech", config="spanish", split="train")
# audiobook rows are grouped by narrator → sample widely to hit distinct speakers
MLS_OFFSETS = [
    0, 2000, 4000, 7000, 10000, 14000, 19000, 25000, 32000, 40000,
    50000, 62000, 75000, 90000, 105000, 122000, 140000, 160000, 180000, 205000,
]


def _rows(offset: int, length: int) -> list[dict]:
    r = requests.get(
        ROWS_URL, params={**MLS, "offset": offset, "length": length}, headers=HDR, timeout=60
    )
    if r.status_code != 200:
        print(f"    !! rows HTTP {r.status_code} @off={offset}: {r.text[:150]}")
        return []
    return r.json().get("rows", [])


def _audio_src(cell: object) -> str | None:
    if isinstance(cell, list) and cell and isinstance(cell[0], dict):
        return cell[0].get("src")
    if isinstance(cell, dict):
        return cell.get("src")
    return None


def _download(url: str, dest: str) -> bool:
    if url.startswith("/"):
        url = "https://datasets-server.huggingface.co" + url
    try:
        with requests.get(url, headers=HDR, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                print(f"    !! clip HTTP {resp.status_code}")
                return False
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    !! clip failed: {e}")
        return False


def fetch_mls() -> int:
    seen: set[str] = set()
    n = 0
    for off in MLS_OFFSETS:
        if n >= PER_SOURCE:
            break
        for item in _rows(off, 3):
            row = item.get("row", {})
            key = str(row.get("speaker_id"))
            if key in seen:
                continue
            src = _audio_src(row.get("audio"))
            if not src:
                continue
            ext = os.path.splitext(src.split("?")[0])[1].lower() or ".wav"
            dest = os.path.join(OUT_DIR, f"mls_es_spk{key}{ext}")
            if _download(src, dest):
                seen.add(key)
                n += 1
                print(f"    [{n}] {dest} ({os.path.getsize(dest) >> 10} KB)")
                break  # one clip per offset → maximise speaker spread
    return n


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("== voxpopuli_es (local parquet) ==")
    v = fetch_voxpopuli()
    print("\n== mls_es (datasets-server) ==")
    m = fetch_mls()
    print(f"\nDone: {v} voxpopuli + {m} mls = {v + m} clips in {OUT_DIR}/")
    print("Audition, then: mkdir -p ref && cp ref_candidates/<good>.wav ref/")
    print("(convert non-wav:  ffmpeg -i in.opus -ar 22050 -ac 1 out.wav)")
    return 0 if (v + m) else 1


if __name__ == "__main__":
    sys.exit(main())
