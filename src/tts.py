"""Local, offline text-to-speech for the Numbers deck audio.

Peninsular-Spanish (es-ES) speech, generated on-device — **no online TTS services**. Model
weights are downloaded once (HuggingFace or a GitHub release), then everything runs offline.
Three backends, selected via the ``TTS_BACKEND`` env var:

- ``piper`` (default) — Piper VITS, native ``es_ES`` voices, runs on CPU. Fast, no GPU needed;
  the safest peninsular *timbre*.
- ``kokoro`` — Kokoro-82M via onnxruntime (no torch), CPU-friendly. More natural than Piper;
  Spanish g2p is espeak-ng Castilian phonology (the /θ/ distinction).
- ``xtts``  — Coqui XTTS-v2, the highest-realism option, meant for the H100 SLURM cluster.
  Clones a Castilian reference voice (``TTS_REF_WAV``) or falls back to built-in speakers.

Voice variety: each backend takes a comma-separated voice list (``TTS_PIPER_VOICES`` /
``TTS_KOKORO_VOICES`` / ``TTS_SPEAKERS``) and picks one *deterministically per phrase*, so the
deck has varied speakers while staying idempotent. Kokoro rotates 3 Spanish voices by default
(all in its one model file); Piper defaults to a single voice (extra voices = extra downloads).

Leading silence: ``TTS_LEAD_SILENCE_MS`` (default 150) pads the start of every clip so audio
doesn't begin abruptly on autoplay.

Switching backends/voices: filenames are content-based (not backend/voice-tagged), so to
re-voice, clear ``anki/numbers/media/`` first (otherwise existing clips are kept).

Filenames are a deterministic content hash of the spoken text (:func:`audio_stem`), so audio
generated on the cluster and the deck built elsewhere agree on names without any manifest.
``make_anki_deck.py`` imports only :func:`find_audio` / :func:`audio_basename` from here — those
are pure-stdlib, so importing this module never pulls in torch/piper.

Heavy backend libraries are imported lazily inside the synth functions and are declared in the
optional ``tts`` dependency group (``uv sync --extra tts``).
"""

# Keep annotations lazy so ``str | None`` etc. don't get evaluated at runtime — some HPC
# clusters still default to Python 3.9, where PEP 604 unions raise TypeError on import.
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile

# Order matters: mp3 preferred (small), wav is the ffmpeg-less fallback. The deck builder
# searches these in order, so a phrase voiced as either extension is picked up transparently.
AUDIO_EXTS = (".mp3", ".wav")


def audio_stem(text: str) -> str:
    """Deterministic, collision-resistant basename stem for a spoken phrase (no extension)."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"es_{digest}"


def audio_basename(text: str, ext: str = ".mp3") -> str:
    """Canonical audio filename for ``text`` (defaults to the preferred mp3 extension)."""
    return audio_stem(text) + ext


def find_audio(media_dir: str, text: str) -> str | None:
    """Return the basename of an already-generated clip for ``text`` in ``media_dir``, or None.

    Matches either extension so the pipeline works whether or not ffmpeg was available at
    generation time. This is what makes the whole thing idempotent from the builder's side.
    """
    stem = audio_stem(text)
    for ext in AUDIO_EXTS:
        if os.path.exists(os.path.join(media_dir, stem + ext)):
            return stem + ext
    return None


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _to_mp3(wav_path: str, mp3_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            wav_path,
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "4",
            mp3_path,
        ],
        check=True,
    )


def _prepend_silence(wav_path: str, ms: int) -> None:
    """Prepend ``ms`` of leading silence to a PCM wav (so clips don't start abruptly).

    Backend-agnostic — operates on the rendered wav before mp3 transcode, so it applies
    regardless of which backend produced the audio or whether ffmpeg is present.
    """
    import wave

    if ms <= 0:
        return
    with wave.open(wav_path, "rb") as r:
        params = r.getparams()
        frames = r.readframes(r.getnframes())
    n = int(params.framerate * ms / 1000)
    silence = b"\x00" * (n * params.sampwidth * params.nchannels)
    with wave.open(wav_path, "wb") as w:
        w.setparams(params)
        w.writeframes(silence + frames)


def _voices(env_var: str, default: str) -> list[str]:
    """Parse a comma-separated voice list from ``env_var`` (falling back to ``default``)."""
    raw = os.getenv(env_var, default)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _pick_voice(text: str, voices: list[str]) -> str:
    """Deterministically choose one voice for ``text`` — stable per phrase, so the deck has
    varied speakers while staying idempotent (same text always maps to the same voice/file)."""
    idx = int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % len(voices)
    return voices[idx]


# --- Piper backend (default, CPU, native es_ES voices) -----------------------------------
# Multiple voices ⇒ multiple model downloads; each is a separate rhasspy/piper-voices model.
# Default is a single voice; set TTS_PIPER_VOICES to a comma list for variety, e.g.
#   TTS_PIPER_VOICES="es_ES-davefx-medium,es_ES-carlfm-x_low,es_ES-sharvard-medium"

_PIPER_VOICES: dict = {}  # voice name -> loaded PiperVoice (lazily imported, so left untyped)


def _piper_voice(voice: str):
    """Load (and cache) a named Piper voice, downloading the onnx model from HuggingFace once."""
    if voice in _PIPER_VOICES:
        return _PIPER_VOICES[voice]

    from huggingface_hub import hf_hub_download
    from piper import PiperVoice

    # rhasspy/piper-voices layout: <lang>/<lang_region>/<speaker>/<quality>/<voice>.onnx[.json]
    lang_region, speaker, quality = voice.split("-")
    lang = lang_region.split("_")[0]
    prefix = f"{lang}/{lang_region}/{speaker}/{quality}/{voice}"
    onnx = hf_hub_download("rhasspy/piper-voices", f"{prefix}.onnx")
    hf_hub_download("rhasspy/piper-voices", f"{prefix}.onnx.json")  # sits beside the onnx
    _PIPER_VOICES[voice] = PiperVoice.load(onnx)
    return _PIPER_VOICES[voice]


def _synth_piper(text: str, wav_path: str) -> None:
    import wave

    voice = _piper_voice(_pick_voice(text, _voices("TTS_PIPER_VOICES", "es_ES-davefx-medium")))
    with wave.open(wav_path, "wb") as wf:
        voice.synthesize_wav(text, wf)


# --- XTTS-v2 backend (cluster/H100, highest realism) -------------------------------------

_XTTS = None


def _xtts_model():
    global _XTTS
    if _XTTS is not None:
        return _XTTS

    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _XTTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _XTTS


def _synth_xtts(text: str, wav_path: str) -> None:
    model = _xtts_model()
    ref_wav = os.getenv("TTS_REF_WAV")  # Castilian reference clip for voice cloning (best accent)
    kwargs = {"text": text, "file_path": wav_path, "language": "es"}
    if ref_wav:
        kwargs["speaker_wav"] = ref_wav  # cloning a single reference: one consistent voice
    else:
        # Rotate built-in speakers for variety (comma list in TTS_SPEAKERS).
        kwargs["speaker"] = _pick_voice(text, _voices("TTS_SPEAKERS", "Ana Florence,Damien Black"))
    model.tts_to_file(**kwargs)


# --- Kokoro-82M backend (local, CPU, onnxruntime — no torch) ------------------------------
# Small StyleTTS2 model; Spanish goes through espeak-ng g2p (Castilian phonology, e.g. the
# /θ/ in "cincuenta"). Model files come from the kokoro-onnx GitHub release (not HuggingFace).

_KOKORO = None
_KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"


def _ensure_kokoro_file(filename: str) -> str:
    import urllib.request

    # Honor XDG_CACHE_HOME so HPC jobs can redirect big model files off the home quota.
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    cache = os.path.join(cache_root, "kokoro-onnx")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, filename)
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(f"{_KOKORO_RELEASE}/{filename}", tmp)
        os.replace(tmp, path)
    return path


def _kokoro_model():
    global _KOKORO
    if _KOKORO is not None:
        return _KOKORO

    from kokoro_onnx import Kokoro

    model = _ensure_kokoro_file("kokoro-v1.0.onnx")
    voices = _ensure_kokoro_file("voices-v1.0.bin")
    _KOKORO = Kokoro(model, voices)
    return _KOKORO


def _synth_kokoro(text: str, wav_path: str) -> None:
    import wave

    import numpy as np

    model = _kokoro_model()
    # Spanish voices ef_dora (F), em_alex (M), em_santa (M) all live in the one model file, so
    # rotating across them gives real speaker variety for free. Override with TTS_KOKORO_VOICES.
    voice = _pick_voice(text, _voices("TTS_KOKORO_VOICES", "ef_dora,em_alex,em_santa"))
    samples, sample_rate = model.create(text, voice=voice, speed=1.0, lang="es")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


_BACKENDS = {"piper": _synth_piper, "xtts": _synth_xtts, "kokoro": _synth_kokoro}


def synthesize(text: str, media_dir: str) -> str:
    """Idempotently produce an audio clip for ``text`` in ``media_dir``; return its basename.

    If a clip already exists (either extension) nothing is regenerated. Otherwise the configured
    backend renders a wav, which is transcoded to mp3 when ffmpeg is present (else kept as wav).
    """
    existing = find_audio(media_dir, text)
    if existing:
        return existing

    os.makedirs(media_dir, exist_ok=True)
    backend = os.getenv("TTS_BACKEND", "piper").lower()
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown TTS_BACKEND {backend!r}; expected one of {sorted(_BACKENDS)}")

    stem = audio_stem(text)
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, stem + ".wav")
        _BACKENDS[backend](text, wav)
        _prepend_silence(wav, int(os.getenv("TTS_LEAD_SILENCE_MS", "150")))
        if _have_ffmpeg():
            out = os.path.join(media_dir, stem + ".mp3")
            _to_mp3(wav, out)
        else:
            out = os.path.join(media_dir, stem + ".wav")
            shutil.copyfile(wav, out)
    return os.path.basename(out)
