"""Local, offline text-to-speech for the Numbers deck audio.

Peninsular-Spanish (es-ES) speech, generated on-device — **no online TTS services**. Model
weights are downloaded once (HuggingFace or a GitHub release), then everything runs offline.
Three backends, selected via the ``TTS_BACKEND`` env var:

- ``piper`` (default) — Piper VITS, native ``es_ES`` voices (``TTS_PIPER_VOICE``), runs on CPU.
  Fast, no GPU needed; the safest peninsular *timbre*.
- ``kokoro`` — Kokoro-82M via onnxruntime (no torch), CPU-friendly. More natural than Piper;
  Spanish g2p is espeak-ng Castilian phonology (the /θ/ distinction). Voice: ``TTS_KOKORO_VOICE``.
- ``xtts``  — Coqui XTTS-v2, the highest-realism option, meant for the H100 SLURM cluster.
  Clones a Castilian reference voice (``TTS_REF_WAV``) or falls back to a built-in speaker.

Switching backends: filenames are content-based (not backend-tagged), so to re-voice with a
different backend, clear ``anki/numbers/media/`` first (otherwise existing clips are kept).

Filenames are a deterministic content hash of the spoken text (:func:`audio_stem`), so audio
generated on the cluster and the deck built elsewhere agree on names without any manifest.
``make_anki_deck.py`` imports only :func:`find_audio` / :func:`audio_basename` from here — those
are pure-stdlib, so importing this module never pulls in torch/piper.

Heavy backend libraries are imported lazily inside the synth functions and are declared in the
optional ``tts`` dependency group (``uv sync --extra tts``).
"""

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


# --- Piper backend (default, CPU, native es_ES voices) -----------------------------------

_PIPER_VOICE = None


def _piper_voice():
    """Load (and cache) the Piper voice, downloading the onnx model from HuggingFace once."""
    global _PIPER_VOICE
    if _PIPER_VOICE is not None:
        return _PIPER_VOICE

    from huggingface_hub import hf_hub_download
    from piper import PiperVoice

    voice = os.getenv("TTS_PIPER_VOICE", "es_ES-davefx-medium")
    # rhasspy/piper-voices layout: <lang>/<lang_region>/<speaker>/<quality>/<voice>.onnx[.json]
    lang_region, speaker, quality = voice.split("-")
    lang = lang_region.split("_")[0]
    prefix = f"{lang}/{lang_region}/{speaker}/{quality}/{voice}"
    onnx = hf_hub_download("rhasspy/piper-voices", f"{prefix}.onnx")
    hf_hub_download("rhasspy/piper-voices", f"{prefix}.onnx.json")  # sits beside the onnx
    _PIPER_VOICE = PiperVoice.load(onnx)
    return _PIPER_VOICE


def _synth_piper(text: str, wav_path: str) -> None:
    import wave

    voice = _piper_voice()
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
        kwargs["speaker_wav"] = ref_wav
    else:
        kwargs["speaker"] = os.getenv("TTS_SPEAKER", "Ana Florence")
    model.tts_to_file(**kwargs)


# --- Kokoro-82M backend (local, CPU, onnxruntime — no torch) ------------------------------
# Small StyleTTS2 model; Spanish goes through espeak-ng g2p (Castilian phonology, e.g. the
# /θ/ in "cincuenta"). Model files come from the kokoro-onnx GitHub release (not HuggingFace).

_KOKORO = None
_KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"


def _ensure_kokoro_file(filename: str) -> str:
    import urllib.request

    cache = os.path.join(os.path.expanduser("~"), ".cache", "kokoro-onnx")
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
    voice = os.getenv("TTS_KOKORO_VOICE", "ef_dora")  # Spanish voices: ef_dora, em_alex, em_santa
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
        if _have_ffmpeg():
            out = os.path.join(media_dir, stem + ".mp3")
            _to_mp3(wav, out)
        else:
            out = os.path.join(media_dir, stem + ".wav")
            shutil.copyfile(wav, out)
    return os.path.basename(out)
