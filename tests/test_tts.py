"""Tests for the audio-filename helpers in tts.py.

Only the pure, network-free helpers are covered (the actual synthesis needs model weights and a
backend). conftest.py already puts src/ on sys.path.
"""

import hashlib
import wave

import tts
from tts import AUDIO_EXTS, audio_basename, audio_stem, find_audio


def _write_wav(path, seconds, framerate=22050):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(b"\x00\x00" * int(framerate * seconds))


def test_audio_stem_is_deterministic_and_stable():
    # Same text -> same stem across calls, and a fixed known value (guards accidental scheme changes).
    assert audio_stem("cuarenta y siete") == audio_stem("cuarenta y siete")
    assert audio_stem("cuarenta y siete") == "es_50fff518d4bc8ccb"
    assert audio_stem("cien euros") != audio_stem("cuarenta y siete")


def test_audio_basename_uses_stem_and_extension():
    stem = audio_stem("veintiún")
    assert audio_basename("veintiún") == stem + ".mp3"
    assert audio_basename("veintiún", ".wav") == stem + ".wav"


def test_find_audio_matches_either_extension(tmp_path):
    text = "tres con cincuenta"
    assert find_audio(str(tmp_path), text) is None

    # mp3 is preferred when both exist
    (tmp_path / (audio_stem(text) + ".wav")).write_bytes(b"")
    assert find_audio(str(tmp_path), text) == audio_stem(text) + ".wav"
    (tmp_path / (audio_stem(text) + ".mp3")).write_bytes(b"")
    assert find_audio(str(tmp_path), text) == audio_stem(text) + ".mp3"
    assert AUDIO_EXTS[0] == ".mp3"


class _StubModel:
    """Fake XTTS model: renders a per-voice duration so we can test the babble-retry logic."""

    def __init__(self, durations):
        self.durations = durations  # voice value -> seconds
        self.calls = []

    def tts_to_file(self, **kwargs):
        voice = kwargs.get("speaker") or kwargs.get("speaker_wav")
        self.calls.append(voice)
        _write_wav(kwargs["file_path"], self.durations[voice])


def _pick_order(text, voices):
    """The order _synth_xtts walks the pool for ``text`` (deterministic first pick, then rotate)."""
    start = int(hashlib.sha1(text.encode()).hexdigest(), 16) % len(voices)
    return [voices[(start + i) % len(voices)][1] for i in range(len(voices))]


def test_synth_xtts_retries_babble_and_keeps_shortest(tmp_path, monkeypatch):
    # First voice tried babbles (10s); the next is fine (1s). Expect a retry, and the 1s clip kept.
    voices = [("preset", "A"), ("preset", "B")]
    first, second = _pick_order("dos", voices)
    stub = _StubModel({first: 10.0, second: 1.0})
    monkeypatch.setattr(tts, "_xtts_model", lambda: stub)
    monkeypatch.setattr(tts, "_xtts_voices", lambda: voices)
    monkeypatch.setenv("TTS_XTTS_MAX_ATTEMPTS", "4")

    out = tmp_path / "out.wav"
    label = tts._synth_xtts("dos", str(out))

    assert stub.calls == [first, second]  # first babbled → retried the next
    assert label == f"preset:{second}"
    assert abs(tts._wav_duration(str(out)) - 1.0) < 0.05


def test_synth_xtts_keeps_shortest_when_all_babble(tmp_path, monkeypatch):
    # Every voice babbles; the shortest (5s) is kept after exhausting attempts.
    voices = [("preset", "A"), ("preset", "B")]
    first, second = _pick_order("dos", voices)
    stub = _StubModel({first: 8.0, second: 5.0})
    monkeypatch.setattr(tts, "_xtts_model", lambda: stub)
    monkeypatch.setattr(tts, "_xtts_voices", lambda: voices)
    monkeypatch.setenv("TTS_XTTS_MAX_ATTEMPTS", "2")

    out = tmp_path / "out.wav"
    label = tts._synth_xtts("dos", str(out))

    assert stub.calls == [first, second]  # both tried
    assert label == f"preset:{second}"  # shortest kept
    assert abs(tts._wav_duration(str(out)) - 5.0) < 0.05
