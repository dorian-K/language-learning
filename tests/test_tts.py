"""Tests for the audio-filename helpers in tts.py.

Only the pure, network-free helpers are covered (the actual synthesis needs model weights and a
backend). conftest.py already puts src/ on sys.path.
"""

from tts import AUDIO_EXTS, audio_basename, audio_stem, find_audio


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
