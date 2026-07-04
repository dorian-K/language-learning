# Portable, backend-aware Python-environment setup for the TTS SLURM jobs — sourced, not run.
# Uses `uv` if on PATH; otherwise a stdlib venv + pip (no uv needed). Sets $PY for the job.
#
# IMPORTANT: only the deps for $TTS_BACKEND are installed. The backends have incompatible
# onnxruntime pins (piper vs kokoro) and not every cluster Python has an onnxruntime wheel,
# so installing all of them at once fails to resolve. XTTS uses torch (bundles its own CUDA),
# NOT onnxruntime — so the cluster/XTTS path avoids onnxruntime entirely.
#
# Set TTS_BACKEND *before* sourcing this file (the SLURM scripts do). Default: piper.
#
# If your cluster needs a module for python3, uncomment and edit:
#   module load Python/3.11    # RWTH: `module spider Python` for the exact name

BACKEND="${TTS_BACKEND:-piper}"
case "$BACKEND" in
    piper)  DEPS="piper-tts>=1.2.0 huggingface-hub" ;;
    kokoro) DEPS="kokoro-onnx>=0.4.0 huggingface-hub" ;;
    xtts)   DEPS="coqui-tts>=0.24.0 huggingface-hub" ;;
    *) echo "setup_env: unknown TTS_BACKEND=$BACKEND (want piper|kokoro|xtts)" >&2; exit 1 ;;
esac

if command -v uv >/dev/null 2>&1; then
    echo "setup_env: using uv (backend=$BACKEND)"
    uv sync --extra "tts-$BACKEND"
    PY="uv run python"
else
    echo "setup_env: uv not found — venv + pip (backend=$BACKEND)"
    python3 -m venv .venv
    ./.venv/bin/python -m pip install -U pip
    # num2words is needed by generate_numbers.py; $DEPS is just the chosen backend.
    ./.venv/bin/python -m pip install num2words $DEPS
    PY="./.venv/bin/python"
fi
echo "setup_env: PY=$PY"
