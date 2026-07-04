# Portable, backend-aware Python-environment setup for the TTS SLURM jobs — sourced, not run.
# Uses `uv` if on PATH; otherwise a stdlib venv + pip (no uv needed). Sets $PY for the job.
#
# DISK: the venv (torch ~2 GB) and model weights (~2 GB) are far too big for an HPC home
# quota. If $HPCWORK is set (RWTH large work filesystem), the venv AND every cache/temp dir
# are placed under it, so home stays tiny. Only the small audio output lands in the repo.
#
# BACKENDS: only the deps for $TTS_BACKEND are installed. They pin incompatible onnxruntime
# versions (piper vs kokoro) and not every cluster Python has an onnxruntime wheel, so
# installing all at once fails to resolve. XTTS uses torch (bundles CUDA), NOT onnxruntime.
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

# --- Redirect venv + all caches/temp to $HPCWORK when available -----------------------------
if [ -n "${HPCWORK:-}" ]; then
    ROOT="$HPCWORK/es-numbers-tts"
    VENV_DIR="$ROOT/venv"
    export PIP_CACHE_DIR="$ROOT/pip-cache"
    export TMPDIR="$ROOT/tmp"                  # pip builds wheels here (can be large)
    export HF_HOME="$ROOT/hf"                  # huggingface_hub: piper voices + XTTS weights
    export TORCH_HOME="$ROOT/torch"
    export TTS_HOME="$ROOT/tts"                # coqui-tts model store
    export XDG_CACHE_HOME="$ROOT/xdg"          # kokoro-onnx model files (see tts.py)
    export UV_CACHE_DIR="$ROOT/uv-cache"
    export UV_PROJECT_ENVIRONMENT="$VENV_DIR"  # uv places its venv here too
    mkdir -p "$VENV_DIR" "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$TTS_HOME" \
        "$XDG_CACHE_HOME"
    echo "setup_env: using \$HPCWORK -> $ROOT"
else
    VENV_DIR=".venv"
    echo "setup_env: \$HPCWORK not set — using ./$VENV_DIR (fine locally, may exceed HPC home quota)"
fi

# --- Create/populate the environment --------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    echo "setup_env: using uv (backend=$BACKEND)"
    uv sync --extra "tts-$BACKEND"
    PY="uv run python"
else
    echo "setup_env: uv not found — venv + pip (backend=$BACKEND) at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install -U pip
    # num2words is needed by generate_numbers.py; $DEPS is just the chosen backend.
    "$VENV_DIR/bin/python" -m pip install num2words $DEPS
    PY="$VENV_DIR/bin/python"
fi
echo "setup_env: PY=$PY"
