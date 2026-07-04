# Portable Python-environment setup for the TTS SLURM jobs — sourced, not executed.
# Uses `uv` if it's on PATH; otherwise falls back to a stdlib venv + pip (no uv needed).
# Sets $PY to the python launcher the job should use for everything after.
#
# The pip fallback installs the same libraries as the pyproject "tts" extra, plus num2words
# (needed by generate_numbers.py). torch comes in via coqui-tts and bundles its own CUDA
# runtime, so a system CUDA module is usually NOT required for XTTS on the GPU node.
#
# If your cluster needs a module first (e.g. for python3), uncomment and edit:
#   module load Python/3.11    # RWTH: `module spider Python` to find the exact name

if command -v uv >/dev/null 2>&1; then
    echo "setup_env: using uv"
    uv sync --extra tts
    PY="uv run python"
else
    echo "setup_env: uv not found — falling back to python venv + pip"
    python3 -m venv .venv
    ./.venv/bin/python -m pip install -U pip
    ./.venv/bin/python -m pip install \
        num2words piper-tts kokoro-onnx coqui-tts huggingface-hub
    PY="./.venv/bin/python"
fi
echo "setup_env: PY=$PY"
