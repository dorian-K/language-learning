#!/bin/bash
# =============================================================================
# Generate the Numbers-deck audio with Coqui XTTS-v2 on an H100 (best quality).
#
# This sandbox has no GPU, so the high-realism XTTS pass runs here, on the cluster.
# Round trip:
#   1. Clone this repo on the cluster and cd into it.
#   2. Edit the TODO placeholders below (account / partition) for your cluster.
#   3. (Optional) drop a Castilian reference clip at ref.wav and set TTS_REF_WAV to
#      voice-clone it — gives the most accurate Spain accent. Otherwise a built-in
#      XTTS speaker is used.
#   4. sbatch slurm/generate_number_audio.slurm
#   5. When it finishes, copy the audio back to your machine, e.g.:
#        rsync -av <cluster>:<repo>/anki/numbers/media/ ./anki/numbers/media/
#   6. Locally: python src/make_anki_deck.py   → bundles the audio into Numbers.apkg
#
# Idempotent: clips already in anki/numbers/media/ are skipped, so re-submitting is safe.
# =============================================================================
#SBATCH --job-name=es-numbers-tts
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
### TODO: set these for your cluster ###
###SBATCH --account=CHANGE_ME
###SBATCH --partition=CHANGE_ME

set -euo pipefail

# --- Environment ------------------------------------------------------------
# Adjust to your cluster's conventions. XTTS needs PyTorch with CUDA; prefer the
# cluster's module system if it provides torch, otherwise let uv/pip resolve it.
# module load cuda/12.4 python/3.11        # <- example; uncomment/edit for your site

# Install deps (Piper + Coqui XTTS) into the project venv. `uv` is recommended:
uv sync --extra tts

# --- Run --------------------------------------------------------------------
# anki/numbers/ is gitignored, so a fresh clone has no phrase list yet. Regenerate the
# number cards first — deterministic, no GPU/network (num2words). Cheap and idempotent.
uv run python src/generate_numbers.py

export TTS_BACKEND=xtts
export COQUI_TOS_AGREED=1                 # accept the Coqui XTTS license non-interactively
# export TTS_REF_WAV="$PWD/ref.wav"       # <- uncomment to voice-clone a Castilian reference
# export TTS_SPEAKER="Ana Florence"       # built-in XTTS speaker used when no ref wav is set

uv run python src/generate_number_audio.py

echo "Audio written to anki/numbers/media/ — rsync it back and run make_anki_deck.py locally."
