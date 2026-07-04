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
#SBATCH --partition=c23g          # RWTH CLAIX-2023 H100 partition (name it explicitly)
#SBATCH --gres=gpu:1             # one GPU; on c23g that is an H100 (no h100 type qualifier)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
### TODO: RWTH usually requires a project account — set yours (e.g. from `r_wlm_usage`): ###
###SBATCH --account=CHANGE_ME

set -euo pipefail

# --- Environment ------------------------------------------------------------
# Sets up $PY (uv if available, else a stdlib venv + pip fallback). See slurm/setup_env.sh.
source slurm/setup_env.sh

# --- Run --------------------------------------------------------------------
# anki/numbers/ is gitignored, so a fresh clone has no phrase list yet. Regenerate the
# number cards first — deterministic, no GPU/network (num2words). Cheap and idempotent.
$PY src/generate_numbers.py

export TTS_BACKEND=xtts
export COQUI_TOS_AGREED=1                 # accept the Coqui XTTS license non-interactively
# export TTS_REF_WAV="$PWD/ref.wav"       # <- uncomment to voice-clone a Castilian reference
# export TTS_SPEAKER="Ana Florence"       # built-in XTTS speaker used when no ref wav is set

$PY src/generate_number_audio.py

echo "Audio written to anki/numbers/media/ — rsync it back and run make_anki_deck.py locally."
