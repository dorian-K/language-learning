#!/bin/bash
# =============================================================================
# SMOKE TEST — validate the XTTS/H100 pipeline on a handful of phrases (~5) before
# committing to the full ~94-clip run. Same environment as the full job, just --limit.
#
# Because generation is idempotent, the clips this produces are KEPT: after you've
# listened and you're happy, run the full job (slurm/generate_number_audio.slurm) and it
# only synthesizes the remaining phrases.
#
# Steps:
#   1. Clone this repo on the cluster and cd into it.
#   2. Edit the account/partition TODOs below for your cluster.
#   3. sbatch slurm/smoke_number_audio.slurm
#   4. Copy the ~5 clips back and give them a listen:
#        rsync -av <cluster>:<repo>/anki/numbers/media/ ./anki/numbers/media/
#      (each file is es_<hash>.mp3; play them directly, or build the deck to hear them in context)
#   5. Happy? Submit slurm/generate_number_audio.slurm for the rest. Not happy? Tweak the voice
#      (TTS_REF_WAV / TTS_SPEAKER), `rm anki/numbers/media/*`, and re-run this smoke test.
# =============================================================================
#SBATCH --job-name=es-numbers-tts-smoke
#SBATCH --partition=c23g          # RWTH CLAIX-2023 H100 partition (name it explicitly)
#SBATCH --gres=gpu:1             # one GPU; on c23g that is an H100 (no h100 type qualifier)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-smoke-%j.out
### TODO: RWTH usually requires a project account — set yours (e.g. from `r_wlm_usage`): ###
###SBATCH --account=CHANGE_ME

set -euo pipefail

# Sets up $PY (uv if available, else a venv+pip fallback). See slurm/setup_env.sh.
source slurm/setup_env.sh

# Deterministic card JSON (gitignored, so absent on a fresh clone). No GPU/network.
$PY src/generate_numbers.py

export TTS_BACKEND=xtts
export COQUI_TOS_AGREED=1
# export TTS_REF_WAV="$PWD/ref.wav"        # <- Castilian reference clip for the best accent
# export TTS_SPEAKER="Ana Florence"        # built-in XTTS speaker used when no ref wav is set

# Just a varied handful of phrases to prove the pipeline + voice.
$PY src/generate_number_audio.py --limit 5

echo "SMOKE done. rsync anki/numbers/media/ back and listen before the full run."
