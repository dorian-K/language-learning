#!/bin/bash
# =============================================================================
# Audition XTTS-v2's ~58 built-in speakers on a peninsular-Spanish diagnostic line.
#
# Unlike generate_number_audio.slurm, this needs NO reference clips — it auditions the
# MODEL's own built-in speakers (not voice cloning) so you can pick which sound Castilian.
# Round trip:
#   1. Clone this repo on the cluster and cd into it.
#   2. Edit the TODO placeholders below (account / partition) for your cluster.
#   3. sbatch slurm/generate_speaker_audition.slurm
#   4. When it finishes, copy the clips back, e.g.:
#        rsync -av <cluster>:<repo>/speaker_audition/ ./speaker_audition/
#   5. Audition them; the sounds to judge are documented in scripts/generate_speaker_audition.py.
#
# Idempotent: clips already in speaker_audition/ are skipped, so re-submitting is safe.
# =============================================================================
#SBATCH --job-name=es-speaker-audition
#SBATCH --partition=c23g          # RWTH CLAIX-2023 H100 partition (name it explicitly)
#SBATCH --gres=gpu:1             # one GPU; on c23g that is an H100 (no h100 type qualifier)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
### TODO: RWTH usually requires a project account — set yours (e.g. from `r_wlm_usage`): ###
###SBATCH --account=CHANGE_ME

set -euo pipefail

# Backend must be set BEFORE setup_env (it installs only this backend's deps).
export TTS_BACKEND=xtts
export COQUI_TOS_AGREED=1                 # accept the Coqui XTTS license non-interactively

# Sets up $PY (uv if available, else a stdlib venv + pip fallback). See slurm/setup_env.sh.
source slurm/setup_env.sh

$PY scripts/generate_speaker_audition.py

echo "Clips written to speaker_audition/ — rsync them back and audition."
