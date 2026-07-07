#!/bin/bash
# =============================================================================
# Generate the vocab HEADWORD audio (Refold ES1K deck) with Coqui XTTS-v2 on an
# H100 (best quality). Sibling of generate_sentence_audio.slurm — this one voices
# the Spanish word (cue / primary target_es), not the example sentence.
#
# UNLIKE the numbers job, the input here (anki/<vocab>/) is NOT in the repo — it
# is gitignored and lives on your laptop. You MUST rsync it up before submitting:
#   rsync -av ./anki/ <cluster>:<repo>/anki/
#
# Round trip:
#   1. Clone this repo on the cluster and cd into it.
#   2. rsync your anki/ card JSON up (see above). Also rsync ref/ if you use clones.
#   3. Edit the TODO placeholders below (account / partition) for your cluster.
#   4. sbatch slurm/generate_word_audio.slurm
#   5. When it finishes, copy the audio back:
#        rsync -av <cluster>:<repo>/anki/word_audio/ ./anki/word_audio/
#   6. Locally: python src/make_anki_deck.py   → bundles the audio into the deck
#
# Idempotent: clips already in anki/word_audio/ are skipped, so re-submitting is safe.
# =============================================================================
#SBATCH --job-name=es-word-tts
#SBATCH --partition=c23g          # RWTH CLAIX-2023 H100 partition (name it explicitly)
#SBATCH --gres=gpu:1             # one GPU; on c23g that is an H100 (no h100 type qualifier)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00          # words are short and fewer than sentences
#SBATCH --output=slurm-%j.out
### TODO: RWTH usually requires a project account — set yours (e.g. from `r_wlm_usage`): ###
###SBATCH --account=CHANGE_ME

set -euo pipefail

# --- Environment ------------------------------------------------------------
# Backend must be set BEFORE setup_env (it installs only this backend's deps).
export TTS_BACKEND=xtts
export COQUI_TOS_AGREED=1                 # accept the Coqui XTTS license non-interactively

# If your vocab folder isn't the default, export VOCAB_SOURCE to match make_anki_deck.py.
# export VOCAB_SOURCE="Refold ES1K"

# PENINSULAR ACCENT: XTTS clones the accent from reference clips. Put several clean, single-speaker
# Castilian .wav clips (~6-15s each) in ref/ (rsync them up first) — XTTS rotates them per phrase
# for a Spain accent WITH voice variety. If ref/ is empty, we fall back to the curated built-in
# preset speakers (tts.py TTS_SPEAKERS), whose accent only rides on language="es" — so we warn
# loudly rather than silently assuming the accent is guaranteed.
export TTS_REF_DIR="$PWD/ref"
if ! ls "$TTS_REF_DIR"/*.wav >/dev/null 2>&1; then
    echo "WARNING: no .wav reference clips in $TTS_REF_DIR — using preset speakers only." >&2
    echo "         Accent is NOT guaranteed peninsular. Add Castilian clips to ref/ for the" >&2
    echo "         cloning-based guarantee, e.g. rsync -av ./ref/ <cluster>:<repo>/ref/" >&2
fi

# MIX IN THE CURATED PRESETS: also rotate XTTS's built-in speakers that were hand-picked as
# peninsular-sounding (the tts.py default TTS_SPEAKERS list, from the audition) alongside any
# cloned refs, for extra timbre variety. With ref/ empty this is a no-op (presets are used
# anyway); with refs present, unset it to use the cloned refs only.
export TTS_MIX_PRESETS=1

# Sets up $PY (uv if available, else a stdlib venv + pip fallback). See slurm/setup_env.sh.
source slurm/setup_env.sh

# --- Run --------------------------------------------------------------------
# Reads the headwords straight from the rsynced card JSON — no generation step needed here.
$PY src/generate_word_audio.py

echo "Audio written to anki/word_audio/ — rsync it back and run make_anki_deck.py locally."
