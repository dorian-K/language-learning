#!/bin/bash
set -e

mkdir -p /app/shared
export SHARED_QUEUE_PATH=/app/shared/queue.json
exec /app/.venv/bin/python -m src.bbc_noticias.discord_bot
