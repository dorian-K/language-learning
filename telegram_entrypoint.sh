#!/bin/bash
set -e
export SHARED_QUEUE_PATH=/app/shared/queue.json
mkdir -p /app/shared
exec /app/.venv/bin/python -m src.bbc_noticias.telegram_bot