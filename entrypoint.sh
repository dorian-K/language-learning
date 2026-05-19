#!/bin/bash
set -e

# Graceful shutdown: kill all background children on SIGTERM/SIGINT
trap 'kill 0 2>/dev/null' SIGTERM SIGINT EXIT

mkdir -p /app/shared

# Copy crontab to /etc/cron.d/ (bind-mount lands at /app/crontab)
cp /app/crontab /etc/cron.d/bbc-noticias
chmod 0644 /etc/cron.d/bbc-noticias

# Export shared queue path so Python modules can find it
export SHARED_QUEUE_PATH=/app/shared/queue.json

# Start cron daemon (runs bot.py at scheduled times)
cron -f &
CRON_PID=$!

# Start Discord bot (long-running, button handler)
if [ -n "$DISCORD_BOT_TOKEN" ]; then
    echo "[entrypoint] Starting Discord bot..."
    /app/.venv/bin/python -m src.bbc_noticias.discord_bot &
else
    echo "[entrypoint] DISCORD_BOT_TOKEN not set — skipping Discord bot"
fi

# Start Telegram bot (long-running, /historia + button handler)
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    echo "[entrypoint] Starting Telegram bot..."
    /app/.venv/bin/python -m src.bbc_noticias.telegram_bot &
else
    echo "[entrypoint] TELEGRAM_BOT_TOKEN not set — skipping Telegram bot"
fi

echo "[entrypoint] All services started. Cron PID=$CRON_PID"

# Wait for all background processes — trap will kill them on exit
wait