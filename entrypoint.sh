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

# Start cron daemon (fires cron.py publish every 30 min)
cron -f &

echo "[entrypoint] Cron daemon started. PID=$!"

wait