#syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

ENV TZ=Europe/Berlin \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install cron + tini (init) + curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        cron tini curl ca-certificates bsdmainutils tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# Install uv for fast dependency install
RUN pip install uv

# Copy pyproject.toml first (for dependency caching)
COPY pyproject.toml uv.lock* .env.example* ./

# Install Python dependencies
RUN uv sync --no-dev

# Copy application code
COPY src/ ./src/

# Copy crontab and entrypoints
COPY crontab /app/crontab
COPY entrypoint.sh /app/entrypoint.sh
COPY bot_entrypoint.sh /app/bot_entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/bot_entrypoint.sh

# Healthcheck (curl localhost or check cron process)
HEALTHCHECK --interval=60s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health 2>/dev/null || pgrep -x cron > /dev/null || exit 1

# tini handles SIGTERM → graceful shutdown; runs as PID 1
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default: cron daemon (bbc-cron behavior)
# Override with docker-compose command or entrypoint to run bots
CMD ["/app/entrypoint.sh"]