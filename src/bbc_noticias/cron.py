"""
Cron daemon entrypoint for the bbc-cron container.

Uses python-crontab schedule approach:
- Container starts `cron -f`
- Crontab fires `python -m src.bbc_noticias.cron publish` at scheduled times
- This module checks `sys.argv[1] == "publish"` and exits after writing to queue

Usage:
    python -m src.bbc_noticias.cron publish   # Called by cron daemon
    python -m src.bbc_noticias.cron schedule # Run scheduler loop directly
"""

import asyncio
import logging
import sys

from .notifier import run as publish_scheduled_story, write_to_queue  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point — check argv and act accordingly."""
    if len(sys.argv) > 1 and sys.argv[1] == "publish":
        # Called by cron daemon: run once and exit
        logger.info("[cron] Cron trigger received — publishing story")
        success = asyncio.run(publish_scheduled_story())
        sys.exit(0 if success else 1)
    else:
        # Fallback: run scheduler loop (not normally used since we use cron daemon)
        logger.warning("[cron] No argv[1]=publish — scheduler loop not implemented. Use cron daemon.")
        sys.exit(1)


if __name__ == "__main__":
    main()