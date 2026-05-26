# AGENTS.md

## Dev commands

```bash
ruff check --fix && ruff format   # lint + format
pyright                           # type check
pytest                            # run tests
```

Pre-commit hooks run all three (`ruff check --fix`, `ruff format`, `pyright`).

---

## Architecture

### Three entry points

| Command | Role |
|---|---|
| `python -m src.bbc_noticias.bot` | One-shot / cron. Fetches RSS, selects story, sends to Discord/Telegram |
| `python -m src.bbc_noticias.discord_bot` | Long-running. Handles `/historia` and button clicks |
| `python -m src.bbc_noticias.telegram_bot` | Long-running. Responds to any message with a story |

### MQTT queue

`bot.py` publishes selected stories to an MQTT broker. The Discord and Telegram bots run as separate containers ([*bbc-discord* and *bbc-telegram*](./docker-compose.yml)) subscribed to that topic. When a user clicks the button (Discord) or requests a story (Telegram), the bot fetches the queued story from MQTT and delivers it.

The MQTT subscriber runs in its own non-asyncio thread — do not use `asyncio.run()` there (throws `RuntimeError: asyncio.run() cannot be called from a running event loop`). Use `asyncio.get_event_loop().run_until_complete()` instead.

### Adapters

`src/bbc_noticias/adapters/telegram.py` and `adapters/discord.py` are wrapped by `telegram_bot.py` and `discord_bot.py`. All four app-level files depend on MQTT and share the same pubsub flow.

---

## Runtime quirks

- `DRY_RUN=true python -m src.bbc_noticias.bot` skips sending messages.
- Telegram message limit is 4096 chars — messages are split at that boundary in `_send_story_to`.
- MQTT `connect()` + `loop_start()` must happen **outside** any lock to avoid deadlocking with `stop()`.
- pyright does not type-check `discord_bot.py` or `tests/` ([pyproject.toml#L41](./pyproject.toml)).

### Telegram spoiler limitation

The LLM outputs translations as `||word||` markers (Discord spoiler format). Telegram's MessageEntity.SPOILER **cannot be mixed with plain text in a single message** — you either send all plain text or all spoilers with entity references. Therefore `||word||` is **stripped to plain text** in `_send_story_to`. Do not attempt to use MessageEntity.SPOILER with mixed content.

### RSS feeds

- `bbc.com/mundo/...` URLs work; `bbc.co.uk/mundo/...` return 403.
- Run `python -m src.bbc_noticias.rss` to print the top 10 recent stories (last 7 days) for review.

---

## Environment

- Copy `.env.example` → `.env` and fill in keys.
- Required: `OPENROUTER_API_KEY`
- One of: `DISCORD_WEBHOOK_URL` or `TELEGRAM_BOT_TOKEN`
- Optional: `TELEGRAM_CHAT_ID`, `TELEGRAM_CHANNEL_ID`, `MAX_AGE_HOURS`, `MAX_STORIES_FOR_SELECTION`, `DRY_RUN`