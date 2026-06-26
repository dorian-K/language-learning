# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run ruff check --fix && uv run ruff format   # lint + format
uv run pyright                                  # type check
uv run pytest                                   # run tests
uv run pytest tests/test_foo.py::test_bar       # run single test
```

Pre-commit hooks run ruff check/format and pyright automatically.

```bash
python -m src.bbc_noticias.bot                  # one-shot cron run
DRY_RUN=true python -m src.bbc_noticias.bot     # dry run (no messages sent)
python -m src.bbc_noticias.discord_bot          # long-running Discord bot
python -m src.bbc_noticias.telegram_bot         # long-running Telegram bot
python -m src.bbc_noticias.rss                  # print top 10 recent BBC stories
docker compose up -d                            # run all services (cron + bots + MQTT)
```

## Architecture

### BBC Noticias Bot (`src/bbc_noticias/`)

Three entry points backed by a shared MQTT message bus:

- **`bot.py`** — one-shot script (run by cron). Fetches BBC Mundo RSS → selects story via LLM → scrapes & simplifies article → sends to Discord webhook → enqueues story payload to MQTT topic `bbc/stories`.
- **`discord_bot.py`** — long-running. Subscribes to MQTT; delivers full story on `/historia` slash command or button click.
- **`telegram_bot.py`** — long-running. Same flow but for Telegram; responds to any user message.

Docker Compose runs four containers: `mosquitto` (Eclipse MQTT broker), `bbc-cron`, `bbc-discord`, `bbc-telegram`. The cron container uses a crontab file; the bot containers use separate Dockerfiles (`Dockerfile.bot`).

Key modules:
- `config.py` — all env-var config in a single `Config` dataclass; call `load()` to get a validated instance.
- `llm.py` — thin OpenAI-compatible client pointed at OpenRouter.
- `rss.py` / `scraper.py` — fetch and parse BBC Mundo stories. Use `bbc.com/mundo/...` URLs, not `bbc.co.uk` (returns 403).
- `selector.py` / `simplifier.py` — LLM calls for story selection and simplification.
- `mqtt.py` — `MQTTPublisher` (lazy-connect) and `MQTTSubscriber` (background thread, auto-reconnects). The subscriber runs outside asyncio — use `asyncio.new_event_loop()` + `loop.run_until_complete()` there, never `asyncio.run()`.
- `adapters/telegram.py`, `adapters/discord.py` — platform-specific message formatting. Telegram strips `||word||` spoiler markers to plain text (MessageEntity.SPOILER cannot mix with plain text in a single message).
- `sent_stories.py` — dedup guard; filters already-sent story links.
- `queue.py` / `queue_service.py` — story queue backed by a JSON file in `shared/`.
- `prompts.py` — LLM prompt strings.

### Anki Deck Pipeline (`src/`)

Sequential one-shot scripts; each reads/writes JSON files and `.apkg` files:

1. `transcribe_folder.py` — transcribes audio with WhisperX (needs GPU + HF token).
2. `extract_vocab_from_transcripts.py` — LLM extracts vocabulary from transcripts → `vocab/lt/`.
3. `extract_from_anki.py` — parses an existing `.apkg` via SQLite and exports notes to JSON.
4. `make_anki_deck.py` — reads `anki/lt/` (vocab) and `anki/irregular_verbs/` (conjugations), builds level-organised decks (`Lt::levelA1` … `Lt::levelB2`, `Lt::Conjugations`), exports `.apkg`.
5. `generate_verb_conjugations.py` — LLM generates 36 verbs × 9 tenses × 6 persons × 2 directions of conjugation cards.
6. `export_anki_snapshot.py` — reverse export: manually-edited `.apkg` → JSON for re-ingestion.

`src/llm.py` (top-level) is the shared LLM client for the Anki scripts; distinct from `src/bbc_noticias/llm.py`.

LLM prompts are in `.txt` files alongside the scripts (`vocab_extract_prompt.txt`, `verb_conjugation_prompt.txt`, `extract_from_anki_repackage_prompt.txt`, etc.).

## Environment

Copy `.env.example` to `.env`. Required: `OPENROUTER_API_KEY`. At least one of `DISCORD_WEBHOOK_URL` or `TELEGRAM_BOT_TOKEN` must be set for messages to be delivered. `TELEGRAM_CHAT_ID` is required when `TELEGRAM_BOT_TOKEN` is set.

## Pyright exclusions

`src/bbc_noticias/discord_bot.py` and `tests/` are excluded from pyright type-checking (see `pyproject.toml`).
