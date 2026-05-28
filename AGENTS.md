# AGENTS.md

## Dev commands

```bash
uv run ruff check --fix && uv run ruff format   # lint + format
uv run pyright                                  # type check
uv run pytest                                   # run tests
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

The MQTT subscriber runs in its own non-asyncio thread — do not use `asyncio.run()` or `asyncio.get_event_loop()` there (both throw when called from a non-main thread). Use `asyncio.new_event_loop()` with `asyncio.set_event_loop()`, then `loop.run_until_complete()` — keep the loop alive and reuse it for all callbacks.

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

## Anki Deck Building

### Core Files

| File | Purpose |
|---|---|
| `src/extract_from_anki.py` | Parses existing `.apkg` Anki decks via SQLite, reconstructs them into `genanki` objects, exports notes as JSON for LLM processing. Key functions: `load_apkg_to_genanki()`, `note_to_llm_str()`, `process_note()`, `b64_encode()` |
| `src/make_anki_deck.py` | Reads processed JSON from `anki/lt/`, creates level-organized decks using `genanki`, exports final `.apkg` files. Uses unified "Symmetrical_ES_EN_DE_Vocab" card model with CSS styling for dual-direction vocab (Spanish↔English/German) |
| `src/export_anki_snapshot.py` | Reverse export: converts manually-edited `.apkg` deck back to JSON snapshot for re-ingestion by `make_anki_deck.py` |
| `src/extract_from_transcrib_vocab.py` | Processes transcript-based vocabulary from `vocab/lt/`, outputs to `anki/lt/` using same `process_note()` function |
| `src/calc_anki_json_stats.py` | Analyzes vocabulary JSON files, reports distribution by CEFR levels (A1-C2) for "earliest_level" and "mandatory_level" fields |

### Data Directories

| Directory | Purpose |
|---|---|
| `anki/` | Output directory for generated `.apkg` deck files |
| `vocab/lt/` | Input directory with raw vocabulary JSON from transcripts |

### Workflow

1. **Extract** — Load existing Anki `.apkg` → JSON via `extract_from_anki.py`
2. **Process** — LLM enriches with translations, example sentences, CEFR levels, German translations (prompt in `src/extract_from_anki_repackage_prompt.txt`)
3. **Generate** — Read JSON → level-organized `.apkg` via `make_anki_deck.py`
4. **Export Snapshot** — Capture manual Anki edits → JSON via `export_anki_snapshot.py`

### Dependencies

`genanki`, `anki` (see `requirements.txt`)

---

## Environment

- Copy `.env.example` → `.env` and fill in keys.
- Required: `OPENROUTER_API_KEY`
- One of: `DISCORD_WEBHOOK_URL` or `TELEGRAM_BOT_TOKEN`
- Optional: `TELEGRAM_CHAT_ID`, `TELEGRAM_CHANNEL_ID`, `MAX_AGE_HOURS`, `MAX_STORIES_FOR_SELECTION`, `DRY_RUN`