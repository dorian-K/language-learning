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
python -m src.bbc_noticias.cron publish         # one-shot cron run
DRY_RUN=true python -m src.bbc_noticias.cron publish  # dry run (no messages sent)
python -m src.bbc_noticias.discord_bot          # long-running Discord bot
python -m src.bbc_noticias.telegram_bot         # long-running Telegram bot
python -m src.bbc_noticias.rss                  # print top 10 recent BBC stories
docker compose up -d                            # run all services (cron + bots + MQTT)
```

## Architecture

### BBC Noticias Bot (`src/bbc_noticias/`)

Three entry points backed by a shared MQTT message bus (Eclipse Mosquitto):

- **`cron.py`** — one-shot, invoked by the crontab at 08:00 CET. Calls `notifier.run()` which fetches RSS, selects + simplifies a story via LLM, then publishes the `StoryPayload` to MQTT topic `bbc/stories`.
- **`discord_bot.py`** — long-running. Subscribes to MQTT; when a story arrives it posts a headline to the stories channel and opens a Discord thread with the full article. Also handles `/historia` slash commands.
- **`telegram_bot.py`** — long-running. Two independent flows (see below).

Docker Compose runs four containers: `mosquitto` (Eclipse MQTT broker), `bbc-cron`, `bbc-discord`, `bbc-telegram`. The cron container uses a crontab file; the bot containers use separate Dockerfiles (`Dockerfile.bot`).

**Telegram has two independent flows with separate sent-story tracking:**
- **Channel (automatic):** MQTT message → `TelegramAdapter.send_story()` → posts full simplified article to `TELEGRAM_CHANNEL_ID`. Tracked in `data/channel_sent.txt` (`sent_stories.py`).
- **DM (on-demand):** Any private text message or `/historia` → `_dm_story_handler` → `get_story_for_user(user_id)` → sends next story this user hasn't seen. Tracked per-user in `data/dm_sent.json` (`dm_sent.py`). Uses 7-day look-back window for a larger pool.

The two tracking systems are independent: the same story can appear in both the channel and a DM.

Key modules:
- `config.py` — all env-var config in a single `Config` dataclass; call `load()` to get a validated instance.
- `llm.py` — thin OpenAI-compatible client pointed at OpenRouter.
- `rss.py` / `scraper.py` — fetch and parse BBC Mundo stories. **Use `bbc.com/mundo/...` URLs — `bbc.co.uk` returns 403.**
- `story_service.py` — full pipeline: fetch RSS → filter sent → select via LLM → simplify → return `StoryPayload`. Two entry points: `get_story_payload()` (channel, uses `sent_stories`) and `get_story_for_user(user_id)` (DM, uses `dm_sent`).
- `mqtt.py` — `MQTTPublisher` (lazy-connect, one-shot) and `MQTTSubscriber` (background thread, auto-reconnects, persistent sessions via `client_id` + `clean_session=False`). The subscriber runs outside asyncio — use `asyncio.run_coroutine_threadsafe()` with a captured running loop, never `asyncio.run()` from a non-main thread.
- `adapters/telegram.py`, `adapters/discord.py` — platform-specific delivery. **Telegram strips `||word||` spoiler markers to plain `(word)` text** — `MessageEntity.SPOILER` cannot mix with plain text in a single message.
- `sent_stories.py` — channel-level dedup (one flat `data/channel_sent.txt`).
- `dm_sent.py` — per-user DM dedup (`data/dm_sent.json`, keyed by Telegram user ID).
- `queue.py` / `queue_service.py` — shared file queue (`shared/queue.json`) used by the Discord button-click flow.
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

Copy `.env.example` to `.env`. Required: `OPENROUTER_API_KEY`. For Discord: `DISCORD_BOT_TOKEN` + `DISCORD_STORIES_CHANNEL_ID`. For Telegram: `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHANNEL_ID`.

## Pyright exclusions

`src/bbc_noticias/discord_bot.py` and `tests/` are excluded from pyright type-checking (see `pyproject.toml`).

---

## Bug history & design decisions

Keep this section updated when fixing bugs or making intentional design changes. Each entry states the rule and why, so it doesn't get re-introduced.

### Telegram MQTT subscriber — always use `post_init`, never `asyncio.get_event_loop()` before `run_polling()`

`TelegramAdapter.start_subscriber()` must be called from the `Application.post_init` async hook, **not** before `run_polling()`. `run_polling()` calls `asyncio.run()` internally which creates a new event loop. Any loop captured before that call is never started; `run_coroutine_threadsafe(...).result()` on it blocks forever, permanently deadlocking the MQTT thread.

Asserted in `start_subscriber()` — will raise immediately with a clear message if called at the wrong time.

### Telegram `post_channel()` — send the full article directly, no button

The channel posts the full simplified article via `_send_story_to()`. An earlier design tried headline + inline "Nueva historia" button → DM on click, but the `CallbackQueryHandler` was wired up while `post_channel()` never sent a button. The button flow was dead. The current design is simpler: full story goes to the channel, DM is a completely separate on-demand flow.

**Do not reintroduce `InlineKeyboardMarkup` or `enqueue_story` in `send_story()`** — the tests assert their absence.

### Telegram DM — per-user tracking, separate from channel

`get_story_for_user(user_id)` uses `dm_sent.py` (keyed by Telegram user ID, stored in `data/dm_sent.json`). `get_story_payload()` (channel/cron) uses `sent_stories.py` (`data/channel_sent.txt`). These are independent — do not merge them or reuse `sent_stories` for DM.

### MQTT persistent sessions — always pass `client_id` to `MQTTSubscriber`

`MQTTSubscriber` must be created with a stable `client_id` (e.g. `"bbc-discord-sub"`, `"bbc-telegram-sub"`). Without it, `clean_session=True` (default) means the broker discards queued QoS-1 messages while the subscriber is offline. With a stable ID, `clean_session=False` enables persistent sessions and offline delivery.

### Discord/Telegram env vars — not `DISCORD_WEBHOOK_URL`

The cron flow uses bot tokens, not webhook URLs. Required: `DISCORD_BOT_TOKEN` + `DISCORD_STORIES_CHANNEL_ID` for Discord; `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHANNEL_ID` for Telegram. `DISCORD_WEBHOOK_URL` is legacy (used only by the unused `bot.py` one-shot script).
