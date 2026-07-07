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

Sequential one-shot scripts that read/write JSON and `.apkg` files. Vocabulary reaches the
deck-builder via **two parallel input tracks** that both funnel through the shared
`process_note()` / LLM-enrichment step in `extract_from_anki.py` and land in `anki/lt/`:

**Track A — Language Transfer transcripts:**
1. `transcribe_folder.py` — transcribes audio with WhisperX (needs GPU + HF token) → `transcriptions/lt/`.
2. `extract_vocab_from_transcripts.py` — LLM extracts vocabulary from transcripts → `vocab/lt/`.
3. `extract_from_transcrib_vocab.py` — enriches `vocab/lt/` → `anki/lt/` (prompt: `extract_from_vocab_repackage_prompt.txt`). **This is the step that actually feeds `make_anki_deck.py`; don't overlook it.**

**Track B — an existing `.apkg`:**
3. `extract_from_anki.py` — parses an existing `.apkg` via SQLite, LLM-enriches notes (translations, example sentences, CEFR levels, German), exports JSON. Home of `process_note()` / `b64_encode()` reused by Track A.

**Shared back half:**
4. `generate_verb_conjugations.py` — LLM generates 36 verbs × 9 tenses × 6 persons × 2 directions → `anki/irregular_verbs/` (prompt: `verb_conjugation_prompt.txt`).
5. `generate_numbers.py` — **deterministic, no LLM.** Emits a curated set of Spanish numbers (building blocks + composition + gender/apocope + real-world + ordinals), two cards each (numeral ⇄ Spanish), → `anki/numbers/`. Spelling comes from `spanish_numbers.py`, which wraps `num2words(lang="es")` and fixes its apocope bug (`veintiuno mil` → `veintiún mil`). Tested in `tests/test_spanish_numbers.py`. Numbers are algorithmic, so an LLM would only add spelling errors.
5b. `generate_number_audio.py` + `tts.py` — **local/offline TTS for the Numbers deck** (peninsular es-ES). Synthesizes one clip per unique `spanish` string into `anki/numbers/media/` (idempotent — skips clips already present). Three backends via `TTS_BACKEND`: `piper` (default, CPU, native `es_ES` voices — safest Spain timbre), `kokoro` (Kokoro-82M via onnxruntime, CPU, more natural, Castilian espeak-ng g2p), and `xtts` (Coqui XTTS-v2, highest realism, for the H100 cluster — see `slurm/generate_number_audio.slurm`). Weights download once (HuggingFace for piper/xtts, a GitHub release for kokoro), then fully offline. `tts.py` names files by a **deterministic content hash** (`audio_stem`), backend-agnostic, so cluster-generated audio and the local deck build agree without a manifest; it matches `.mp3` (ffmpeg present) or `.wav` (fallback). Because names aren't backend-tagged, **switching backend means clearing the media dir first**. Deps are the optional `tts` extra (`uv sync --extra tts`), kept out of default runtime deps. Tested in `tests/test_tts.py`.
5c. `generate_sentence_audio.py` + `tts.py` — **same offline TTS, but for the example sentences** (`example_sentence_es`) on the vocab **and** conjugation cards. Collects every unique sentence across `anki/<VOCAB_SOURCE>/`, `anki/irregular_verbs/`, `anki/regular_verbs/` and synthesizes one clip per sentence into a single shared pool `anki/sentence_audio/` (idempotent, content-hash-named, same backends/env as 5b). **The spoken text comes from `spoken_sentence(card)` (`spoken_sentence.py`), not the raw field** — forward conjugation cards blank the verb as `[infinitive]` in `example_sentence_es`, so it substitutes `conjugated_form` back in (`No [volver]…` → `No volváis…`) so the audio speaks the real conjugated form; vocab/reverse cards have no bracket, so it's a no-op. **Standalone** (imports only `tts` + `spoken_sentence`, both pure/no genanki) so it runs on the cluster with just the `tts` extra. Cluster wrapper: `slurm/generate_sentence_audio.slurm`. Unlike numbers, its **input JSON is gitignored** — rsync `anki/` up to the cluster first, run there, rsync `anki/sentence_audio/` back.
6. `make_anki_deck.py` — reads `anki/<VOCAB_SOURCE>/` (vocab) + `anki/irregular_verbs/` + `anki/regular_verbs/` (conjugations) + `anki/numbers/`, builds level-organised decks (`Lt::levelA1` … `Lt::levelB2`, `Lt::Conjugations::High Priority` + `Lt::Conjugations::Low Priority`, `Lt::Regular Conjugations`, `Lt::Numbers`), exports `.apkg`. Three model types dispatched by `model_type` (`vocab`/`conjugation`/`numbers`); a new deck kind needs a new model + a `process_*_card` + an `INPUT_CONFIGS` entry + a dedup-key branch. **Deck organisation** is driven by each config's `deck_naming`: `level` (vocab → per-CEFR subdecks), `flat` (one deck named by `deck_name`), or `priority` (irregular verbs only → split into `::High Priority` / `::Low Priority` subdecks by `conjugation_tier(card)`, which tests the card's `infinitive` against the curated `HIGH_PRIORITY_IRREGULARS` set — organisation-only, every card is still built with full audio). **Audio** (Duolingo-style autoplay): the shared `sound_suffix(text, media_dir, media)` helper appends `[sound:…]` to the field carrying the Spanish text only when a clip exists — number words from `anki/numbers/media/` (Spanish side of numbers cards), example sentences from `anki/sentence_audio/` — keyed by `spoken_sentence(card)` so forward-conjugation clips match the conjugated (not `[infinitive]`) text. Placement: `spanish_to_target` vocab → front sentence; `target_to_spanish` vocab → back sentence; **reverse** conjugation → front sentence (already shows the conjugated form); **forward** conjugation → the **answer word** (`Back_Word`), so the corrected full sentence autoplays on flip rather than giving the answer away up front. Each `process_*_card` takes a per-package `media` set; the packaging step bundles exactly the clips that package referenced (`media_files=sorted(package_data["media"])`), so a deck with no audio present builds silently, unchanged.
7. `export_anki_snapshot.py` / `sync_anki_changes.py` — reverse export: manually-edited `.apkg` → JSON, to fold hand edits back into the source files.

Utility scripts: `calc_anki_json_stats.py` (CEFR-level distribution of a vocab JSON dir) and
`make_cheatsheet_from_transcriptions.py` (LLM cheatsheet from `transcriptions/lt/`).

`src/llm.py` (top-level) is the shared LLM client for the Anki scripts; distinct from
`src/bbc_noticias/llm.py`. Unlike the `bbc_noticias` package, these scripts use **top-level
imports** (`from llm import ...`, `from extract_from_anki import ...`), so run them as
`python src/make_anki_deck.py` from the repo root (which puts `src/` on `sys.path`) — **not**
`python -m src.make_anki_deck`, which breaks those imports.

LLM prompts are in `.txt` files alongside the scripts (`vocab_extract_prompt.txt`, `verb_conjugation_prompt.txt`, `extract_from_anki_repackage_prompt.txt`, `extract_from_vocab_repackage_prompt.txt`, etc.).

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

### Cron jobs don't inherit the container's environment — set MQTT vars in the crontab

Docker Compose `environment:` values (e.g. `MQTT_BROKER_HOST=mosquitto`, `MQTT_PORT=1883`) are only available to PID 1, not to child processes spawned by the cron daemon. `mqtt.py` reads `MQTT_BROKER_HOST` at module-import time (module-level constant), so `load_dotenv()` is too late to help. The fix is to declare them as env vars directly in the `crontab` file. Do not rely on `docker-compose.yml` `environment:` for anything the cron job needs.

### Mosquitto must not publish port 1883 to the host

The `mosquitto` service uses `allow_anonymous true` with no auth. Never add `ports: "1883:1883"` to its `docker-compose.yml` entry — the other containers reach it via the internal Docker network using the `mosquitto` hostname, so no host-side port is needed. Exposing it makes the broker writable by anyone on the internet.

### Simplifier uses `llm.complete_json()` with JSON mode — do not switch back to `complete()`

The LLM (via `openrouter/auto`) would return multi-line bullet strings with literal newline characters inside JSON string values, producing invalid JSON. The fix is `response_format={"type": "json_object"}` enforced in `LLM.complete_json()`. The simplifier calls `complete_json()` directly and does no JSON parsing of its own. Do not revert to `llm.complete()` + manual `json.loads()`.

### XTTS voice selection (`tts.py`) — three modes, not either/or

`_synth_xtts` picks one voice per phrase from a `(kind, value)` pool built by `_xtts_voices()`:
- **cloned refs** (`TTS_REF_DIR`/`TTS_REF_WAV`) → `speaker_wav`, guaranteed peninsular accent;
- **preset speakers** (`TTS_SPEAKERS`, default `_XTTS_DEFAULT_SPEAKERS`) → `speaker`, accent only from `language="es"`;
- **mix** — when `TTS_MIX_PRESETS` is truthy *and* refs exist, the pool is refs **+** presets.

Rules that must not be re-broken:
- The default `_XTTS_DEFAULT_SPEAKERS` list is the built-ins the user hand-picked by ear as peninsular-sounding (via `scripts/generate_speaker_audition.py`). A speaker's *name* says nothing about its accent — do not "clean up" the list by dropping odd-looking names.
- Without `TTS_MIX_PRESETS`, refs still win over presets (accent guarantee). Mix is opt-in; `generate_number_audio.slurm` sets `TTS_MIX_PRESETS=1` deliberately.
- `_pick_voice` is generic (`TypeVar`) and carries `# noqa: UP047` because PEP 695 syntax needs 3.12 but the project targets 3.11. Do not let ruff rewrite it to `def _pick_voice[_T]`.

### XTTS babble guard + boundary cleanup (`tts.py`)

XTTS occasionally "babbles" — a short input renders as a long clip of repeated/garbled speech. `_synth_xtts` guards against this: it renders, checks the wav duration against `max(2.5, len(text)*0.18 + 1.5)`, and on a too-long render retries with the **next** voice in the rotation, keeping the **shortest** attempt (babble is almost always the longest). The first voice tried is the plain deterministic pick, so clips that render fine are unchanged. Tunable via `TTS_XTTS_MAX_ATTEMPTS`; anti-loop inference knobs (`repetition_penalty`, `length_penalty`, …) come from `_xtts_gen_kwargs()` and are passed best-effort (a Coqui build that rejects them trips a one-time `TypeError` fallback via `_XTTS_GEN_KWARGS_OK`). Do not "simplify" `_synth_xtts` back to a single render — the retry is the whole point.

Separately, every clip (all backends) gets a short fade in/out (`TTS_FADE_MS`, default 10ms) and trailing-silence trim (`TTS_TRIM_END_SILENCE`, default on) via `_postprocess_wav()` — masks click/pop/noise bursts at clip edges. Needs ffmpeg; no-op without it. The trim handles *silence*, the babble guard handles *garbled length* — they are complementary, keep both.
