"""
Discord bot — responds to slash commands and button clicks.
Runs alongside the cron container; they communicate via a shared queue file.
"""

import logging
import os
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import discord
from dotenv import load_dotenv

from src.bbc_noticias.config import load as load_config
from src.bbc_noticias.rss import fetch_stories
from src.bbc_noticias.scraper import fetch_article
from src.bbc_noticias.selector import select_best_story
from src.bbc_noticias.sent_stories import filter_unsent
from src.bbc_noticias.simplifier import simplify_article
from src.bbc_noticias.llm import LLM
from src.bbc_noticias.queue import pop_story, pending_count

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# --- Discord UI components ---

class StoryButton(discord.ui.Button):
    def __init__(self):
        super().__init__(
            label="📰 Nueva historia",
            style=discord.ButtonStyle.primary,
            custom_id="request_story",
        )

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        story = pop_story()
        if not story:
            # No queued story — fetch one live
            await interaction.followup.send(
                "⏳ No hay historias en cola, buscando una nueva...", ephemeral=True
            )
            try:
                story = await fetch_and_pick_story(interaction.client.llm)
            except Exception as e:
                await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)
                return
        try:
            await send_story_thread(interaction, story)
        except Exception as e:
            logger.error("[bot] send_story_thread failed, re-enqueueing story: %s", e)
            # Discord failed — put it back in the queue so it can be retried
            from src.bbc_noticias.queue import enqueue_story
            enqueue_story(story)
            await interaction.followup.send(f"❌ Error al enviar: {e}", ephemeral=True)
            return
        await interaction.followup.send("✅ ¡Historias enviadas!", ephemeral=True)


class StoryView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(StoryButton())


async def fetch_and_pick_story(llm: LLM) -> dict:
    """Fetch RSS stories and let LLM pick the best one (runs blocking work in thread pool)."""
    import asyncio

    def blocking() -> list[dict]:
        stories = fetch_stories(max_age_hours=48)
        logger.info("[bot] fetch_stories returned %d stories", len(stories))
        # Avoid already-sent stories so the button/slash always pick fresh ones
        unsent_links = set(filter_unsent([s["link"] for s in stories]))
        stories = [s for s in stories if s["link"] in unsent_links]
        logger.info("[bot] filter_unsent reduced to %d unsent stories", len(stories))
        return stories

    def blocking_with_llm(stories: list[dict]) -> dict:
        selected = select_best_story(stories, llm)
        logger.info("[bot] LLM selected: %s", selected.get("title", "unknown"))
        return selected

    stories = await asyncio.to_thread(blocking)
    if not stories:
        raise RuntimeError("No se encontraron historias en las últimas 48 horas.")
    return await asyncio.to_thread(blocking_with_llm, stories)


async def send_story_thread(interaction: discord.Interaction, story: dict) -> None:
    """Send a story as a message in the current channel, then create a thread on it."""
    channel = interaction.channel
    if not isinstance(channel, discord.TextChannel):
        await interaction.followup.send("❌ Este comando solo funciona en un canal de texto.", ephemeral=True)
        return

    logger.info("[bot] Sending story: %s", story.get("title", "unknown"))

    message_content = (
        f"📰 **{story['title']}**\n"
        f"🔗 {story['link']}\n\n"
        f"_Fuente: {story.get('source', 'BBC')} — {story.get('pub_date', '')[:10]}_"
    )

    # Post the message
    msg = await channel.send(message_content)
    logger.info("[bot] Posted message to channel, created thread")

    # Create a thread on that message
    thread_name = story["title"][:100]
    thread = await channel.create_thread(
        name=thread_name,
        type=discord.ChannelType.public_thread,
        message=msg,
        auto_archive_duration=60,
    )
    logger.info("[bot] Created thread: %s", thread.name)

    # Simplify and post full article in the thread (runs blocking work in thread pool)
    try:
        logger.info("[bot] Fetching article: %s", story["link"])
        article_text = await asyncio.to_thread(fetch_article, story["link"])
        if not article_text:
            logger.warning("[bot] fetch_article returned empty for: %s", story["link"])
            simplified = story.get("description", "Sin descripción disponible.")
        elif not isinstance(article_text, str):
            logger.error("[bot] fetch_article returned type %s, expected str", type(article_text).__name__)
            simplified = story.get("description", "Sin descripción disponible.")
        elif len(article_text) <= 50:
            logger.warning("[bot] fetch_article returned too short (%d chars): %s", len(article_text), story["link"])
            simplified = story.get("description", "Sin descripción disponible.")
        else:
            logger.info("[bot] Fetched %d chars, simplifying with LLM...", len(article_text))
            simplified = await asyncio.to_thread(simplify_article, article_text, interaction.client.llm)
            logger.info("[bot] LLM simplification done, got %d chars", len(simplified) if isinstance(simplified, str) else -1)
        # Guard against LLM returning non-string (e.g. JSON dict)
        if not isinstance(simplified, str):
            logger.error("[bot] simplify_article returned type %s, expected str", type(simplified).__name__)
            simplified = story.get("description", "Sin descripción disponible.")
        # Discord messages are max 2000 chars — split if needed
        for i in range(0, len(simplified), 1900):
            await thread.send(f"📖 **{story['title']}**\n\n{simplified[i : i + 1900]}")
        logger.info("[bot] Story sent successfully")
    except Exception as e:
        logger.error("[bot] Failed to simplify story: %s — %s", type(e).__name__, e)
        await thread.send(f"📖 {story.get('description', 'Sin descripción disponible.')}")


# --- Bot setup ---

intents = discord.Intents.default()
intents.message_content = True
intents.guild_messages = True
intents.messages = True

client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)


@tree.command(
    name="historia",
    description="Obtén una nueva historia de BBC Mundo (en español)",
)
async def historia(interaction: discord.Interaction):
    """Slash command: pick a story and send it with a thread."""
    await interaction.response.defer(ephemeral=False)

    story = pop_story()
    if not story:
        try:
            story = await fetch_and_pick_story(client.llm)
        except Exception as e:
            await interaction.followup.send(f"❌ No se pudo obtener historia: {e}")
            return

    try:
        await send_story_thread(interaction, story)
    except Exception as e:
        logger.error("[bot] send_story_thread failed, re-enqueueing story: %s", e)
        from src.bbc_noticias.queue import enqueue_story
        enqueue_story(story)
        await interaction.followup.send(f"❌ Error al enviar: {e}")
        return


@tree.command(
    name="cola",
    description="Muestra cuántas historias hay pendientes en la cola",
)
async def cola(interaction: discord.Interaction):
    count = pending_count()
    await interaction.response.send_message(
        f"📋 Hay **{count}** historia{'s' if count != 1 else ''} en cola.",
        ephemeral=True,
    )


@client.event
async def on_ready() -> None:
    logger.info("[bot] Logged in as %s", client.user)

    # Sync global slash commands
    synced = await tree.sync()
    logger.info("[bot] Synced %d global commands", len(synced))

    # Register persistent view (for button)
    client.add_view(StoryView())
    logger.info("[bot] Views registered")

    # Send button anchor message so users can click
    channel_id = os.getenv("BOT_CHANNEL_ID", "").strip()
    if channel_id:
        try:
            channel = await client.fetch_channel(int(channel_id))
            if isinstance(channel, discord.TextChannel):
                await channel.send(
                    "📰 ¡Haz clic en el botón para recibir una historia de BBC Mundo!",
                    view=StoryView(),
                )
                logger.info("[bot] Button anchor sent to channel %s", channel_id)
        except Exception as e:
            logger.warning("[bot] Could not send button anchor: %s", e)
    else:
        logger.warning("[bot] BOT_CHANNEL_ID not set — no button anchor message sent")

    # Init LLM
    client.llm = LLM()
    logger.info("[bot] LLM ready")


def main() -> None:
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        logger.error("[bot] DISCORD_BOT_TOKEN not set")
        sys.exit(1)

    client.run(token, log_handler=None)


if __name__ == "__main__":
    main()