"""
Discord bot — responds to slash commands and button clicks.

Refactored to use PlatformAdapter pattern:
- DiscordAdapter for Discord-specific posting
- StoryService for platform-agnostic fetch/select/simplify pipeline
"""

import asyncio
import logging
import os

import discord
from discord import app_commands

from . import queue_service
from .story_service import get_story_payload
from .adapters.discord import DiscordAdapter


BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
STORIES_CHANNEL_ID = int(os.getenv("DISCORD_STORIES_CHANNEL_ID", "0"))
FORUM_CHANNEL_ID = int(os.getenv("DISCORD_FORUM_CHANNEL_ID", "0"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class BotClient(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.adapter = DiscordAdapter(client=self)
        self._tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        await self._tree.sync()


client = BotClient()


# ── Discord UI components ───────────────────────────────────────────────────

class StoryButton(discord.ui.Button):
    def __init__(self):
        super().__init__(
            label="📰 Nueva historia",
            style=discord.ButtonStyle.primary,
            custom_id="request_story",
        )

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        try:
            payload = await get_story_payload()
        except Exception as e:
            logger.error("[bot] get_story_payload failed (button): %s", e)
            await interaction.followup.send(
                "❌ No se encontró ninguna historia. Prueba otra vez.",
                ephemeral=True,
            )
            return

        if not payload:
            await interaction.followup.send(
                "❌ No se encontró ninguna historia. Prueba otra vez.",
                ephemeral=True,
            )
            return

        try:
            await client.adapter.send_story(payload)
            await interaction.followup.send("✅ ¡Historia enviada!", ephemeral=True)
        except Exception as e:
            logger.error("[bot] send_story failed, re-enqueueing: %s", e)
            queue_service.enqueue_story({
                "title": payload.headline,
                "link": payload.url,
            })
            await interaction.followup.send(f"❌ Error al enviar: {e}", ephemeral=True)


class StoryView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(StoryButton())


# ── Slash command ───────────────────────────────────────────────────────────

@client._tree.command(
    name="historia",
    description="Publica una historia nueva de BBC Mundo en el canal de historias.",
)
async def historia(interaction: discord.Interaction, hours: int = 48):
    await interaction.response.defer()

    try:
        payload = await get_story_payload(max_age_hours=hours)
    except Exception as e:
        logger.error("[historia] get_story_payload failed: %s", e)
        await interaction.followup.send(
            "❌ Error interno. Revisa los logs.", ephemeral=True
        )
        return

    if not payload:
        await interaction.followup.send(
            "❌ No se encontró ninguna historia. Prueba otra vez.", ephemeral=True
        )
        return

    try:
        await client.adapter.send_story(payload)
        await interaction.followup.send("✅ ¡Historia publicada!", ephemeral=True)
    except Exception as e:
        logger.error("[historia] send_story failed: %s", e)
        await interaction.followup.send(
            "❌ Error al publicar la historia. Revisa los logs.", ephemeral=True
        )


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    client.run(BOT_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
