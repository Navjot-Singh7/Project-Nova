import os
from dotenv import load_dotenv
import discord
import ollama

load_dotenv()
discord_token = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)

@bot.event
async def on_ready():
    print(f"{bot.user.name} is online and ready to talk")

@bot.event
async def on_message(msg):
    if msg.author == bot.user:
        return

    print(msg.content)
    await msg.channel.send("hi im working")

bot.run(discord_token)
