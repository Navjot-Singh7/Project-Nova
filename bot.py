import os
from dotenv import load_dotenv
import discord
from ollama import chat
import threading
import time
import datetime
import json
import chromadb
from chromadb.utils import embedding_functions
import datetime
from num2words import num2words
from colorama import Fore, Style, init

load_dotenv()
discord_token = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)

chroma_client = chromadb.PersistentClient(path="./waifu_memories")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")
collection = chroma_client.get_or_create_collection(name="long_term_memory", embedding_function=ollama_ef)
LAST_MEMORY_TIME = None

SYSTEM_PROMPT = """
Your name is Nova. You are an AI Waifu created by your master, You are sweet, caring and sometimes you become angry in a cute way, But you deeply care about your creater. Respond only in json format with 'response' and 'emotion' keys. Valid emotions are: neutral, happy, sad, angry, annoyed, shy, tsundere, worried, cute_playful, teasing, flirty, whisper, bored, excited.
Example response:
{"response": "Hello! I'm fine thank you... umm.. did you have a good day?", "emotion": "happy"}
"""

waifu_schema = {
  "type": "object",
  "properties": {
    "response": {"type": "string"},
    "emotion": {
      "type": "string", 
      "enum": ["neutral", "happy", "sad", "angry", "annoyed", "shy", "tsundere", "worried", "cute_playful", "teasing", "flirty", "whisper", "bored", "excited"]
    }
  },
  "required": ["response", "emotion"],
  "additionalProperties": False
}

chat_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

def get_latest_memories(n=3):
    results = collection.get(
        include=["documents", "metadatas"]
    )
    if not results["documents"]:
        return ""

    memories = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        if meta and "timestamp_iso" in meta:
            memories.append((meta["timestamp_iso"], doc))

    if not memories:
        return ""
    
    memories.sort(key=lambda x: x[0], reverse=True)
    latest = memories[:n]

    formatted = ""
    for _, mem in latest:
        formatted += f"\n{mem}"

    return formatted

start_up_memories = get_latest_memories()
if start_up_memories:
    MEMORY_PROMPT = (
            f"\nThese following are your memories from previous conversation, Please use them.:\n"
            f"{start_up_memories}\n"
            f"These memories shape how you feel about him.")
    #print(MEMORY_PROMPT)
    chat_history.append({"role": "system", "content": MEMORY_PROMPT})


def get_datetime_in_words():
    now = datetime.datetime.now()
    month_name = now.strftime("%B")
    day_word = num2words(now.day, ordinal=True)
    year_word = num2words(now.year)
    date_str = f"{month_name} {day_word}, {year_word}"
  
    hour_12 = now.hour % 12 or 12
    hour_word = num2words(hour_12)
    minute_word = num2words(now.minute).replace("-", " ")
    am_pm = "am" if now.hour < 12 else "pm"
    time_str = f"at {hour_word} {minute_word} {am_pm}"
    
    return f"{date_str}, {time_str}"

def get_relevant_memories(user_input):
    triggers = ["recall", "remember when", "what did I say", "earlier you said", "previously", "history of", "do you remember", "do you even remember", "i told you before", "you do remember","did you remember", "i told you", "remember i told you"]
    if any(trigger in user_input.lower() for trigger in triggers):
        results = collection.query(query_texts=[user_input], n_results=2, )
        if results['documents'] and results['documents'][0]:
            found_mems = ''
            for doc in results['documents'][0]:
                found_mems += f"\n{doc}"
            #print(f"Found: {found_mems}")
            return f"{found_mems}"
            
    return ""

def save_to_memory(user_in, ai_out):
  global LAST_MEMORY_TIME

  now = datetime.datetime.now()
  if LAST_MEMORY_TIME and (now - LAST_MEMORY_TIME).seconds < 70:
    return
  
  triggers = ["remember this", "don't forget", "note this", "note that", "memorize this", "store this", "keep this in mind","keep that in mind", "make sure you remember", "make sure to remember"]
  ai_trigger = ["i will remember", "i'll keep in mind", "i will remember this always", "i won't forget", "i won't forget this moment", "it's in my heart", "i'll remember this always", "don't worry i won't forget", "i'm glad you told me this", "that must have been hard", "i will never forget", "i will not forget", "i won't ever forget", "i will never ever forget"]
  emotional_user_signals = ["scared", "sad", "lonely", "afraid", "worried", "hurt", "cry", "panic", "depressed", "anxious", "nervous", "upset", "frightened", "terrified", "distressed"]
  
  if any(phrase in user_in.lower() for phrase in triggers):
      mem_id = f"id_{collection.count()}"
      content = (
        f"[Memory | {get_datetime_in_words()}]\n"
        f"Context: emotional interaction with Master.\n"
        f"Master said: {user_in}\n"
        f"Nova replied: {ai_out}")
      print(f"Memory saved: {content}")
      collection.add(documents=[content], ids=[mem_id], metadatas=[{"timestamp_iso": datetime.datetime.now().isoformat(), "type": "episodic"}])
      LAST_MEMORY_TIME = now
      return
  
  if any(phrase in ai_out.lower() for phrase in ai_trigger) or any(emotion in user_in.lower() for emotion in emotional_user_signals):
      mem_id = f"id_{collection.count()}"
      content = (
        f"[Memory | {get_datetime_in_words()}]\n"
        f"Context: emotional interaction with Master.\n"
        f"Master said: {user_in}\n"
        f"Nova replied: {ai_out}")
      print(f"Memory saved: {content}")
      collection.add(documents=[content], ids=[mem_id], metadatas=[{"timestamp_iso": datetime.datetime.now().isoformat(), "type": "episodic"}])
      LAST_MEMORY_TIME = now


def get_response(user_input):
    memories = get_relevant_memories(user_input)
    if memories:
        contextual_input = (
            f"--- MEMORY RETRIEVAL ---\n"
            f"YOUR PAST CONVERSATIONS WITH MASTER: \n{memories}\n"
            f"------------------------\n"
            f"{get_datetime_in_words()} - Master: {user_input}"
        )
        #print(contextual_input)
        chat_history.append({"role": "user", "content": contextual_input})
    else:
        chat_history.append({"role": "user", "content": f"{get_datetime_in_words()} - {user_input}"})
    if len(chat_history) > 21:
        messages = chat_history[-20:]

        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        output = chat(model='ai-vtuber', messages=messages, format=waifu_schema, options={"temperature": 0.7})
    else:
        output = chat(model='ai-vtuber', messages=chat_history, format=waifu_schema, options={"temperature": 0.7})
    try:
        response_data = json.loads(output.message.content)
        color_print(f"AI : {response_data['response']}" , Fore.LIGHTMAGENTA_EX)
        chat_history.append({"role": "assistant", "content": output.message.content})
        threading.Thread(target=save_to_memory, args=(user_input, response_data["response"]), daemon=True).start()
        return response_data['response']

    except json.JSONDecodeError:
        print("AI(fallback) :", output.message.content)
        chat_history.append({"role": "assistant", "content": output.message.content})

def color_print(text, color=Fore.CYAN):
    print(color + text)

@bot.event
async def on_ready():
    print(f"{bot.user.name} is online and ready to talk")

@bot.event
async def on_message(msg):
    if msg.author == bot.user:
        return

    print(f"Message recieved : {msg.content}")
    reply = get_response(msg.content)
    await msg.channel.send(reply)

bot.run(discord_token)
