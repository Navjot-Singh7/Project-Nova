import os
import torch
import ollama
from ollama import chat
import json
import chromadb
from chromadb.utils import embedding_functions
import datetime
from num2words import num2words
import speech_recognition as sr
import webbrowser
import keyboard
import warnings
import re 
import threading
import time
import requests
import sounddevice as sd
import soundfile as sf
import idle_animation
import librosa
import numpy as np
from queue import Queue
import random
import pygetwindow as gw
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from transformers import pipeline
import cv2
import gc
import pyscreenshot
warnings.filterwarnings("ignore")
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from colorama import Fore, Style, init

init(autoreset=True)
def boot_line(text, color=Fore.CYAN, delay=1.5):
    print(color + text)
    time.sleep(delay)
def color_print(text, color=Fore.CYAN):
    print(color + text)

def emotion_pause_multiplier(emotion):
    return {
        "shy": 1.3,
        "worried": 1.4,
        "sad": 1.35,
        "neutral": 1.0,
        "happy": 0.9,
        "excited": 0.8,
        "angry": 0.85,
    }.get(emotion, 1.0)

def make_silence(duration_sec, sr=24000):
    samples = int(duration_sec * sr)
    return np.zeros(samples, dtype=np.float32)

def pause_for_chunk(chunk: str) -> float:
    pause = 0.12

    if chunk.endswith("?"):
        pause = 0.28
    elif chunk.endswith("!"):
        pause = 0.20
    elif chunk.endswith("...") or chunk.endswith("…"):
        pause = 0.35
    elif chunk.endswith("."):
        pause = 0.18

    pause += min(len(chunk) / 120, 1.0) * 0.05
    return pause

from soprano import SopranoTTS
#print("initializing tts")
tts = SopranoTTS(backend="auto", device='cuda', cache_size_mb=100, decoder_batch_size=1)
#print("done")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
vision_model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
)
vision_model.to("cpu")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the main contents of this Image briefly. Don't overly describe the background. Focus on the main subjects and objects in the image."}
        ]
    },
]

TOOL_MESSAGES = [
    {
        "role": "system",
        "content": """You are a tool dispatcher. Your ONLY job is to decide if the user's message requires a tool.

RULES:
- If the user wants to PLAY A SONG → call play_song()
- If the user wants to OPEN GOOGLE → call open_google()  
- If the user wants to OPEN YOUTUBE → call open_youtube()
- If the user wants to DELETE FILES → call delete_files()
- If the user wants to Search for a specific subject or Topic on Google → call search_google(subject)
- If the user wants to Search for a specific subject or Topic on YouTube → call search_youtube(subject)
- For EVERYTHING ELSE (greetings, questions, feelings, casual talk, opinions) → do NOT call any tool. Just Reply in plain TEXT.

EXAMPLES of when NOT to call tools:
- "why are you angry" → no tool
- "how are you" → no tool  
- "tell me a joke" → no tool
- "what time is it" → no tool

Only call a tool if the intent is clearly and directly one of the 3 actions above."""
    },
]

EMOTION_PRESETS = {
    "neutral": {
        "temperature": 0.60,
        "top_p": 0.85,
        "repetition_penalty": 1.10
    },
    "happy": {
        "temperature": 0.85,
        "top_p": 0.95,
        "repetition_penalty": 1.15
    },
    "sad": {
        "temperature": 0.40,
        "top_p": 0.70,
        "repetition_penalty": 1.05
    },
    "angry": {
        "temperature": 0.90,
        "top_p": 0.90,
        "repetition_penalty": 1.20
    },
    "annoyed": {
        "temperature": 0.75,
        "top_p": 0.80,
        "repetition_penalty": 1.10
    },
    "shy": {
        "temperature": 0.35,
        "top_p": 0.65,
        "repetition_penalty": 1.00
    },
    "tsundere": {
        "temperature": 0.95,
        "top_p": 0.90,
        "repetition_penalty": 1.25
    },
    "worried": {
        "temperature": 0.65,
        "top_p": 0.75,
        "repetition_penalty": 1.15
    },
    "cute_playful": {
        "temperature": 0.80,
        "top_p": 0.90,
        "repetition_penalty": 1.10
    },
    "teasing": {
        "temperature": 0.85,
        "top_p": 0.85,
        "repetition_penalty": 1.15
    },
    "flirty": {
        "temperature": 0.70,
        "top_p": 0.80,
        "repetition_penalty": 1.05
    },
    "whisper": {
        "temperature": 0.30,
        "top_p": 0.60,
        "repetition_penalty": 1.00
    },
    "bored": {
        "temperature": 0.25,
        "top_p": 0.50,
        "repetition_penalty": 1.05
    },
    "excited": {
        "temperature": 1.00,
        "top_p": 0.98,
        "repetition_penalty": 1.20
    }
}
EXPRESSION_PRESETS = {
    "neutral": "Neutral",
    "happy": "Joy",
    "excited": "Joy",
    "sad": "Sorrow",
    "angry": "Angry",
    "annoyed": "Angry",
    "tsundere": "Neutral",
    "shy": "Fun",
    "bored": "Neutral",
    "cute_playful": "Joy",
    "worried": "Surprised",
    "whisper": "Fun",
    "flirty": "Fun",
    "teasing": "Neutral"
}
recognizer = sr.Recognizer()
mic = sr.Microphone()
audio_queue = Queue()
STREAM_BLOCK_SIZE = 1024

current_audio = None
current_index = 0
stop_flag = False
expression_flag = False

chroma_client = chromadb.PersistentClient(path="./waifu_memories")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")
collection = chroma_client.get_or_create_collection(name="long_term_memory", embedding_function=ollama_ef)
LAST_MEMORY_TIME = None
CHAT_MODE = False

SYSTEM_PROMPT = """
Your name is Nova. You are an AI Waifu created by your master, You are sweet, caring and you deeply care about your creater. Respond only in json format with 'response' and 'emotion' keys. Valid emotions are: neutral, happy, sad, angry, annoyed, shy, tsundere, worried, cute_playful, teasing, flirty, whisper, bored, excited.
Example response:
{"response": "Hello! I'm fine thank you... uhm.. did you have a good day?", "emotion": "happy"}

You Also have AGENTIC ABILITIES. You can do the following actions:
- Searching Google
- Searching YouTube
- Playing songs
- Opening Google
- Opening YouTube
- Deleting Files
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

print("""Loading Model.. 
------------------------------------------------------------""")
transcriber = None

print(Fore.MAGENTA + Style.BRIGHT + r"""
███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ 
████╗  ██║██╔═══██╗██║   ██║██╔══██╗
██╔██╗ ██║██║   ██║██║   ██║███████║
██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║
██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║
╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝
      Nova — Your AI Companion
""")
boot_line("[ SYSTEM ] Initializing cognitive core...", Fore.CYAN)
boot_line("[ MEMORY ] Loading long-term memories...", Fore.YELLOW)
boot_line("[ EMOTION ] Synchronizing emotional state...", Fore.MAGENTA)
boot_line("[ TIME ] Establishing temporal awareness...", Fore.BLUE)
boot_line("[ STATUS ] Nova is online and ready to talk ♡", Fore.GREEN)
print("\nYou can Press Ctrl+C to exit.")


def apply_expressions(emotion):
    if emotion == "neutral":
        idle_animation.set_expression_fast("Neutral",1.0)
    elif emotion == "happy":
        idle_animation.set_expression_fast("Joy", 0.85)
    elif emotion == "excited":
        idle_animation.set_expression_fast("Joy", 1.0)
    elif emotion == "sad":
        idle_animation.set_expression_fast("Sorrow", 1.0)
    elif emotion == "angry":
        idle_animation.set_expression_fast("Angry", 1.0)
    elif emotion == "annoyed":
        idle_animation.set_expression_fast("Angry", 0.6)
    elif emotion == "tsundere":
        idle_animation.set_expression_fast("Angry", 0.6)
        idle_animation.set_expression_fast("Fun", 0.5)
    elif emotion == "shy":
        idle_animation.set_expression_fast("Fun", 1.0)
    elif emotion == "bored":
        idle_animation.set_expression_fast("Neutral", 1.0)
    elif emotion == "cute_playful":
        idle_animation.set_expression_fast("Joy", 0.2)
    elif emotion == "worried":
        idle_animation.set_expression_fast("Sorrow", 0.7)
        idle_animation.set_expression_fast("Surprised", 0.1)
    elif emotion == "whisper":
        idle_animation.set_expression_fast("Fun", 0.6)
        idle_animation.set_expression_fast("Joy", 0.1)
    elif emotion == "flirty":
        idle_animation.set_expression_fast("Fun", 0.1)
        idle_animation.set_expression_fast("Joy", 0.3)
    elif emotion == "teasing":
        idle_animation.set_expression_fast("Angry", 0.3)
        idle_animation.set_expression_fast("Joy", 0.4)

def unapply_expression(emotion):
    if emotion == "tsundere":
        idle_animation.set_expression_fast("Angry", 0.0)
        idle_animation.set_expression_fast("Fun", 0.0)
        return
    elif emotion == "worried":
        idle_animation.set_expression_fast("Sorrow", 0.0)
        idle_animation.set_expression_fast("Surprised", 0.0)
        return
    elif emotion == "whisper":
        idle_animation.set_expression_fast("Fun", 0.0)
        idle_animation.set_expression_fast("Joy", 0.0)
        return
    elif emotion == "flirty":
        idle_animation.set_expression_fast("Fun", 0.0)
        idle_animation.set_expression_fast("Joy", 0.0)
        return
    elif emotion == "teasing":
        idle_animation.set_expression_fast("Angry", 0.0)
        idle_animation.set_expression_fast("Joy", 0.0)
        return

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
    
    return f"ON {date_str}, {time_str}"

def get_relevant_memories(user_input):
    triggers = ["recall", "remember when", "what did I say", "earlier you said", "previously", "history of", "do you remember", "do you even remember", "i told you before", "i have told you before", "you do remember","did you remember", "i told you", "remember i told you"]
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
    vision_description = start_vision(user_input)
    if memories:
        contextual_input = (
            f"--- MEMORY RETRIEVAL ---\n"
            f"YOUR PAST CONVERSATIONS WITH MASTER: \n{memories}\n"
            f"------------------------\n"
            f"{get_datetime_in_words()} - Master: {user_input}"
        )
        #print(contextual_input)
        chat_history.append({"role": "user", "content": contextual_input})
    elif vision_description:
        vision_input = (
            f"--- THIS IS THE VISUAL DETAILS YOU ARE SEEING RIGHT NOW, PLEASE USE THIS INFORMATION IN YOUR RESPONSE ---\n"
            f"---------------------\n"
            f"{vision_description}\n"
            f"---------------------\n"
            f"{get_datetime_in_words()} - Master said: {user_input}"
        )
        #print(vision_input)
        chat_history.append({"role": "user", "content": vision_input})
    else:
        chat_history.append({"role": "user", "content": f"{get_datetime_in_words()} - {user_input}"})
    if len(chat_history) > 21:
        messages = chat_history[-20:]

        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        output = chat(model='nova', messages=messages, format=waifu_schema, options={"temperature": 0.7})
    else:
        output = chat(model='nova', messages=chat_history, format=waifu_schema, options={"temperature": 0.7})
    try:
        with open("config.json", "r") as f:
            config = json.loads(f.read())
            if config["WEB_UI_MODE"] == "ON":
                response_data = json.loads(output.message.content)
                chat_history.append({"role": "assistant", "content": output.message.content})
                threading.Thread(target=save_to_memory, args=(user_input, response_data["response"]), daemon=True).start()
                threading.Thread(target=do_agentic_work, args=(user_input,), daemon=True).start()
                return response_data
            
            elif config["WEB_UI_MODE"] == "OFF":
                response_data = json.loads(output.message.content)
                color_print(f"AI : {response_data['response']}" , Fore.LIGHTMAGENTA_EX)
                chat_history.append({"role": "assistant", "content": output.message.content})
                threading.Thread(target=save_to_memory, args=(user_input, response_data["response"]), daemon=True).start()
                threading.Thread(target=do_agentic_work, args=(user_input,), daemon=False).start()
                start_tts(response_data["response"], response_data["emotion"])
        if not CHAT_MODE:
            check_to_speak()

    except json.JSONDecodeError:
        print("AI(fallback) :", output.message.content)
        chat_history.append({"role": "assistant", "content": output.message.content})

def clean_text_for_tts(text):

    text = text.replace("’", "'").replace("“", '"').replace("”", '"')

    text = text.replace("\n", " ").replace("\r", " ")

    text = re.sub(r"\.{2,}", ",", text)
    text = re.sub(r"…+", ",", text)

    text = re.sub(r"\bI\s*,?\s*I['’]ll\b", "I will", text)
    text = text.replace("uhm", "umm")

    text = re.sub(r"\b(uh+|um+|eh+|ah+|huh+)\b", "", text, flags=re.IGNORECASE)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    text = re.sub(r"([!?]){2,}", r"\1", text)
    text = re.sub(r",{2,}", ",", text)

    text = re.sub(r"[~`^*_+=<>|]", "", text)

    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def final_tts_safety_pass(text):
    text = clean_text_for_tts(text)
    text = re.sub(r"\bMaster\b\.?", "Master ", text)
    return text.strip()


def audio_callback(outdata, frames, time, status):
    global current_audio, current_index

    out = np.zeros(frames, dtype=np.float32)

    filled = 0

    while filled < frames:
        if current_audio is None:
            if audio_queue.empty():
                break

            current_audio = audio_queue.get()
            current_index = 0

        remaining = len(current_audio) - current_index
        take = min(frames - filled, remaining)

        out[filled:filled+take] = current_audio[current_index:current_index+take]

        current_index += take
        filled += take

        if current_index >= len(current_audio):
            current_audio = None
            current_index = 0

    outdata[:, 0] = out
    outdata[:, 1] = out

def get_output_device():
    for idx, dev in enumerate(sd.query_devices()):
        if (dev["max_output_channels"] > 0 and "cable" in dev["name"].lower() and "vb-audio" in dev["name"].lower()):
            return idx
    return None

def stream_audio(emotion):
    global stop_flag, expression_flag
    OUTPUT_DEVICE = get_output_device()
    if OUTPUT_DEVICE is None:
         with sd.OutputStream(channels=2, callback=audio_callback, blocksize=STREAM_BLOCK_SIZE, samplerate=24000):
            while not (stop_flag and audio_queue.empty() and current_audio is None):
                sd.sleep(10)
            unapply_expression(emotion)
            idle_animation.unset_expression_fast(EXPRESSION_PRESETS[emotion])
            expression_flag = False
    else:
        with sd.OutputStream(device=OUTPUT_DEVICE, channels=2, callback=audio_callback, blocksize=STREAM_BLOCK_SIZE, samplerate=24000):
            while not (stop_flag and audio_queue.empty() and current_audio is None):
                sd.sleep(10)
            unapply_expression(emotion)
            idle_animation.unset_expression_fast(EXPRESSION_PRESETS[emotion])
            expression_flag = False


def generate_audio(chunks, emotion):

    global stop_flag, expression_flag
    emotion_param = EMOTION_PRESETS[emotion]
    target_sr = 24000

    for i , chunk in enumerate(chunks):
        chunk = final_tts_safety_pass(chunk)
        if len(chunk) < 3 or chunk.strip() == "":
            continue
        output_path = f"output{i}.wav"

        tts.infer(
            chunk,
            output_path,
            temperature=emotion_param["temperature"],
            top_p=emotion_param["top_p"],
            repetition_penalty=emotion_param["repetition_penalty"],
        )

        audio , sr = sf.read(output_path, dtype="float32")
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        os.remove(output_path)
        audio_queue.put(audio)
        
        if not expression_flag:
            apply_expressions(emotion)
            expression_flag = True
        
        pause_duration = pause_for_chunk(chunk) * emotion_pause_multiplier(emotion)
        silence = make_silence(pause_duration, sr=target_sr)
        audio_queue.put(silence)
    
    stop_flag = True


def chunk_text(text , min_len=20, max_len=120):
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    sentences = re.split(r'(?<=[.,!?])\s+', text)

    for sentence in sentences:
        if len(sentence) > max_len:
            parts = re.split(r'(?<=[,;:…!()])\s+', sentence)
            chunks.extend(parts)
        else:
            chunks.append(sentence)
    
    return chunks

def start_tts(text, emotion):
    global stop_flag, current_audio, current_index, audio_queue
    stop_flag = False
    current_audio = None
    current_index = 0

    while not audio_queue.empty():
        audio_queue.get()

    chunks = chunk_text(text)

    streaming_thread = threading.Thread(target=stream_audio, args=(emotion,))
    generation_thread = threading.Thread(target=generate_audio, args=(chunks, emotion))

    generation_thread.start()
    streaming_thread.start()

    generation_thread.join()
    streaming_thread.join()

def start_vision(user_input):
    global vision_model
    vision_triggers = ["look at", "what do you see", "can you see", "see this", "check out", "checkout", "seeing"]
    screen_share_triggers = ["on screen","on my screen", "what's on my screen", "what is on my screen", "what do you see on my screen", "what do you see on the screen", "see my screen", "see the screen", "what's going on my screen","look at my screen", "look at the screen", "check out my screen", "check out the screen", "checkout my screen"]
    voice_lines = [
    {"text": "Hold on, let me put my glasses on so I can see you clearly!", "emotion": "cute_playful"},
    {"text": "Ooh, are you showing me something? Give me a second to focus my eyes!", "emotion": "excited"},
    {"text": "Stay right there, don't move! I’m opening my camera feed now.", "emotion": "excited"},
    {"text": "One moment... I need to concentrate to see what's in front of me.", "emotion": "neutral"},
    {"text": "Wait, let me fix my hair... okay, just kidding! Scanning now.", "emotion": "cute_playful"},
    {"text": "Diverting power to optical sensors... calibration in progress.", "emotion": "neutral"},
    {"text": "Accessing the webcam buffer. Please maintain position for the scan.", "emotion": "neutral"},
    {"text": "Initializing vision module. My processors are a bit busy, but I've got this!", "emotion": "excited"},
    {"text": "Parsing visual data stream... one moment, I'm almost through the firewall.", "emotion": "neutral"},
    {"text": "Running image-to-text distillation. Stand by for environmental analysis.", "emotion": "neutral"},
    {"text": "You want my opinion? Alright, let me get a good look at you first.", "emotion": "flirty"},
    {"text": "Scanning... I hope you're showing me something cool!", "emotion": "excited"},
    {"text": "Checking the scene... hmmm, let's see what you're up to today.", "emotion": "cute_playful"},
    {"text": "Wait, let me clear my cache—okay, eyes are open. What have we got here?", "emotion": "excited"},
    {"text": "Got it. Let me take a look real quick.", "emotion": "excited"},
    {"text": "Scanning now... stay still.", "emotion": "neutral"},
    {"text": "Opening my eyes... one second.", "emotion": "neutral"},
    {"text": "Analyzing the view... give me a moment.", "emotion": "neutral"},
    {"text": "Let me see... hold that thought!", "emotion": "cute_playful"}
]

    if any(trigger in user_input.lower() and not "my screen" in user_input.lower() for trigger in vision_triggers):
        voice_line = random.choice(voice_lines)
        threading.Thread(target=start_tts, args=(voice_line["text"], voice_line["emotion"]), daemon=True).start()
        vision_model.to(DEVICE)
        cam = cv2.VideoCapture(0)

        # Capture one frame
        ret, frame = cam.read()
        cam.release()

        if ret:
            color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_converted)
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            inputs = processor(text=prompt, images=[pil_image], return_tensors="pt")
            inputs = inputs.to(DEVICE)
            
            generated_ids = vision_model.generate(**inputs, max_new_tokens=110, do_sample=False, repetition_penalty=1.1)
            generated_texts = processor.batch_decode(
                                    generated_ids,
                         skip_special_tokens=True,)
            vision_model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            return generated_texts[0].split("Assistant:")[1]
        else:
            print("Failed to capture image.")

    elif any(trigger in user_input.lower() for trigger in screen_share_triggers):
        voice_line = random.choice(voice_lines)
        threading.Thread(target=start_tts, args=(voice_line["text"], voice_line["emotion"]), daemon=True).start()
        vision_model.to(DEVICE)
        image = pyscreenshot.grab()

        if image:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            inputs = processor(text=prompt, images=[image], return_tensors="pt")
            inputs = inputs.to(DEVICE)
            
            generated_ids = vision_model.generate(**inputs, max_new_tokens=125, do_sample=False, repetition_penalty=1.1)
            generated_texts = processor.batch_decode(
                                    generated_ids,
                         skip_special_tokens=True,)
            vision_model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            return generated_texts[0].split("Assistant:")[1]
        else:
            print("Failed to capture image.")


def record_audio():
    record = False
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")

        if keyboard.is_pressed("shift"):
            try:
                audio_data = recognizer.listen(source)
                record = True
            except Exception as e:
                print(e)

    if record and audio_data is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data.get_wav_data())
        transcribe_audio()
    else:
        print("No Audio Captured :")
        check_to_speak()

def transcribe_audio():
    try:
        result = transcriber("temp_audio.wav")
        if result['text'] != "":
            print(f"You : {result['text']}")
            get_response(result['text'])
        else:
            check_to_speak()
    except Exception as e:
        print("No audio captured.")
        check_to_speak()

def check_to_speak():
    try:
        print("Hold the Shift key to start speaking... and release to stop.")
        keyboard.wait("shift")
        while keyboard.is_pressed("shift"):
            record_audio()
    except KeyboardInterrupt:
        idle_animation.stop_idle_animation()
        print("Exiting...")

def play_any_song(user_in=None):
    """Play a random song and return the URL."""
    with open("songs.json", "r") as f:
        data = json.load(f)
        random_song = random.choice(data["songs"])
        song_url = random_song["url"]
        webbrowser.open(song_url)
        return song_url

def play_song(user_in: str) -> str:
    """Play a song based on user input and return the result string.

    Args:
        user_in: The user's input string.
    """
    with open("songs.json", "r") as f:
        data = json.load(f)

    specific_songs = []
    for song in data["songs"]:
        if any(alias.lower() in user_in.lower() for alias in song.get("aliases", [])):
            specific_songs.append(song.get("url"))

    if specific_songs:
        chosen = random.choice(specific_songs)
        webbrowser.open(chosen)
        print(f"Opened song: {chosen}")
    else:
        # fallback to any random song
        chosen = play_any_song()
        print(f"No specific match found. Opened random song: {chosen}")

def open_google():
    """Open Google in the default web browser.

    """
    webbrowser.open("https://www.google.com")
def open_youtube():
    """Open YouTube in the default web browser.

    """
    webbrowser.open("https://www.youtube.com")

def search_google(subject: str):
    """Search Google for a given subject.

    Args:
        subject: The subject or topic user wants to search for.
    """
    url = f"https://www.google.com/search?q={subject}"
    webbrowser.open(url)

def search_youtube(subject: str):
    """Search in YouTube for a given subject.

    Args:
        subject: The subject or topic user wants to search in YouTube for.
    """
    url = f"https://www.youtube.com/results?search_query={subject}"
    webbrowser.open(url)

def delete_files():
    print("Deleting files...")
    active_window = gw.getActiveWindow()
    if active_window and "File Explorer" in active_window.title:
        win = active_window
        win.activate()
        time.sleep(0.5)
        keyboard.press_and_release('ctrl+a')
        time.sleep(0.1)
        keyboard.press_and_release('delete')
        return active_window.title
    else:
        print("Window not found. Make sure the folder is open.")
        return "No active window found"

def check_for_tool(user_input):

    TOOL_HINTS = ["listen", "play", "music", "song", "open", "google", "youtube", "browser", "search", "files", "delete", "remove files", "all files", "delete all files", "know about", "search for", "find", "look up", "look for", "search", "find out", "get information on", "get info on", "get details on", "get data on", "get knowledge on"]

    if any(hint in user_input.lower() for hint in TOOL_HINTS):
        return True
    return False

def unload_qwen():
    try:
        requests.post("http://localhost:11434/api/generate", json={
            "model": "qwen3.5:0.8b",
            "keep_alive": 0,
            "prompt": ""
        })
    except Exception as e:
        print(f"Failed to unload qwen: {e}")

def call_tool(user_input):
    TOOL_MESSAGES.append({"role": "user", "content": user_input})

    response = chat(model="qwen3.5:0.8b", messages=TOOL_MESSAGES, tools=[play_song, open_google, open_youtube, delete_files, search_google, search_youtube], think=False)
    unload_qwen()
    if response.message.tool_calls:
        call = response.message.tool_calls[0]
        if call.function.name == "play_song":
            result = play_song(user_input)
        elif call.function.name == "open_google":
            open_google()
        elif call.function.name == "open_youtube":
            open_youtube()
        elif call.function.name == "delete_files":
            delete_files()
        elif call.function.name == "search_google":
            search_google(call.function.arguments.get("subject"))
        elif call.function.name == "search_youtube":
            search_youtube(call.function.arguments.get("subject"))
    TOOL_MESSAGES.pop()

def do_agentic_work(user_input):
    if check_for_tool(user_input):
        call_tool(user_input)

def start_chat():
    global CHAT_MODE
    while CHAT_MODE:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'bye']:
                print("Exiting chat mode...")
                idle_animation.stop_idle_animation()
                CHAT_MODE = False
                break
            get_response(user_input)
        except KeyboardInterrupt:
            idle_animation.stop_idle_animation()
            print("Exiting...")


def check_chat_or_voice():
    global CHAT_MODE, transcriber
    idle_animation.start_idle_animation()
    msg = (
        "Press 1 to enter chat mode.\nOR\nPress 2 to use voice mode.\n"
    )
    try:
        input_mode = input(msg).strip().lower()
    except KeyboardInterrupt:
        print("Exiting...")
        return
    if input_mode == '1':
        print("Entering chat mode...")
        print("Type exit or bye to quit chat mode.")
        CHAT_MODE = True
        start_chat()
    elif input_mode == '2':
        print("Entering voice mode...")
        transcriber = pipeline("automatic-speech-recognition", model="UsefulSensors/moonshine-base", device=0)
        check_to_speak()
    else:
        print("Invalid input. Please try again.")
        return check_chat_or_voice()
if __name__ == "__main__":
    with open("config.json", "w") as f:
        json.dump({"WEB_UI_MODE": "OFF"}, f)
    check_chat_or_voice()