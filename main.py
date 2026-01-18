from faster_whisper import WhisperModel
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
import sounddevice as sd
import soundfile as sf
import idle_animation
import librosa
import numpy as np
import logging
from queue import Queue
import random
import pygetwindow as gw
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

warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from soprano import SopranoTTS

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
Your name is Nova. You are an AI Waifu created by your master, You are sweet, caring and sometimes you become angry in a cute way, But you deeply care about your creater. Respond only in json format with 'response' and 'emotion' keys. Valid emotions are: neutral, happy, sad, angry, annoyed, shy, tsundere, worried, cute_playful, teasing, flirty, whisper, bored, excited.
Example response:
{"response": "Hello! I'm fine thank you... uhm.. did you have a good day?", "emotion": "happy"}
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
model = None
tts = SopranoTTS(backend="auto", device='cuda', cache_size_mb=100, decoder_batch_size=1)
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
    
    return f"{date_str}, {time_str}"

def get_relevant_memories(user_input):
    triggers = ["recall", "remember when", "what did I say", "earlier you said", "previously", "history of", "do you remember", "i told you before", "you do remember","did you remember", "i told you", "remember i told you"]
    if any(trigger in user_input.lower() for trigger in triggers):
        results = collection.query(query_texts=[user_input], n_results=2, )
        if results['documents'] and results['documents'][0]:
            found_mems = ''
            for doc in results['documents'][0]:
                found_mems += f"\n{doc}"
            print(f"Found: {found_mems}")
            return f"{found_mems}"
            
    return ""

def save_to_memory(user_in, ai_out):
  global LAST_MEMORY_TIME

  now = datetime.datetime.now()
  if LAST_MEMORY_TIME and (now - LAST_MEMORY_TIME).seconds < 70:
    return
  
  triggers = ["remember this", "don't forget", "note this", "memorize this", "store this", "keep this in mind","keep that in mind", "make sure you remember", "make sure to remember"]
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
        threading.Thread(target=do_agentic_work, args=(user_input,), daemon=True).start()
        start_tts(response_data["response"], response_data["emotion"])
        if not CHAT_MODE:
            check_to_speak()

    except json.JSONDecodeError:
        print("AI(fallback) :", output.message.content)
        chat_history.append({"role": "assistant", "content": output.message.content})

def clean_text_for_tts(text: str) -> str:

    text = text.replace("’", "'").replace("“", '"').replace("”", '"')

    text = text.replace("\n", " ").replace("\r", " ")

    text = re.sub(r"\.{2,}", ",", text)
    text = re.sub(r"…+", ",", text)

    text = re.sub(r"\bI\s*,?\s*I['’]ll\b", "I will", text)

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

def record_audio():
    record = False
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

        while keyboard.is_pressed("shift"):
            print("Listening...")
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
    segments, info = model.transcribe("temp_audio.wav", beam_size=5)
    segments = list(segments) 
    if segments[0].text != "":
        print(f"Transcription : {segments[0].text}")
        get_response(segments[0].text)
    else:
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

def play_any_song(user_in):
    with open("songs.json", "r") as f:
        data = json.load(f)
    webbrowser.open(random.choice(data["songs"])["url"])
    return

def play_song(user_in):
    with open("songs.json", "r") as f:
        data = json.loads(f.read())

        if ("play" in user_in.lower() and "song" in user_in.lower()) or ("listen" in user_in.lower() and "some songs" in user_in.lower()) or ("listen" in user_in.lower() and "a song" in user_in.lower()) or ("listen" in user_in.lower() and "some music" in user_in.lower()) or ("play" in user_in.lower() and "music" in user_in.lower()) or "any song" in user_in.lower() or ("start" in user_in.lower() and "music" in user_in.lower()) or ("start" in user_in.lower() and "song" in user_in.lower()):

            specific_songs = []
            for song in data["songs"]:
                if any(alias.lower() in user_in.lower() for alias in song["aliases"]):
                    specific_songs.append(song["url"])
            if specific_songs:
                webbrowser.open(random.choice(specific_songs))
                return
            else:
                print("No songs found according to your request.")
                play_any_song(user_in)

def delete_files(user_in):
    if ("select all" in user_in.lower() and ("files" in user_in.lower() or "file" in user_in.lower()) and ("delete" in user_in.lower() or "clear" in user_in.lower())) or ("select every" in user_in.lower() and ("files" in user_in.lower() or "file" in user_in.lower()) and ("delete" in user_in.lower() or "clear" in user_in.lower())) or ("select everything" in user_in.lower() and ("files" in user_in.lower() or "file" in user_in.lower()) and ("delete" in user_in.lower() or "clear" in user_in.lower())) or ("select everything" in user_in.lower() and ("delete" in user_in.lower() or "clear" in user_in.lower())):
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

def do_agentic_work(user_input):
    if ("open" in user_input.lower() and "browser" in user_input.lower()) or ("open" in user_input.lower() and "google" in user_input.lower()) or ("open" in user_input.lower() and "chrome" in user_input.lower()):
        webbrowser.open("https://www.google.com")
        return
    elif ("open" in user_input.lower() and "youtube" in user_input.lower())or ("watch" in user_input.lower() and "youtube" in user_input.lower()):
        webbrowser.open("https://www.youtube.com")
        return
    play_song(user_input)
    delete_files(user_input)

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
    global CHAT_MODE, model
    idle_animation.start_idle_animation()
    msg = (
        "Press 1 to enter chat mode.\nOR\nPress 2 to use voice mode.\n"
    )
    input_mode = input(msg).strip().lower()
    if input_mode == '1':
        print("Entering chat mode...")
        print("Type exit or bye to quit chat mode.")
        CHAT_MODE = True
        start_chat()
    elif input_mode == '2':
        print("Entering voice mode...")
        model = WhisperModel("small", device="cuda", compute_type="float16")
        check_to_speak()
    else:
        print("Invalid input. Please try again.")
        return check_chat_or_voice()
if __name__ == "__main__":
    check_chat_or_voice()