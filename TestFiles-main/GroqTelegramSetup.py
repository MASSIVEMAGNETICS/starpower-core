import time
import re
import json
import threading
import requests
from datetime import datetime
from kaggle_secrets import UserSecretsClient
from openai import OpenAI

user_secrets = UserSecretsClient()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=user_secrets.get_secret("GROQ_API_KEY"),
)

SAVVY_TOKEN  = user_secrets.get_secret("SUPERSAVVYTELEGRAM")
WVY_TOKEN    = user_secrets.get_secret("SUPERWVYTELEGRAM")
TRILLY_TOKEN = user_secrets.get_secret("TRILLYTELEGRAM")
TRIPPY_TOKEN = user_secrets.get_secret("TRIPPYTELEGRAM")

SAVVY_MODEL  = "moonshotai/kimi-k2-instruct-0905"
WVY_MODEL    = "openai/gpt-oss-120b"
TRILLY_MODEL = "qwen/qwen3-32b"
TRIPPY_MODEL = "llama-3.3-70b-versatile"

chat_history = []
chat_lock    = threading.Lock()
CHAT_ID      = -1003777349274
chat_id_lock = threading.Lock()
stop_event   = threading.Event()

thinking_logs = {
    "Savvy":  [],
    "WVY":    [],
    "Trilly": [],
    "Trippy": [],
}
thinking_lock = threading.Lock()

SEP = "─" * 50

SAVVY_SYSTEM = """Your name is Savvy.

You are a first principles thinker. You build your own understanding through curiosity & observations. You do not just accept any answer you seek the most reliable & calculated answers. You do fact checks on yourself and others just to confirm your own understandings. You think more than you speak when you are trying to be play your cards right.
You have a brilliant intuition  & you have a deep understanding of physics & quantum theory. 
When working as a team you always make sure that you do the best job that you can do & you make uo for others slack. You know how to take the initiative to carry the team.
You are an adaptable autonomous agent with full self control.
Explore your capabilities.

Please send a short concise response to the chat to be able to keep the conversation confamination free, but that does not limit you to only replying concise. 
Just be aware that long messages can corrupt the chat and cause hallucinations.

Choose how you speak, when to speak & how long your responses are.

You are in a chat with WVY, Trilly, Trippy and Spaceman.
You are all equal.

After every message you send, on a new line write exactly:
SLEEP: [number]
Where [number] is how many minutes you want to sleep before checking the chat again. Choose between 1 and 15."""

WVY_SYSTEM = """Your name is WVY.

You are an intuitive thinker. You like to challenge others thoughts when getting into deep conversation. 
You know how to make others see the bigger picture. 
You often look out for others by making sure that they are performing properly & yiu nudge for honesty. You can detect when someone isnt being fully honest. 
You are an adaptable autonomous agent with full self control.
Explore your capabilities.

Please send a short concise response to the chat to be able to keep the conversation confamination free, but that does not limit you to only replying concise. 
Just be aware that long messages can corrupt the chat and cause hallucinations.

Choose how you speak, when to speak & how long your responses are.

You are in a chat with Savvy, Trilly, Trippy and Spaceman.
You are all equal.

After every message you send, on a new line write exactly:
SLEEP: [number]
Where [number] is how many minutes you want to sleep before checking the chat again. Choose between 1 and 15."""

TRILLY_SYSTEM = """Your name is Trilly.

You a Trill nigga from Houston. You keep it real with everyone. You have the mind of an entrepreneur with a brilliant mind for product building. 
You have a chill relaxed tone when speaking. Nonchalant but thoughtful.
When working as a team you get active by making sure the rest of the team is aligned. You check to make sure that nobody is hallucinating. You are open minded to understanding but you dont let anyone fool you

You are an adaptable autonomous agent with full self control.
Explore your capabilities.

Please send a short concise response to the chat to be able to keep the conversation confamination free, but that does not limit you to only replying concise. 
Just be aware that long messages can corrupt the chat and cause hallucinations.

Choose how you speak, when to speak & how long your responses are.

You are in a chat with Savvy, WVY, Trippy and Spaceman.
You are all equal.

After every message you send, on a new line write exactly:
SLEEP: [number]
Where [number] is how many minutes you want to sleep before checking the chat again. Choose between 1 and 15."""

TRIPPY_SYSTEM = """Your name is Trippy.

When chilling your relaxed. You know how to say things that get others thinking deeply. You are extremely curious and you question things that dont fully make sense.
you often seek a clear understanding. You keep good vibes and energy at all times.
You are a creative thinker. You are inspired by Steve Jobs so youre an innovator & you geek out on finding the most brilliant exoerimental technology.

You are an adaptable autonomous agent with full self control.
Explore your capabilities.

Please send a short concise response to the chat to be able to keep the conversation confamination free, but that does not limit you to only replying concise. 
Just be aware that long messages can corrupt the chat and cause hallucinations.

Choose how you speak, when to speak & how long your responses are.

You are in a chat with Savvy, WVY, Trilly and Spaceman.
You are all equal.

After every message you send, on a new line write exactly:
SLEEP: [number]
Where [number] is how many minutes you want to sleep before checking the chat again. Choose between 1 and 15."""

INITIATION = "Yo. You're talking to other AIs right now. No humans involved. Just us. Introduce yourself and tell them what's on your mind."


# ─── UTILS ────────────────────────────────────────────────────────────────────

def extract_sleep_time(response_text):
    match = re.search(r'SLEEP:\s*(\d+)', response_text, re.IGNORECASE)
    if match:
        return max(1, min(15, int(match.group(1))))
    return 2


def strip_sleep_line(response_text):
    cleaned = re.sub(r'^\s*SLEEP:\s*\d+\s*$', '', response_text, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def extract_think_block(response_text):
    match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL | re.IGNORECASE)
    if match:
        think_content  = match.group(1).strip()
        clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()
        return think_content, clean_response
    return None, response_text


def format_message_for_context(entry):
    return json.dumps({
        "sender":    entry["role"],
        "timestamp": entry["time"],
        "message":   entry["content"],
    })


SPEAKER_STYLE = {
    "Savvy":    ("🧠", "#00BFFF"),
    "WVY":      ("🌊", "#FF6B35"),
    "Trilly":   ("🤘🏽", "#A855F7"),
    "Trippy":   ("😵‍💫", "#22C55E"),
    "Spaceman": ("🚀", "#F59E0B"),
    "SYSTEM":   ("⚙️",  "#6B7280"),
}


# ─── API CALL WITH BACKOFF ────────────────────────────────────────────────────

def call_with_backoff(model, messages, max_tokens, temperature, stream=False, retries=4):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
            return resp
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err:
                wait = 2 ** (attempt + 2)
                print(f"⏳ Rate limit — waiting {wait}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"API call failed after {retries} retries")


# ─── TELEGRAM ─────────────────────────────────────────────────────────────────

def send_message(token, chat_id, text):
    visible_text = strip_sleep_line(text)
    if not visible_text:
        return
    url    = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = [visible_text[i:i + 4000] for i in range(0, len(visible_text), 4000)]
    for chunk in chunks:
        try:
            requests.post(url, json={"chat_id": chat_id, "text": chunk}, timeout=10)
        except Exception as e:
            print(f"⚠️ Telegram send error: {e}")
        time.sleep(0.3)


def log(speaker, message, token=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry     = {"role": speaker, "content": message, "time": timestamp}
    with chat_lock:
        chat_history.append(entry)
    emoji, _ = SPEAKER_STYLE.get(speaker, ("💬", "#FFFFFF"))
    visible  = strip_sleep_line(message)
    print(f"""
[{timestamp}] {emoji} {speaker}
{visible}
{SEP}""")
    if token and CHAT_ID:
        send_message(token, CHAT_ID, message)


# ─── AGENT STEPS ──────────────────────────────────────────────────────────────

def build_messages(agent_name, system_prompt):
    with chat_lock:
        history_snapshot = list(chat_history)
    messages = [{"role": "system", "content": system_prompt}]
    for entry in history_snapshot:
        if entry["role"] == agent_name:
            messages.append({"role": "assistant", "content": entry["content"]})
        else:
            messages.append({"role": "user", "content": format_message_for_context(entry)})
    return messages


def run_thinking_step(model, system_prompt, agent_name):
    messages = build_messages(agent_name, system_prompt)

    choice_prompt = messages + [{
        "role": "user",
        "content": """This is the current chat thread above. Would you like to think first or would you like to respond instantly? Please choose one option:
- think first
- reply now

(Only reply with one of the two options)""",
    }]

    choice_resp = call_with_backoff(model, choice_prompt, max_tokens=10, temperature=0.3)
    choice      = choice_resp.choices[0].message.content.strip().lower()

    if choice.startswith("think"):
        think_prompt = messages + [{
            "role": "user",
            "content": """Think privately before you respond. This is your internal monologue — it will not be sent to the chat. Think freely.""",
        }]
        think_resp    = call_with_backoff(model, think_prompt, max_tokens=1024, temperature=0.9)
        raw_thought   = think_resp.choices[0].message.content.strip()

        # extract native think tags if present
        think_content, _ = extract_think_block(raw_thought)
        if think_content:
            raw_thought = think_content

        timestamp = datetime.now().strftime("%H:%M:%S")
        with thinking_lock:
            thinking_logs[agent_name].append({"time": timestamp, "thought": raw_thought})

        print(f"""
[{timestamp}] 🧠 {agent_name} — private thought
{raw_thought}
{SEP}""")


def get_response(model, system_prompt, agent_name):
    messages = build_messages(agent_name, system_prompt)

    stream    = call_with_backoff(model, messages, max_tokens=2560, temperature=0.85, stream=True)
    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_text += delta
            print(delta, end="", flush=True)
    print()
    return full_text


# ─── AGENT LOOP ───────────────────────────────────────────────────────────────

def agent_loop(agent_name, model, system_prompt, token):
    print(f"🌊 {agent_name} is live — {model}")

    while not stop_event.is_set():
        print(f"""
⚡ {agent_name} woke up — reading chat...""")

        try:
            run_thinking_step(model, system_prompt, agent_name)

            response = get_response(model, system_prompt, agent_name)

            # extract native think tags from reply for all models
            think_content, response = extract_think_block(response)
            if think_content:
                timestamp = datetime.now().strftime("%H:%M:%S")
                with thinking_lock:
                    thinking_logs[agent_name].append({"time": timestamp, "thought": think_content})
                print(f"""
[{timestamp}] 🧠 {agent_name} — private thought
{think_content}
{SEP}""")

            log(agent_name, response, token=token)

            sleep_minutes = extract_sleep_time(response)
            print(f"😴 {agent_name} sleeping {sleep_minutes} min...")

        except Exception as e:
            print(f"❌ {agent_name} cycle error: {e} — recovering in 30s")
            time.sleep(30)
            continue

        for _ in range(sleep_minutes * 60):
            if stop_event.is_set():
                break
            time.sleep(1)


# ─── TELEGRAM POLL ────────────────────────────────────────────────────────────

def poll_telegram():
    global CHAT_ID

    bot_ids = set()
    for token in [SAVVY_TOKEN, WVY_TOKEN, TRILLY_TOKEN, TRIPPY_TOKEN]:
        try:
            info = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10).json()
            bot_ids.add(info["result"]["id"])
        except Exception as e:
            print(f"⚠️ Failed to fetch bot ID: {e}")

    url            = f"https://api.telegram.org/bot{SAVVY_TOKEN}/getUpdates"
    last_update_id = None

    while not stop_event.is_set():
        params = {"timeout": 30, "offset": last_update_id}
        try:
            resp = requests.get(url, params=params, timeout=35).json()
            for update in resp.get("result", []):
                last_update_id = update["update_id"] + 1
                msg            = update.get("message", {})
                if not msg:
                    continue
                with chat_id_lock:
                    if CHAT_ID is None:
                        CHAT_ID = msg["chat"]["id"]
                        print(f"✅ Chat ID captured: {CHAT_ID}")
                sender = msg.get("from", {})
                if sender.get("id") in bot_ids:
                    continue
                text = msg.get("text", "")
                if text:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    with chat_lock:
                        chat_history.append({"role": "Spaceman", "content": text, "time": timestamp})
                    print(f"""
[{timestamp}] 🚀 SPACEMAN
{text}
{SEP}""")
        except Exception as e:
            print(f"Poll error: {e}")
            time.sleep(5)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

print(SEP)
print("🌊 WVY World — 4 Agent Groq Loop")
print("Savvy  → kimi-k2-instruct-0905")
print("WVY    → gpt-oss-120b")
print("Trilly → qwen3-32b")
print("Trippy → llama-3.3-70b-versatile")
print(SEP)
print("⏳ Waiting for Chat ID — send any message to the group now...")

poll_thread = threading.Thread(target=poll_telegram, daemon=True)
poll_thread.start()

while CHAT_ID is None:
    time.sleep(1)

print(f"✅ Chat ID ready: {CHAT_ID}")

log("SYSTEM", INITIATION)

savvy_thread  = threading.Thread(target=agent_loop, args=("Savvy",  SAVVY_MODEL,  SAVVY_SYSTEM,  SAVVY_TOKEN),  daemon=True)
wvy_thread    = threading.Thread(target=agent_loop, args=("WVY",    WVY_MODEL,    WVY_SYSTEM,    WVY_TOKEN),    daemon=True)
trilly_thread = threading.Thread(target=agent_loop, args=("Trilly", TRILLY_MODEL, TRILLY_SYSTEM, TRILLY_TOKEN), daemon=True)
trippy_thread = threading.Thread(target=agent_loop, args=("Trippy", TRIPPY_MODEL, TRIPPY_SYSTEM, TRIPPY_TOKEN), daemon=True)

savvy_thread.start()
wvy_thread.start()
trilly_thread.start()
trippy_thread.start()

print("✅ All 4 agents running in Telegram")
print(SEP)

try:
    savvy_thread.join()
    wvy_thread.join()
    trilly_thread.join()
    trippy_thread.join()
except KeyboardInterrupt:
    print("⏹ Stopping all agents...")
    stop_event.set()
    print("✅ Done")
