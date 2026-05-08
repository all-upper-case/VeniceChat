import os
import json
import time
import datetime
import requests
import re
import base64
import uuid
import wave
import io
import traceback
import threading
import queue
from collections import Counter
from flask import Flask, request, jsonify, render_template, Response, stream_with_context

try:
    import numpy as np
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: numpy or faiss-cpu not installed. RAG features will be disabled.")

app = Flask(__name__)
from devtools import devtools_bp
app.register_blueprint(devtools_bp)
@app.route('/dev', methods=['GET', 'POST'])
def phone_dev_tools():
    import shlex
    import subprocess
    import tempfile
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent

    def run_cmd(args, timeout=60):
        proc = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        output = proc.stdout or ""
        if proc.stderr:
            output += "\n--- STDERR ---\n" + proc.stderr
        output += f"\n--- EXIT CODE: {proc.returncode} ---\n"
        return output

    if request.method == 'GET':
        return """<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phone Dev</title>
<style>
body {
    background: #111;
    color: #eee;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    padding: 16px;
}
textarea {
    width: 100%;
    box-sizing: border-box;
    min-height: 220px;
    background: #1e1e1e;
    color: #eee;
    border: 1px solid #444;
    border-radius: 10px;
    padding: 12px;
    font-family: monospace;
    font-size: 14px;
}
button {
    width: 100%;
    padding: 14px;
    margin-top: 10px;
    border: 0;
    border-radius: 10px;
    background: #2563eb;
    color: white;
    font-weight: bold;
    font-size: 16px;
}
button.gray {
    background: #444;
}
button.red {
    background: #b91c1c;
}
pre {
    white-space: pre-wrap;
    word-break: break-word;
    background: #050505;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 12px;
    min-height: 200px;
}
</style>
</head>
<body>
<h2>Phone Dev</h2>

<p>Paste either a fetch command or a full patch block.</p>

<textarea id="input" placeholder='Examples:

python fetch.py main.py --contains "create_pipeline_chat" --context 80

---PATCH---
NAME: example
FILE: main.py
COUNT: 1
---OLD---
old code
---NEW---
new code
---END---'></textarea>

<button onclick="send('run')">Run Fetch Command</button>
<button class="gray" onclick="send('dry')">Dry Run Patch</button>
<button class="red" onclick="send('apply')">Apply Patch</button>

<h3>Output</h3>
<button class="gray" onclick="copyOutput()">Copy Output</button>
<pre id="output"></pre>

<script>
async function send(mode) {
    const text = document.getElementById('input').value;
    const res = await fetch('/dev?mode=' + encodeURIComponent(mode), {
        method: 'POST',
        headers: {'Content-Type': 'text/plain'},
        body: text
    });
    const out = await res.text();
    document.getElementById('output').textContent = out;
}

async function copyOutput() {
    const text = document.getElementById('output').textContent;
    await navigator.clipboard.writeText(text);
    alert('Copied output.');
}
</script>
</body>
</html>"""

    mode = request.args.get('mode', 'run')
    body = request.get_data(as_text=True).strip()

    if not body:
        return "No input provided.\n", 400, {"Content-Type": "text/plain"}

    try:
        if mode in ('dry', 'apply') or body.startswith('---PATCH---'):
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                suffix='.txt',
                prefix='phone_patch_',
                delete=False,
                dir=ROOT
            ) as tmp:
                tmp.write(body)
                patch_file = Path(tmp.name)

            try:
                args = ['python', 'edit.py', patch_file.name]
                if mode != 'apply':
                    args.append('--dry-run')
                output = run_cmd(args, timeout=90)
                return output, 200, {"Content-Type": "text/plain"}
            finally:
                try:
                    patch_file.unlink()
                except OSError:
                    pass

        # Fetch command mode.
        parts = shlex.split(body)

        if len(parts) >= 2 and parts[0] in ('python', 'python3') and parts[1] == 'fetch.py':
            output = run_cmd(parts, timeout=60)
            return output, 200, {"Content-Type": "text/plain"}

        if len(parts) >= 1 and parts[0] == 'fetch.py':
            output = run_cmd(['python'] + parts, timeout=60)
            return output, 200, {"Content-Type": "text/plain"}

        return (
            "Only fetch commands and ---PATCH--- blocks are allowed here.\n\n"
            "Valid examples:\n"
            "python fetch.py main.py --contains \"create_pipeline_chat\" --context 80\n\n"
            "---PATCH---\n"
            "NAME: example\n"
            "FILE: main.py\n"
            "COUNT: 1\n"
            "---OLD---\n"
            "old code\n"
            "---NEW---\n"
            "new code\n"
            "---END---\n"
        ), 400, {"Content-Type": "text/plain"}

    except Exception as e:
        return f"ERROR: {e}\n", 500, {"Content-Type": "text/plain"}

# --- CONFIG ---
# Using Venice.ai as the primary inference provider
VENICE_API_KEY = os.environ.get('VENICE_API_KEY')
FAL_KEY = os.environ.get('FAL_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
VENICE_URL = 'https://api.venice.ai/api/v1/chat/completions'
VENICE_EMBED_URL = 'https://api.venice.ai/api/v1/embeddings'
VENICE_CHARACTERS_URL = 'https://api.venice.ai/api/v1/characters'
VENICE_BASE_URL = 'https://api.venice.ai/api/v1'
FAL_URL = "https://fal.run/fal-ai/z-image/turbo"

FILES = {
    "active_meta": 'data/active_chat_meta.json',
    "venice_settings": 'settings/venice_settings.json',
    "venice_img_settings": 'settings/venice_img_settings.json',
    "refiner_settings": 'settings/refiner_settings.json',
    "img_settings": 'settings/image_settings.json',
    "summarizer_settings": 'settings/summarizer_settings.json',
    "wfm_settings": 'settings/wfm_settings.json',
    "model_history": 'data/model_history.json',
    "pipeline_settings": 'settings/pipeline_settings.json',
    "pipeline_architect_prompt": 'prompts/system_prompt_pipeline_architect.txt',
    "pipeline_scribe_prompt": 'prompts/system_prompt_pipeline_scribe.txt',
    "main_prompt": 'prompts/system_prompt_main.txt',
    "img_prompt_instr": 'prompts/system_prompt_imgprompt.txt',
    "visual_prompt": 'prompts/system_prompt_visual.txt',
    "architect_prompt": 'prompts/system_prompt_architect.txt',
    "user_mimic_prompt": 'prompts/system_prompt_user_mimic.txt',
    "refine_prompt": 'prompts/system_prompt_refine.txt',
    "summary_note_prompt": 'prompts/system_prompt_summary_note.txt',
    "rag_note_prompt": 'prompts/system_prompt_rag_note.txt',
    "lore_extractor_prompt": 'prompts/system_prompt_lore_extractor.txt',
    "summary_consolidator_prompt": 'prompts/system_prompt_summary_consolidator.txt',
    "venice_dupe_prompt": 'prompts/system_prompt_venice_dupe.txt',
    "audit_prompt": 'prompts/system_prompt_audit.txt',
    "evaluator_prompt": 'prompts/system_prompt_evaluator.txt',
    "banned_phrases": 'prompts/banned_phrases.txt',
    "interface_settings": 'settings/interface_settings.json',
    "conversations_dir": 'conversations',
    "uploads_dir": 'static/uploads',
    "lorebook": 'data/lorebook.txt',
    "lorebook_index": 'data/lorebook.index',
    "lorebook_chunks": 'data/lorebook_chunks.json',
    "rag_settings": 'settings/rag_settings.json',
    "tts_settings": 'settings/tts_settings.json',
    "payload_logs_dir": 'payload_logs',
    "tts_logs_dir": 'tts_logs',
    "character_cache": 'data/character_cache.json',
    "audio_cache_dir": 'static/audio_cache',
    "balance": 'data/balance.json',
    "api_ledger": 'data/api_ledger.json'
}

VENICE_SPEECH_URL = 'https://api.venice.ai/api/v1/audio/speech'
MISTRAL_SPEECH_URL = 'https://api.mistral.ai/v1/audio/speech'

# --- UTILS ---
import hashlib
os.makedirs(FILES["audio_cache_dir"], exist_ok=True)

def get_persisted_balance():
    return read_json(FILES["balance"], {"balance": None})

def save_persisted_balance(balance):
    if balance is not None:
        write_json(FILES["balance"], {"balance": balance, "timestamp": datetime.datetime.now().isoformat()})

def get_tts_cache_path(text, model, voice, speed, ref_audio=None):
    """Generates a unique filename based on TTS parameters, prefixed with start of text."""
    # Clean text for prefix: alphanumeric and spaces only, first 20 chars
    clean_prefix = re.sub(r'[^a-zA-Z0-9\s]', '', text[:30]).strip().replace(' ', '_')
    words = clean_prefix.split('_')[:5] # Take first 5 words
    prefix = "_".join(words)

    # Hash the ref_audio if present to distinguish between different clones of the same text
    ref_hash = hashlib.md5(ref_audio.encode('utf-8')).hexdigest()[:8] if ref_audio else "none"

    hash_input = f"{text}|{model}|{voice}|{speed}|{ref_hash}"
    cache_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
    return f"{prefix}_{cache_hash}.wav"

LAST_IO = {"endpoint": None, "request": None, "response": None, "timestamp": None}
LAST_TTS_LOG = {"event": "None", "timestamp": None, "data": "No TTS jobs run yet."}

def update_last_io(endpoint, req, res):
    global LAST_IO
    LAST_IO = {
        "endpoint": endpoint,
        "request": req,
        "response": res,
        "timestamp": datetime.datetime.now().isoformat()
    }

def fetch_real_balance():
    """Fetches the actual current balance from Venice billing API."""
    try:
        r = requests.get(
            f"{VENICE_BASE_URL}/billing/balance",
            headers={"Authorization": f"Bearer {VENICE_API_KEY}"},
            timeout=5
        )
        if r.ok:
            data = r.json()
            bal = data.get("balance")
            if bal is not None:
                return str(bal)
    except:
        pass
    return None

def log_api_call(feature, model, chat_file=None, usage=None, balance_before=None, balance_after=None):
    """Logs a Venice API call to the ledger with cost analysis."""
    try:
        ledger = read_json(FILES["api_ledger"], {"calls": []})
        if "calls" not in ledger:
            ledger = {"calls": []}

        # Calculate costs from usage
        estimated_cost = None
        cache_hit_rate = 0
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0

        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            details = usage.get("prompt_tokens_details", {})
            cached_tokens = details.get("cached_tokens", 0)
            if prompt_tokens > 0:
                cache_hit_rate = round((cached_tokens / prompt_tokens) * 100, 1)

            # Look up model pricing from venice_models.json
            models_data = read_json('data/venice_models.json', {})
            input_rate = 0
            output_rate = 0
            cache_rate = 0
            for group in models_data.values():
                for m in group:
                    if m.get('id') == model and m.get('pricing'):
                        pricing_str = m['pricing']
                        parts = pricing_str.split('(')
                        base_prices = parts[0].replace('$', '').replace(' ', '').split('/')
                        input_rate = float(base_prices[0]) if base_prices[0] else 0
                        output_rate = float(base_prices[1]) if len(base_prices) > 1 and base_prices[1] else 0
                        if len(parts) > 1:
                            cache_match = re.search(r'\$?([0-9.]+)', parts[1])
                            if cache_match:
                                cache_rate = float(cache_match.group(1))
                        else:
                            cache_rate = input_rate * 0.5
                        break
                if input_rate > 0:
                    break

            uncached_input = prompt_tokens - cached_tokens
            estimated_cost = round(
                (uncached_input / 1_000_000) * input_rate +
                (cached_tokens / 1_000_000) * cache_rate +
                (completion_tokens / 1_000_000) * output_rate,
                6
            )

        # Calculate actual cost from balance delta
        actual_cost = None
        if balance_before is not None and balance_after is not None:
            try:
                actual_cost = round(float(balance_before) - float(balance_after), 6)
            except (ValueError, TypeError):
                pass

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "feature": feature,
            "model": model,
            "chat_file": chat_file or os.path.basename(get_active_chat_path()),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "cache_hit_rate": cache_hit_rate,
            "estimated_cost": estimated_cost,
            "actual_cost": actual_cost,
            "balance_before": balance_before,
            "balance_after": balance_after
        }

        ledger["calls"].append(entry)

        # Keep only last 500 entries to prevent unbounded growth
        if len(ledger["calls"]) > 500:
            ledger["calls"] = ledger["calls"][-500:]

        write_json(FILES["api_ledger"], ledger)
    except Exception as e:
        print(f"Ledger logging error: {e}")

def log_tts_event(event_type, data):
    """Logs raw TTS API events to the tts_logs directory and memory."""
    global LAST_TTS_LOG
    LAST_TTS_LOG = {
        "event": event_type,
        "timestamp": datetime.datetime.now().isoformat(),
        "data": data
    }
    os.makedirs(FILES["tts_logs_dir"], exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_fn = f"tts_{ts}_{event_type}.json"
    log_path = os.path.join(FILES["tts_logs_dir"], log_fn)
    write_json(log_path, data)

import copy
FILE_CACHE = {}

def read_json(path, default, use_cache=True):
    if not os.path.exists(path): return default
    if use_cache:
        mtime = os.path.getmtime(path)
        if path in FILE_CACHE and FILE_CACHE[path]['mtime'] == mtime:
            return copy.deepcopy(FILE_CACHE[path]['data'])
    try:
        with open(path, 'r', encoding='utf-8') as f: 
            data = json.load(f)
            if use_cache:
                FILE_CACHE[path] = {'mtime': os.path.getmtime(path), 'data': data}
            return copy.deepcopy(data) if use_cache else data
    except: return default

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4)
    if path in FILE_CACHE:
        del FILE_CACHE[path]

def read_text(path, default=""):
    if not os.path.exists(path): return default
    mtime = os.path.getmtime(path)
    if path in FILE_CACHE and FILE_CACHE[path]['mtime'] == mtime:
        return FILE_CACHE[path]['data']
    try:
        with open(path, 'r', encoding='utf-8') as f: 
            data = f.read().strip()
            FILE_CACHE[path] = {'mtime': mtime, 'data': data}
            return data
    except: return default

def write_text(path, content):
    with open(path, 'w', encoding='utf-8') as f: f.write(content)
    if path in FILE_CACHE:
        del FILE_CACHE[path]

os.makedirs(FILES["conversations_dir"], exist_ok=True)
os.makedirs(FILES["uploads_dir"], exist_ok=True)
os.makedirs(FILES["payload_logs_dir"], exist_ok=True)
if not os.path.exists(FILES["banned_phrases"]): write_text(FILES["banned_phrases"], "")
if not os.path.exists(FILES["lorebook"]): write_text(FILES["lorebook"], "")

def save_base64_image(data_uri):
    """Saves base64 image data to a file and returns the local URL."""
    try:
        if not data_uri.startswith('data:image'):
            return data_uri

        header, encoded = data_uri.split(",", 1)

        # Verify WebP integrity for Venice base64 strings
        # Venice strings usually start with 'UklGR' (RIFF in base64)
        if len(encoded) < 100:
            print("[ERROR] Base64 string is too short to be a valid image.")
            return "/static/error_placeholder.png"

        ext = header.split(";")[0].split("/")[1]
        if ext == 'jpeg': ext = 'jpg'

        filename = f"img_{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(FILES["uploads_dir"], filename)

        with open(filepath, "wb") as f:
            f.write(base64.b64decode(encoded))

        return f"/static/uploads/{filename}"
    except Exception as e:
        print(f"Error saving image: {e}")
        return data_uri

def get_cache_key(path, suffix=""):
    base = os.path.basename(path).replace('.json', '')
    key = base + suffix
    key = "".join([c for c in key if c.isalnum() or c in '-_'])
    return key[:64]

def get_base64_from_local(url):
    """Converts a local URL back to base64 for API transmission."""
    try:
        if not url.startswith('/static/uploads/'):
            return url

        filename = os.path.basename(url)
        filepath = os.path.join(FILES["uploads_dir"], filename)

        ext = filename.split('.')[-1]
        if ext == 'jpg': ext = 'jpeg'

        with open(filepath, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/{ext};base64,{encoded}"
    except Exception as e:
        print(f"Error loading image for API: {e}")
        return url

# Standard Venice Parameters to ensure raw output
VENICE_DEFAULTS = {
    "venice_parameters": {
        "include_venice_system_prompt": False
    }
}

def apply_claude_caching(messages, model):
    if not isinstance(model, str) or not model.startswith('claude-'):
        return messages

    new_messages = []
    last_system_idx = -1
    for i, m in enumerate(messages):
        if m.get('role') == 'system':
            last_system_idx = i
        else:
            break

    for i, m in enumerate(messages):
        if i == last_system_idx:
            content = m.get('content', '')
            if isinstance(content, str):
                new_messages.append({
                    "role": m.get("role", "system"),
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                })
            elif isinstance(content, list) and len(content) > 0 and content[-1].get("type") == "text":
                new_content = [dict(block) for block in content]
                new_content[-1]["cache_control"] = {"type": "ephemeral"}
                new_messages.append({"role": m.get("role", "system"), "content": new_content})
            else:
                new_messages.append(m)
        else:
            new_messages.append(m)
    return new_messages

# --- RAG ENGINE ---
def split_lore_into_chunks(text):
    settings = read_json(FILES["rag_settings"], {"max_chars": 1200, "min_chars": 200})
    max_chars = settings.get("max_chars", 1200)
    min_chars = settings.get("min_chars", 200)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            if len(current) >= min_chars:
                chunks.append(current.strip())
            current = para
        else:
            if current:
                current += "\n\n" + para
            else:
                current = para

    if current and len(current) >= min_chars:
        chunks.append(current.strip())

    return chunks

def get_venice_embeddings(texts):
    headers = {"Authorization": f"Bearer {VENICE_API_KEY}", "Content-Type": "application/json"}
    vectors = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        payload = {
            "model": "text-embedding-ada-002",
            "input": batch
        }
        r = requests.post(VENICE_EMBED_URL, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        vectors.extend([item['embedding'] for item in data['data']])
    return np.array(vectors, dtype="float32")

def rebuild_lore_index():
    if not FAISS_AVAILABLE:
        return False, "FAISS/Numpy not installed. Please run: pip install numpy faiss-cpu"

    text = read_text(FILES["lorebook"])
    if not text:
        return False, "Lorebook is empty."

    chunks = split_lore_into_chunks(text)
    if not chunks:
        return False, "No valid chunks produced."

    try:
        embeddings = get_venice_embeddings(chunks)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, FILES["lorebook_index"])
        write_json(FILES["lorebook_chunks"], chunks)
        return True, f"Index rebuilt with {len(chunks)} chunks."
    except Exception as e:
        return False, str(e)

def retrieve_lore(query, k=3):
    if not FAISS_AVAILABLE:
        return []

    if not os.path.exists(FILES["lorebook_index"]) or not os.path.exists(FILES["lorebook_chunks"]):
        return []

    try:
        index = faiss.read_index(FILES["lorebook_index"])
        chunks = read_json(FILES["lorebook_chunks"], [])

        q_vec = get_venice_embeddings([query])
        distances, indices = index.search(q_vec, k)

        retrieved = []
        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                retrieved.append(chunks[idx])
        return retrieved
    except:
        return []


# --- CHAT DATA HELPERS ---
def load_chat_data(path):
    raw = read_json(path, [])
    if isinstance(raw, list): raw = {"messages": raw, "summaries": [], "visual_memory": "", "self_memory": "", "memory_logs": [], "character_slug": None}
    if "summaries" not in raw: raw["summaries"] = []
    if "visual_memory" not in raw: raw["visual_memory"] = ""
    if "self_memory" not in raw: raw["self_memory"] = ""
    if "memory_logs" not in raw: raw["memory_logs"] = []
    if "character_slug" not in raw: raw["character_slug"] = None
    if "chat_type" not in raw: raw["chat_type"] = "standard"
    if "pipeline_phase" not in raw: raw["pipeline_phase"] = "architect"
    if "blueprint" not in raw: raw["blueprint"] = ""
    if "pipeline_settings" not in raw: raw["pipeline_settings"] = {}
    
    # Initialize dual-track history for pipelines
    if raw.get("chat_type") == "pipeline":
        if "architect_messages" not in raw:
            raw["architect_messages"] = raw.get("messages", [])
        if "scribe_messages" not in raw:
            # Scribe starts fresh with just the system prompt (which build_context handles)
            raw["scribe_messages"] = []
    else:
        # For standard chats, ensure 'messages' is used
        if "messages" not in raw:
            raw["messages"] = []

    # CLEANUP: Strip accumulated directive duplicates from system prompt.
    # A prior bug caused build_context() to permanently append [LIVE INLINE SUMMARY
    # DIRECTIVE] and [AI SELF-MEMORY TOOL] instructions to the saved system prompt
    # on every turn. This strips all accumulated copies so they can be cleanly
    # re-injected by build_context() each time.
    if raw["messages"] and raw["messages"][0].get("role") == "system":
        content = raw["messages"][0].get("content", "")
        dirty = False
        for marker in ["[LIVE INLINE SUMMARY DIRECTIVE]", "[AI SELF-MEMORY TOOL]"]:
            idx = content.find(marker)
            if idx != -1:
                content = content[:idx].rstrip()
                dirty = True
        if dirty:
            raw["messages"][0]["content"] = content
            save_chat_data(path, raw)

    # MIGRATION: Convert existing inline live_summaries to standard summary objects
    modified = False
    sums = raw["summaries"]
    for i, m in enumerate(raw["messages"]):
        if m.get("role") == "assistant" and "live_summary" in m:
            covered = any(s["start_index"] <= i <= s["end_index"] for s in sums)
            if not covered:
                start_idx = i - 1
                while start_idx > 0 and raw["messages"][start_idx].get("role") != "user":
                    start_idx -= 1
                if start_idx < 1: start_idx = i - 1 if i > 0 else 1
                sums.append({
                    "start_index": start_idx,
                    "end_index": i,
                    "content": m["live_summary"],
                    "type": "live"
                })
                modified = True
    
    if modified:
        sums.sort(key=lambda x: x["start_index"])
        save_chat_data(path, raw)

    return raw

def save_chat_data(path, data):
    write_json(path, data)
    os.utime(path, None)

def get_active_chat_path():
    meta = read_json(FILES["active_meta"], {"filename": None})
    fn = meta.get("filename")

    # Check if we have a pending character from a selection event
    pending_character = meta.get("pending_character")

    if not fn or not os.path.exists(os.path.join(FILES["conversations_dir"], fn)):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"New_Chat_{ts}.json"

        # Preserve character slug if it was just selected
        new_meta = {"filename": fn}
        if pending_character:
            new_meta["pending_character"] = pending_character
        write_json(FILES["active_meta"], new_meta)

        write_json(os.path.join(FILES["conversations_dir"], fn), {
            "messages": [{"role": "system", "content": read_text(FILES["main_prompt"])}],
            "summaries": [],
            "visual_memory": "",
            "self_memory": "",
            "memory_logs": [],
            "character_slug": pending_character,
            "chat_type": "standard",
            "pipeline_phase": "architect",
            "blueprint": "",
            "pipeline_settings": {}
        })
    return os.path.join(FILES["conversations_dir"], fn)

def clean_for_api(messages):
    cleaned = []
    for m in messages:
        c = m.get('content', '')
        if isinstance(c, str) and c.startswith('__IMG_JSON__'): continue
        cleaned.append({"role": m.get("role"), "content": c})
    return cleaned

def get_text_content(m):
    c = m.get('content', '')
    if isinstance(c, str):
        if c.startswith('__IMG_JSON__'): return ""
        return c
    if isinstance(c, list):
        return "\n".join([p['text'] for p in c if p.get('type') == 'text' and 'text' in p])
    return ""

def is_vision_model(model_id):
    models_data = read_json('data/venice_models.json', {})
    for group in models_data.values():
        for m in group:
            if m.get('id') == model_id and m.get('vision'):
                return True
    return False

def is_reasoning_model(model_id):
    models_data = read_json('data/venice_models.json', {})
    for group in models_data.values():
        for m in group:
            if m.get('id') == model_id:
                if 'REASONING' in m.get('tags', []) or 'Reasoning' in m.get('traits', ''):
                    return True
                desc = m.get('description', '').lower()
                if 'reasoning effort' in desc or 'reasoning' in desc or 'thinking' in desc:
                    return True
                if any(x in model_id.lower() for x in ['deepseek-v', 'kimi-k2', 'glm-5', 'trinity-large']):
                    return True
                return False
    return False

# --- SUMMARIZATION ENGINE ---
def process_summaries(chat_data):
    s_set = read_json(FILES["summarizer_settings"], {"enabled": False})
    if not s_set.get("enabled", False) or s_set.get("live_summary_enabled", False): return

    msgs = chat_data["messages"]
    sums = chat_data["summaries"]

    threshold = int(s_set.get("trigger_threshold_turns", 10))
    keep = int(s_set.get("recent_turns_to_keep", 4))
    batch_size = int(s_set.get("batch_size", 4))

    while True:
        last_sum_idx = sums[-1]["end_index"] if sums else 0
        valid_indices = [i for i, m in enumerate(msgs) if i > 0 and not (isinstance(m.get('content', ''), str) and m.get('content', '').startswith('__IMG_JSON__'))]
        available_valid = [i for i in valid_indices if i > last_sum_idx]

        if len(available_valid) <= keep:
            break

        summarizable_valid = available_valid[:-keep]

        if len(available_valid) >= threshold and len(summarizable_valid) >= batch_size:
            batch_indices = summarizable_valid[:batch_size]
            start_idx = batch_indices[0]
            actual_end = batch_indices[-1]

            text_msgs_before = len([i for i in valid_indices if i < start_idx])
            h_start = text_msgs_before + 1
            h_end = h_start + len(batch_indices) - 1

            yield f"Summarizing messages #{h_start} through #{h_end}..."

            batch = msgs[start_idx : actual_end + 1]
            batch_cleaned = clean_for_api(batch)
            text_to_sum = "\n".join([f"{m['role'].upper()}: {get_text_content(m)[:1000]}" for m in batch_cleaned])

            context_summaries = sums[-3:]
            context_str = "\n\n".join([f"PREVIOUS SUMMARY BLOCK:\n{s['content']}" for s in context_summaries])

            user_content = ""
            if context_str:
                user_content += f"--- CONTEXT: PREVIOUS SUMMARIES (Do not re-summarize these) ---\n{context_str}\n\n"
            user_content += f"--- NEW MESSAGES TO SUMMARIZE (Summarize the following only) ---\n{text_to_sum}"

            try:
                h = {"Authorization": f"Bearer {VENICE_API_KEY}"}
                sum_model = s_set.get("model", "qwen3-4b")
                p = {
                    "model": sum_model,
                    "temperature": 0.3,
                    "messages": apply_claude_caching([
                        {"role": "system", "content": s_set.get("system_prompt", "Summarize.")},
                        {"role": "user", "content": user_content}
                    ], sum_model),
                    "prompt_cache_key": "summarizer_job",
                    "venice_parameters": {
                        "include_venice_system_prompt": False,
                        "strip_thinking_response": True
                    }
                }
                r_sum = requests.post(VENICE_URL, headers=h, json=p)
                r_sum.raise_for_status()
                resp = r_sum.json()

                # Log summarizer call to ledger in background
                def log_sum_bg(u, b_before, m_id):
                    log_api_call(
                        feature="Batch Summarizer",
                        model=m_id,
                        usage=u,
                        balance_before=b_before,
                        balance_after=fetch_real_balance()
                    )
                threading.Thread(target=log_sum_bg, args=(resp.get('usage'), r_sum.headers.get('x-venice-balance-usd'), sum_model)).start()

                if 'choices' not in resp or not resp['choices']:
                    break

                summary_text = resp['choices'][0]['message']['content']
                usage = resp.get('usage', {})

                sums.append({
                    "start_index": start_idx,
                    "end_index": actual_end,
                    "content": summary_text,
                    "usage": usage
                })
            except Exception as e:
                print(f"Summarizer failed: {e}")
                break
        else:
            break


def build_context(chat_data, user_query=None, current_model=None):
    s_set = read_json(FILES["summarizer_settings"], {"enabled": False})
    rag_set = read_json(FILES["rag_settings"], {"enabled": False, "k": 3})
    vision_capable = is_vision_model(current_model) if current_model else False

    # Story Pipeline Track Selection: Architect and Scribe now have completely separate histories
    if chat_data.get("chat_type") == "pipeline":
        phase = chat_data.get("pipeline_phase", "architect")
        if phase == "scribe":
            msgs = chat_data.get("scribe_messages", [])
            # If scribe history is empty, initialize with system prompt
            if not msgs:
                msgs = [{"role": "system", "content": read_text(FILES["pipeline_scribe_prompt"])}]
                chat_data["scribe_messages"] = msgs
                # Don't save here to avoid mutation during context build, but ensure msgs is local
        else:
            msgs = chat_data.get("architect_messages", [])
            if not msgs:
                msgs = [{"role": "system", "content": read_text(FILES["pipeline_architect_prompt"])}]
                chat_data["architect_messages"] = msgs
    else:
        msgs = chat_data.get("messages", [])

    if not msgs: return []

    if not user_query:
        for m in reversed(msgs):
            if m.get("role") == "user" and not (isinstance(m.get('content', ''), str) and m.get('content', '').startswith('__IMG_JSON__')):
                user_query = get_text_content(m)
                break

    # CRITICAL: Deep copy the system prompt to prevent mutating the original
    # chat_data dict.
    context = [{"role": msgs[0]["role"], "content": msgs[0].get("content", "")}]

    if chat_data.get("chat_type") == "pipeline":
        phase = chat_data.get("pipeline_phase", "architect")
        blueprint = chat_data.get("blueprint", "")
        settings = chat_data.get("pipeline_settings", {})
        
        if phase == "architect":
            architect_sys = read_text(FILES["pipeline_architect_prompt"])
            if settings:
                settings_block = "\n\n[PROJECT SPECIFIC CONSTRAINTS]\n"
                for k, v in settings.items():
                    if v: settings_block += f"- {k.upper()}: {v}\n"
                architect_sys += settings_block
            context[0]["content"] = architect_sys
        else:
            context[0]["content"] = read_text(FILES["pipeline_scribe_prompt"])
            
        blueprint_text = blueprint if blueprint else "(Empty - Please start building the Story Blueprint!)"
        context.append({"role": "system", "content": f"--- CURRENT STORY BLUEPRINT ---\n{blueprint_text}"})
    
    # Static Venice Dupe to improve caching (replaces official dynamic Venice prompt)
    context.append({
        "role": "system",
        "content": read_text(FILES["venice_dupe_prompt"])
    })

    banned = read_text(FILES["banned_phrases"])
    if banned:
        context.append({
            "role": "system",
            "content": f"[REPETITION CONTROL UNIT]\nSTRICT BANNED LIST:\n{banned}\n\nTask: Generate prose while ensuring zero overlap with the list above."
        })

    vis_mem = chat_data.get("visual_memory", "")
    if vis_mem:
        context.append({"role": "system", "content": f"--- PERMANENT CHARACTER VISUALS & LORE ---\n{vis_mem}"})

    v_set = read_json(FILES["venice_settings"], {})
    if v_set.get("auto_memory_enabled"):
        self_mem = chat_data.get("self_memory", "")
        if self_mem:
            context.append({
                "role": "system", 
                "content": f"--- AI INTERNAL NOTEPAD (Your Persistent Self-Memory) ---\n{self_mem}"
            })
        
        # Inject surgical tool instructions into the system prompt
        tool_instr = "\n\n[AI SELF-MEMORY TOOL]\nYou have a persistent 'Internal Notepad' at a fixed position near the start of your context. Note that while its position is static, its contents are dynamic and cumulative; the version you see always reflects the most up-to-date state resulting from all cumulative tool calls and manual edits, regardless of the chat chronology. Use it to track long-term facts, motivations, or user feedback that might otherwise be lost during summarization. To update it, output exactly this format at the VERY END of your response:\n\n[MEMORY_ACTION]\n[ADD]\nText to append to the end of the notepad.\n[/ADD]\n[REPLACE]\nExact text to find and remove.\n[WITH]\nNew text to replace it with.\n[/REPLACE]\n[/MEMORY_ACTION]"
        if context[0]["role"] == "system":
            context[0]["content"] += tool_instr

    # --- STABLE HISTORY BLOCK (Summaries & Raw History) ---
    v_set = read_json(FILES["venice_settings"], {})
    vision_detail = "high" if v_set.get("vision_high_res", True) else "low"

    if s_set.get("live_summary_enabled") and chat_data.get("chat_type") != "pipeline":
        live_sum_instr = "\n\n[LIVE INLINE SUMMARY DIRECTIVE]\nAt the VERY END of your response, after you have finished writing your full narrative prose, you must generate a highly concrete and specific summary of the entire turn (the user's action and your response). This summary must capture essential, exact details (names, specific physical actions, important plot developments, and exact emotional shifts) rather than generic filler. Write this summary in the same narrative tense, person, and stylistic voice as the main story. You MUST place this summary at the end of your message, strictly enclosed within [SUM] and [/SUM] tags. Do not place the summary at the beginning."
        if context[0]["role"] == "system":
            context[0]["content"] += live_sum_instr

    if s_set.get("enabled", False) or s_set.get("live_summary_enabled", False):
        sums = chat_data.get("summaries", [])
        active_sums = [s for s in sums if not s.get("disabled", False)]
        active_sums.sort(key=lambda x: x["start_index"])

        # Calculate threshold index for swapping
        keep = int(s_set.get("recent_turns_to_keep", 12))
        batch_size = int(s_set.get("batch_size", 4))
        valid_indices = [i for i, m in enumerate(msgs) if i > 0 and not (isinstance(m.get('content', ''), str) and m.get('content', '').startswith('__IMG_JSON__'))]
        
        V = len(valid_indices)
        summarizable_count = V - keep
        threshold_idx = 0
        if summarizable_count > 0:
            summarized_count = (summarizable_count // batch_size) * batch_size
            if summarized_count > 0:
                threshold_idx = valid_indices[summarized_count] if summarized_count < V else len(msgs)

        # Only use summaries that end before the threshold, or are consolidated archives
        applicable_sums = [s for s in active_sums if s["end_index"] < threshold_idx or s.get("is_consolidated")]

        if applicable_sums:
            context.append({"role": "system", "content": read_text(FILES["summary_note_prompt"])})

        msg_idx = 1
        sum_idx = 0

        while msg_idx < len(msgs):
            if sum_idx < len(applicable_sums) and msg_idx == applicable_sums[sum_idx]["start_index"]:
                s = applicable_sums[sum_idx]
                prefix = "CONSOLIDATED ARCHIVE (Distant Past Context):" if s.get("is_consolidated") else "RECENT SUMMARY (Immediate Past Context):"
                clean_sum = re.sub(r'<think>.*?</think>', '', s['content'], flags=re.DOTALL).strip()
                context.append({"role": "system", "content": f"--- {prefix} ---\n{clean_sum}"})
                msg_idx = s["end_index"] + 1
                sum_idx += 1
            elif sum_idx < len(applicable_sums) and msg_idx > applicable_sums[sum_idx]["start_index"]:
                sum_idx += 1
            else:
                c = msgs[msg_idx].get('content', '')
                if not (isinstance(c, str) and c.startswith('__IMG_JSON__')):
                    if isinstance(c, list):
                        if vision_capable:
                            # Convert local URL back to base64 for API
                            new_content = []
                            for part in c:
                                if part.get('type') == 'image_url':
                                    img_url = part['image_url']['url']
                                    if img_url.startswith('/static/uploads/'):
                                        new_content.append({
                                            "type": "image_url",
                                            "image_url": {
                                                "url": get_base64_from_local(img_url),
                                                "detail": part['image_url'].get('detail', 'auto')
                                            }
                                        })
                                    else:
                                        new_content.append(part)
                                else:
                                    new_content.append(part)
                            context.append({"role": msgs[msg_idx]["role"], "content": new_content})
                        else:
                            context.append({"role": msgs[msg_idx]["role"], "content": get_text_content(msgs[msg_idx])})
                    else:
                        context.append({"role": msgs[msg_idx]["role"], "content": c})
                msg_idx += 1
    else:
        v_set = read_json(FILES["venice_settings"], {})
        vision_detail = "high" if v_set.get("vision_high_res", True) else "low"
        for m in clean_for_api(msgs[1:]):
            c = m.get('content', '')
            if isinstance(c, list):
                if vision_capable:
                    new_content = []
                    for part in c:
                        if part.get('type') == 'image_url':
                            img_url = part['image_url']['url']
                            if img_url.startswith('/static/uploads/'):
                                new_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": get_base64_from_local(img_url),
                                        "detail": part['image_url'].get('detail', 'auto')
                                    }
                                })
                            else:
                                new_content.append(part)
                        else:
                            new_content.append(part)
                    context.append({"role": m["role"], "content": new_content})
                else:
                    context.append({"role": m["role"], "content": get_text_content(m)})
            else:
                context.append({"role": m["role"], "content": c})

    # --- VOLATILE CONTEXT BLOCK (RAG) ---
    # Injected at the end of context so it doesn't break cache for the history above it
    if rag_set.get("enabled", False) and user_query:
        k = int(rag_set.get("k", 3))
        lore_chunks = retrieve_lore(user_query, k=k)
        if lore_chunks:
            lore_text = "\n\n---\n\n".join(lore_chunks)
            context.append({
                "role": "system", 
                "content": f"{read_text(FILES['rag_note_prompt'])}\n{lore_text}"
            })

    return context

# --- SECURITY KEY: Change this to something private ---
SECRET_KEY = "JosieSecret123"

@app.route('/update-files', methods=['POST'])
def update_files():
    # 1. Security Check
    if request.headers.get('X-Secret-Key') != SECRET_KEY:
        return "Unauthorized", 401
    
    # 2. Get the raw text from the Shortcut
    content = request.get_data(as_text=True)
    if not content:
        return jsonify({"status": "error", "message": "No data received"}), 400

    updated_files = []
    # 3. The 'Scissors' Logic: Split text by the filename tags
    # regex matches the file header format (---F-I-L-E:path/to/file---)
    parts = re.split(r'---' + r'FILE:(.*?)---', content)
    
    # parts[0] is everything before the first tag (usually empty)
    # The rest alternates: [filename, code, filename, code...]
    for i in range(1, len(parts), 2):
        filepath = parts[i].strip()
        code = parts[i+1].strip()
        
        if not filepath:
            continue

        # Handle subdirectories (e.g., static/script.js)
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            
        # 4. Overwrite the file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            updated_files.append(filepath)
        except Exception as e:
            print(f"Error updating {filepath}: {e}")

    return jsonify({
        "status": "success",
        "updated": updated_files,
        "message": f"Successfully updated {len(updated_files)} files."
    })

@app.route('/fetch-files', methods=['POST'])
def fetch_files():
    if request.headers.get('X-Secret-Key') != SECRET_KEY:
        return "Unauthorized", 401
        
    data = request.json
    if not data or 'paths' not in data:
        return jsonify({"status": "error", "message": "No paths provided"}), 400
        
    paths = data['paths']
    result_files = {}
    
    def add_file(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                result_files[filepath] = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    for p in paths:
        if os.path.isfile(p):
            add_file(p)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                for file in files:
                    full_path = os.path.join(root, file)
                    full_path = full_path.replace('\\', '/')
                    add_file(full_path)
                    
    return jsonify({
        "status": "success",
        "files": result_files
    })

# --- ROUTES ---

@app.route('/venice/discovery/<path:endpoint>')
def venice_discovery_proxy(endpoint):
    """Proxies GET requests to Venice discovery and billing endpoints."""
    try:
        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        # Map frontend pseudo-paths to actual API endpoints
        endpoint_map = {
            "models": f"{VENICE_BASE_URL}/models",
            "traits": f"{VENICE_BASE_URL}/models/traits",
            "mapping": f"{VENICE_BASE_URL}/models/compatibility_mapping",
            "balance": f"{VENICE_BASE_URL}/billing/balance",
            "rate_limits": f"{VENICE_BASE_URL}/api_keys/rate_limits"
        }
        
        target_url = endpoint_map.get(endpoint)
        if not target_url:
            return jsonify({"error": f"Invalid discovery endpoint: {endpoint}"}), 400
            
        r = requests.get(target_url, headers=headers)
        update_last_io(f"DISCOVERY: {endpoint}", None, f"Status {r.status_code}")
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/venice/import_model', methods=['POST'])
def venice_import_model():
    """Adds a discovered model to the local venice_models.json file."""
    try:
        new_model = request.json
        category = request.args.get('category', 'Imported Models')
        
        models_data = read_json('data/venice_models.json', {})
        if category not in models_data:
            models_data[category] = []
            
        # Check for duplicates
        if any(m.get('id') == new_model.get('id') for m in models_data[category]):
            return jsonify({"success": False, "message": "Model already exists in this category."})
            
        models_data[category].append(new_model)
        write_json('data/venice_models.json', models_data)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index(): return render_template('index.html')

@app.route('/architect_chat', methods=['POST'])
def architect_chat():
    hist = request.json.get('history', [])
    context_initial = [{"role": "system", "content": read_text(FILES["architect_prompt"])}] + hist

    def generate():
        # 1. Start Main Generation
        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        payload = {
            "model": "venice-uncensored",
            "temperature": 0.7,
            "messages": apply_claude_caching(context_initial, "venice-uncensored"), 
            "stream": True,
            "prompt_cache_key": "architect_session",
            **VENICE_DEFAULTS
        }
        try:
            with requests.post(VENICE_URL, headers=headers, json=payload, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if "[DONE]" in decoded: break
                        try:
                            chunk = json.loads(decoded[6:])
                            if len(chunk['choices'])>0:
                                c = chunk['choices'][0]['delta'].get('content', '')
                                yield f"data: {json.dumps({'content': c})}\n\n"
                        except: pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/create_scenario_chat', methods=['POST'])
def create_scenario_chat():
    prompt = request.json.get('prompt')
    if not prompt: return jsonify({"error": "No prompt provided"}), 400

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"Scenario_{ts}.json"
    path = os.path.join(FILES["conversations_dir"], fn)

    messages = [
        {"role": "system", "content": read_text(FILES["main_prompt"])},
        {"role": "user", "content": prompt}
    ]

    try:
        v_set = read_json(FILES["venice_settings"], {})
        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        model_id = v_set.get("model", "venice-uncensored")
        payload = {
            "model": model_id,
            "temperature": float(v_set.get("temperature", 0.7)),
            "messages": apply_claude_caching(messages, model_id),
            "prompt_cache_key": fn.replace('.json', ''),
            **VENICE_DEFAULTS
        }
        resp = requests.post(VENICE_URL, headers=headers, json=payload).json()
        opening_content = resp['choices'][0]['message']['content']
        messages.append({"role": "assistant", "content": opening_content})
    except Exception as e:
        messages.append({"role": "assistant", "content": "(Error generating opening scene. You may need to regenerate.)"})

    save_chat_data(path, {
        "messages": messages,
        "summaries": [],
        "visual_memory": ""
    })

    write_json(FILES["active_meta"], {"filename": fn})

    return jsonify({"success": True, "filename": fn})


def fuzzy_replace(text, find, replace):
    """
    Performs a robust replacement by ignoring differences in whitespace, 
    newlines, and indentation.
    """
    if not find or not find.strip():
        return text, False

    # Escape special regex characters in the 'find' block, 
    # but replace any whitespace sequence with a flexible whitespace regex
    # Match any whitespace (including newlines) one or more times
    pattern_str = re.escape(find.strip())
    pattern_str = re.sub(r'\\ ', r'\\s+', pattern_str) # handle escaped spaces
    pattern_str = re.sub(r'(\\\n|\\r|\\t)+', r'\\s+', pattern_str) # handle escaped newlines/tabs
    
    # Also handle multiple spaces in the source regex
    pattern_str = re.sub(r'(\\s\+)+', r'\\s+', pattern_str)

    try:
        # Use re.DOTALL to let '.' match newlines
        pattern = re.compile(pattern_str, re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if match:
            new_text = text[:match.start()] + replace + text[match.end():]
            return new_text, True
    except re.error:
        pass
    
    # Fallback to literal search if regex fails
    if find.strip() in text:
        return text.replace(find.strip(), replace), True
        
    return text, False

@app.route('/create_pipeline_chat', methods=['POST'])
def create_pipeline_chat():
    settings = request.json.get('settings', {})
    concept = request.json.get('concept', '')
    if not concept: return jsonify({"error": "No concept provided"}), 400

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"Pipeline_{ts}.json"
    path = os.path.join(FILES["conversations_dir"], fn)

    architect_msgs = [
        {"role": "system", "content": read_text(FILES["pipeline_architect_prompt"])},
        {"role": "user", "content": f'''I want to build a story blueprint.

### CONCEPT:
{concept}

### SETTINGS:
{json.dumps(settings, indent=2)}'''}
    ]

    save_chat_data(path, {
        "architect_messages": architect_msgs,
        "scribe_messages": [],
        "messages": [], # Fallback for other systems
        "summaries": [],
        "visual_memory": "",
        "chat_type": "pipeline",
        "pipeline_phase": "architect",
        "pipeline_settings": settings,
        "blueprint": ""
    })

    write_json(FILES["active_meta"], {"filename": fn})

    return jsonify({"success": True, "filename": fn})

@app.route('/continue_generation', methods=['POST'])
def continue_generation():
    # Deprecated for Pipeline
    pass

@app.route('/continue_pipeline', methods=['POST'])
def continue_pipeline():
    path = get_active_chat_path()
    chat_data = load_chat_data(path)
    
    # We want to continue the very last assistant message
    if not chat_data["messages"] or chat_data["messages"][-1]["role"] != "assistant":
        return jsonify({"error": "Last message is not from assistant"}), 400
        
    idx = len(chat_data["messages"]) - 1
    model_to_use = chat_data["messages"][-1].get("model")
    if not model_to_use:
        custom_model = request.json.get('custom_model')
        v_set = read_json(FILES["venice_settings"], {})
        model_to_use = custom_model if custom_model else v_set.get("model", "venice-uncensored")

    def generate():
        nonlocal chat_data
        temp_data = {"messages": chat_data["messages"], "summaries": chat_data.get("summaries", []), "visual_memory": chat_data.get("visual_memory", "")}
        context = build_context(temp_data, user_query="", current_model=model_to_use)
        
        # Init full with the existing content!
        full = chat_data["messages"][idx].get("content", "")
        reasoning = chat_data["messages"][idx].get("reasoning", "")
        
        payload = {
            "model": model_to_use,
            "messages": context,
            "stream": True,
            "temperature": 0.8
        }
        
        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        try:
            with requests.post(VENICE_URL, headers=headers, json=payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                full_response_json = []
                usage = None
                chunk_buffer = ""
                
                # We skip live summarization and memory for now to keep the code simpler since this is just pipeline
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if "[DONE]" in decoded: break
                        try:
                            chunk_raw = decoded[6:]
                            chunk = json.loads(chunk_raw)
                            full_response_json.append(chunk)
                            if "usage" in chunk: usage = chunk["usage"]
                            if len(chunk['choices'])>0:
                                delta = chunk['choices'][0]['delta']
                                
                                if 'reasoning_content' in delta and delta['reasoning_content']:
                                    r_part = delta['reasoning_content']
                                    reasoning += r_part
                                    chat_data[track_key][idx]["reasoning"] = reasoning
                                    yield f"data: {json.dumps({'reasoning': r_part})}\n\n"

                                if 'content' in delta and delta['content']:
                                    c = delta['content']
                                    full += c
                                    yield f"data: {json.dumps({'content': c})}\n\n"
                        except:
                            pass
                            
                chat_data["messages"][idx]["content"] = full
                save_chat_data(path, chat_data)

            # Post-stream blueprint extraction
            if chat_data.get("chat_type") == "pipeline" and chat_data.get("pipeline_phase") == "architect":
                current_bp = chat_data.get("blueprint", "")
                bp_updated = False
                
                # 1. Rewrite Blueprint
                rewrite_pattern = re.compile(r'\[REWRITE_BLUEPRINT\](.*?)\[/REWRITE_BLUEPRINT\]', re.DOTALL)
                matches = list(rewrite_pattern.finditer(full))
                if matches:
                    current_bp = matches[-1].group(1).strip()
                    bp_updated = True
                    full = rewrite_pattern.sub('', full)
                
                # 2. Add to Blueprint
                add_pattern = re.compile(r'\[ADD_TO_BLUEPRINT\](.*?)\[/ADD_TO_BLUEPRINT\]', re.DOTALL)
                for match in add_pattern.finditer(full):
                    addition = match.group(1).strip()
                    if addition:
                        current_bp = current_bp + "\n\n" + addition if current_bp else addition
                        bp_updated = True
                full = add_pattern.sub('', full)
                
                # 3. Edit Blueprint (Find/Replace)
                edit_pattern = re.compile(r'\[EDIT_BLUEPRINT\]\s*<find>(.*?)</find>\s*<replace>(.*?)</replace>\s*\[/EDIT_BLUEPRINT\]', re.DOTALL)
                edit_errors = []
                for match in edit_pattern.finditer(full):
                    old_text = match.group(1).strip()
                    new_text = match.group(2).strip()
                    if old_text and old_text in current_bp:
                        current_bp = current_bp.replace(old_text, new_text)
                        bp_updated = True
                    elif old_text:
                        edit_errors.append(f"Failed to find exact text to replace: '{old_text[:50]}...'")
                full = edit_pattern.sub('', full)

                if bp_updated:
                    chat_data["blueprint"] = current_bp.strip()
                    save_chat_data(path, chat_data)
                    yield f"data: {json.dumps({'blueprint_update': chat_data['blueprint']})}\n\n"
                
                # Collect the raw tool call text for display BEFORE stripping
                raw_tool_calls_lines = []
                for m in list(rewrite_pattern.finditer(full)) if 'rewrite_pattern' in dir() else []:
                    raw_tool_calls_lines.append(f"[REWRITE_BLUEPRINT]\n{m.group(1).strip()}\n[/REWRITE_BLUEPRINT]")
                
                # Re-scan original full for all tool call types for display
                tool_calls_display = []
                for m in re.finditer(r'\[REWRITE_BLUEPRINT\](.*?)\[/REWRITE_BLUEPRINT\]', full, re.DOTALL):
                    tool_calls_display.append(f"ðŸ”„ REWRITE_BLUEPRINT:\n{m.group(1).strip()}")
                for m in re.finditer(r'\[ADD_TO_BLUEPRINT\](.*?)\[/ADD_TO_BLUEPRINT\]', full, re.DOTALL):
                    tool_calls_display.append(f"âž• ADD_TO_BLUEPRINT:\n{m.group(1).strip()}")
                for m in re.finditer(r'\[EDIT_BLUEPRINT\]\s*<find>(.*?)</find>\s*<replace>(.*?)</replace>\s*\[/EDIT_BLUEPRINT\]', full, re.DOTALL):
                    tool_calls_display.append(f"âœï¸ EDIT_BLUEPRINT:\n  FIND: {m.group(1).strip()}\n  REPLACE: {m.group(2).strip()}")
                
                # Clean up tags from chat history
                full = full.strip()
                chat_data["messages"][idx]["content"] = full
                if tool_calls_display:
                    tool_calls_text = "\n\n---\n\n".join(tool_calls_display)
                    chat_data["messages"][idx]["blueprint_tool_calls"] = tool_calls_text
                    yield f"data: {json.dumps({'tool_calls': tool_calls_text})}\n\n"
                save_chat_data(path, chat_data)
                yield f"data: {json.dumps({'content_overwrite': full})}\n\n"
                
                if edit_errors:
                    err_msg = "\n".join(edit_errors)
                    chat_data["messages"].append({"role": "system", "content": f"Automated Notice: Your targeted blueprint edit failed.\n{err_msg}\nPlease ensure the <find> block exactly matches the text currently in the document, including punctuation and whitespace. You may use [REWRITE_BLUEPRINT] if targeted editing fails."})
                    save_chat_data(path, chat_data)

        except Exception as e:
            chat_data["messages"][idx]["content"] = full + f"\n\n*(Stream error: {str(e)})*"
            save_chat_data(path, chat_data)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


def process_blueprint_tools(chat_data, full_text):
    """
    Parses and executes blueprint tools from generated text.
    Returns (cleaned_text, tool_calls_display_text, updated_blueprint)
    """
    current_bp = chat_data.get("blueprint", "")
    bp_updated = False
    tool_calls_display = []
    
    # 1. Rewrite Blueprint
    rewrite_pattern = re.compile(r'\[REWRITE_BLUEPRINT\](.*?)\[/REWRITE_BLUEPRINT\]', re.DOTALL)
    for match in rewrite_pattern.finditer(full_text):
        content = match.group(1).strip()
        if content:
            current_bp = content
            bp_updated = True
            tool_calls_display.append(f"🔄 REWRITE_BLUEPRINT:\n{content}")
    
    # 2. Add to Blueprint
    add_pattern = re.compile(r'\[ADD_TO_BLUEPRINT\](.*?)\[/ADD_TO_BLUEPRINT\]', re.DOTALL)
    for match in add_pattern.finditer(full_text):
        addition = match.group(1).strip()
        if addition:
            current_bp = current_bp + "\n\n" + addition if current_bp else addition
            bp_updated = True
            tool_calls_display.append(f"➕ ADD_TO_BLUEPRINT:\n{addition}")
            
    # 3. Edit Blueprint (Fuzzy/Fuzzy-ish)
    edit_pattern = re.compile(r'\[EDIT_BLUEPRINT\]\s*<find>(.*?)</find>\s*<replace>(.*?)</replace>\s*\[/EDIT_BLUEPRINT\]', re.DOTALL)
    edit_errors = []
    for match in edit_pattern.finditer(full_text):
        old_text = match.group(1).strip()
        new_text = match.group(2).strip()
        
        # Try exact first
        if old_text and old_text in current_bp:
            current_bp = current_bp.replace(old_text, new_text)
            bp_updated = True
            tool_calls_display.append(f"✍️ EDIT_BLUEPRINT (Exact Match):\n  FIND: {old_text[:100]}...\n  REPLACE: {new_text[:100]}...")
        elif old_text:
            # Try fuzzy/normalized replace
            new_bp, success = fuzzy_replace(current_bp, old_text, new_text)
            if success:
                current_bp = new_bp
                bp_updated = True
                tool_calls_display.append(f"✍️ EDIT_BLUEPRINT (Fuzzy Match):\n  FIND: {old_text[:100]}...\n  REPLACE: {new_text[:100]}...")
            else:
                edit_errors.append(f"Failed to find exact or fuzzy text to replace: '{old_text[:50]}...'")

    # Clean tags from text
    cleaned = re.sub(r'\[(?:ADD_TO|REWRITE|EDIT)_BLUEPRINT\].*?(\[\/(?:ADD_TO|REWRITE|EDIT)_BLUEPRINT\]|$)', '', full_text, flags=re.DOTALL).strip()
    
    return cleaned, "\n\n---\n\n".join(tool_calls_display), current_bp, bp_updated, edit_errors

@app.route('/toggle_pipeline_phase', methods=['POST'])
def toggle_pipeline_phase():
    path = get_active_chat_path()
    chat_data = load_chat_data(path)
    if chat_data.get("chat_type") != "pipeline":
        return jsonify({"error": "Not a pipeline chat"}), 400
    
    current = chat_data.get("pipeline_phase", "architect")
    new_phase = "scribe" if current == "architect" else "architect"
    chat_data["pipeline_phase"] = new_phase
    
    if new_phase == "scribe":
        if "scribe_messages" not in chat_data: chat_data["scribe_messages"] = []
        chat_data["scribe_messages"].append({
            "role": "system", 
            "content": "PHASE CHANGE: Now in SCRIBE mode. Your job as Architect is done. You are now the Scribe. Follow the [STORY BLUEPRINT] strictly. Use vivid prose. Do not use blueprint tools anymore; just write the story."
        })
    else:
        if "architect_messages" not in chat_data: chat_data["architect_messages"] = []
        chat_data["architect_messages"].append({
            "role": "system", 
            "content": "PHASE CHANGE: Now in ARCHITECT mode. Focus on building and refining the Story Blueprint. Use [ADD_TO_BLUEPRINT], [REWRITE_BLUEPRINT], or [EDIT_BLUEPRINT] tools."
        })
    
    save_chat_data(path, chat_data)
    return jsonify({"success": True, "new_phase": new_phase})

@app.route('/update_blueprint', methods=['POST'])
def update_blueprint():
    data = request.json
    path = get_active_chat_path()
    chat_data = load_chat_data(path)
    chat_data["blueprint"] = data.get("blueprint", "")
    save_chat_data(path, chat_data)
    return jsonify({"success": True})

@app.route('/get_history', methods=['GET'])
def get_history():
    path = get_active_chat_path()
    raw_data = read_json(path, {})
    if isinstance(raw_data, dict) and raw_data.get("is_arena"):
        return jsonify(raw_data)
        
    data = load_chat_data(path)
    
    is_pipeline = data.get("chat_type") == "pipeline"
    phase = data.get("pipeline_phase", "architect")
    
    if is_pipeline:
        history_key = "scribe_messages" if phase == "scribe" else "architect_messages"
        msgs = data.get(history_key, [])
        if not msgs:
            # First time scribe load
            if phase == "scribe":
                msgs = [{"role": "system", "content": read_text(FILES["pipeline_scribe_prompt"])}]
                data["scribe_messages"] = msgs
                save_chat_data(path, data)
            else:
                msgs = [{"role": "system", "content": read_text(FILES["pipeline_architect_prompt"])}]
                data["architect_messages"] = msgs
                save_chat_data(path, data)
    else:
        msgs = data.get("messages", [])

    if len(msgs) > 0 and msgs[0]["role"] == "system":
        if is_pipeline:
            if phase == "architect":
                msgs[0]["content"] = read_text(FILES["pipeline_architect_prompt"])
            else:
                msgs[0]["content"] = read_text(FILES["pipeline_scribe_prompt"])
        else:
            msgs[0]["content"] = read_text(FILES["main_prompt"])

    has_backup = False
    if "backup_summaries" in data and data["backup_summaries"] is not None:
        if data["backup_summaries"] != data.get("summaries", []):
            has_backup = True

    return jsonify({
        "history": msgs, 
        "summaries": data["summaries"], 
        "visual_memory": data["visual_memory"], 
        "self_memory": data.get("self_memory", ""),
        "memory_logs": data.get("memory_logs", []),
        "character_slug": data.get("character_slug"),
        "has_backup": has_backup,
        "audit_context": data.get("audit_context", {"batches": [], "includeRaw": True}),
        "chat_type": data.get("chat_type", "standard"),
        "pipeline_phase": data.get("pipeline_phase", "architect"),
        "blueprint": data.get("blueprint", "")
    })

@app.route('/check_summary_status', methods=['POST'])
def check_summary_status():
    path = get_active_chat_path()
    data = load_chat_data(path)
    msgs = data["messages"]
    sums = data["summaries"]

    req_set = request.json.get('settings', {})
    keep = int(req_set.get("recent_turns_to_keep", 4))
    batch_size = int(req_set.get("batch_size", 4))

    last_idx = sums[-1]["end_index"] if sums else 0

    valid_indices = [i for i, m in enumerate(msgs) if i > 0 and not (isinstance(m.get('content', ''), str) and m.get('content', '').startswith('__IMG_JSON__'))]
    available_valid = [i for i in valid_indices if i > last_idx]

    summarizable_count = len(available_valid) - keep
    if summarizable_count < 0: summarizable_count = 0

    batches_needed = summarizable_count // batch_size

    return jsonify({
        "total_messages": len(msgs),
        "unsummarized_text_messages": len(available_valid),
        "batches_pending": batches_needed
    })

@app.route('/force_summarize', methods=['POST'])
def force_summarize():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        mode = data.get('mode')
        path = get_active_chat_path()
        chat_data = load_chat_data(path)

        if mode == 'all':
            chat_data["backup_summaries"] = json.loads(json.dumps(chat_data.get("summaries", [])))
            chat_data["summaries"] = []
            chat_data = process_summaries(chat_data)
            save_chat_data(path, chat_data)
            return jsonify({"success": True})

        elif mode == 'batch':
            b_idx = data.get('batch_index')
            if b_idx is not None and 0 <= b_idx < len(chat_data.get("summaries", [])):
                chat_data["backup_summaries"] = json.loads(json.dumps(chat_data.get("summaries", [])))
                s_set = read_json(FILES["summarizer_settings"], {"enabled": False})
                sums = chat_data["summaries"]
                target = sums[b_idx]

                msgs = chat_data["messages"]
                batch_segment = msgs[target["start_index"] : target["end_index"] + 1]
                batch_cleaned = clean_for_api(batch_segment)
                text_to_sum = "\n".join([f"{m['role'].upper()}: {get_text_content(m)[:1000]}" for m in batch_cleaned])

                context_summaries = sums[max(0, b_idx-3):b_idx]
                context_str = "\n\n".join([f"PREVIOUS SUMMARY BLOCK:\n{s['content']}" for s in context_summaries])

                user_content = ""
                if context_str:
                    user_content += f"--- CONTEXT: PREVIOUS SUMMARIES (Do not re-summarize these) ---\n{context_str}\n\n"
                user_content += f"--- NEW MESSAGES TO SUMMARIZE (Summarize the following only) ---\n{text_to_sum}"

                h = {"Authorization": f"Bearer {VENICE_API_KEY}"}
                sum_model = s_set.get("model", "qwen3-4b")
                cache_key = os.path.basename(path).replace('.json', '') + "_sum"
                p = {
                    "model": sum_model,
                    "temperature": 0.3,
                    "messages": apply_claude_caching([
                        {"role": "system", "content": s_set.get("system_prompt", "Summarize.")},
                        {"role": "user", "content": user_content}
                    ], sum_model),
                    "prompt_cache_key": cache_key,
                    "venice_parameters": {
                        "include_venice_system_prompt": False,
                        "strip_thinking_response": True
                    }
                }

                resp_raw = requests.post(VENICE_URL, headers=h, json=p)
                resp = resp_raw.json()

                if 'choices' not in resp:
                    return jsonify({"error": f"API Error: {resp}"}), 500

                summary_text = resp['choices'][0]['message']['content']
                usage = resp.get('usage', {})

                target["content"] = summary_text
                target["usage"] = usage
                save_chat_data(path, chat_data)
                return jsonify({"success": True})
            return jsonify({"error": "Invalid batch index"}), 400

        elif mode == 'batches':
            indices = data.get('batch_indices', [])
            chat_data["backup_summaries"] = json.loads(json.dumps(chat_data.get("summaries", [])))
            s_set = read_json(FILES["summarizer_settings"], {"enabled": False})
            sums = chat_data["summaries"]

            for b_idx in indices:
                if b_idx is not None and 0 <= b_idx < len(sums):
                    target = sums[b_idx]
                    if target.get("is_consolidated"): continue

                    batch_segment = chat_data["messages"][target["start_index"] : target["end_index"] + 1]
                    batch_cleaned = clean_for_api(batch_segment)
                    text_to_sum = "\n".join([f"{m['role'].upper()}: {get_text_content(m)[:1000]}" for m in batch_cleaned])

                    context_summaries = sums[max(0, b_idx-3):b_idx]
                    context_str = "\n\n".join([f"PREVIOUS SUMMARY BLOCK:\n{s['content']}" for s in context_summaries if not s.get("disabled")])

                    user_content = ""
                    if context_str:
                        user_content += f"--- CONTEXT: PREVIOUS SUMMARIES (Do not re-summarize these) ---\n{context_str}\n\n"
                    user_content += f"--- NEW MESSAGES TO SUMMARIZE (Summarize the following only) ---\n{text_to_sum}"

                    h = {"Authorization": f"Bearer {VENICE_API_KEY}"}
                    sum_model = s_set.get("model", "qwen3-4b")
                    cache_key = os.path.basename(path).replace('.json', '') + f"_sum_{b_idx}"
                    p = {
                        "model": sum_model,
                        "temperature": 0.3,
                        "messages": apply_claude_caching([
                            {"role": "system", "content": s_set.get("system_prompt", "Summarize.")},
                            {"role": "user", "content": user_content}
                        ], sum_model),
                        "prompt_cache_key": cache_key,
                        "venice_parameters": {
                            "include_venice_system_prompt": False,
                            "strip_thinking_response": True
                        }
                    }

                    try:
                        resp = requests.post(VENICE_URL, headers=h, json=p).json()
                        if 'choices' in resp:
                            target["content"] = resp['choices'][0]['message']['content']
                            target["usage"] = resp.get('usage', {})
                    except Exception as e:
                        print(f"Batch {b_idx} failed: {e}")

            save_chat_data(path, chat_data)
            return jsonify({"success": True})

        return jsonify({"error": "Invalid mode"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/toggle_summary_state', methods=['POST'])
def toggle_summary_state():
    data = request.json
    indices = data.get('indices', [])
    state = data.get('disabled', False)
    path = get_active_chat_path()
    chat_data = load_chat_data(path)
    sums = chat_data.get("summaries", [])

    for idx in indices:
        if 0 <= idx < len(sums):
            sums[idx]["disabled"] = state

    save_chat_data(path, chat_data)
    return jsonify({"success": True})

@app.route('/branch_chat', methods=['POST'])
def branch_chat():
    try:
        idx = request.json.get('index')
        path = get_active_chat_path()
        data = load_chat_data(path)

        is_pipeline = data.get("chat_type") == "pipeline"
        phase = data.get("pipeline_phase", "architect")
        history_key = "scribe_messages" if phase == "scribe" else "architect_messages"
        
        sliced_msgs = data.get(history_key, [])[:idx + 1]
        sliced_sums = [s for s in data.get("summaries", []) if s["end_index"] <= idx]

        new_data = {
            "summaries": sliced_sums,
            "visual_memory": data.get("visual_memory", "")
        }
        
        if is_pipeline:
            new_data["chat_type"] = "pipeline"
            new_data["pipeline_phase"] = phase
            new_data["blueprint"] = data.get("blueprint", "")
            new_data["pipeline_settings"] = data.get("pipeline_settings", {})
            new_data[history_key] = sliced_msgs
            # Keep other track empty or copy it? Separation suggests keeping it empty or as is
            other_key = "architect_messages" if phase == "scribe" else "scribe_messages"
            new_data[other_key] = data.get(other_key, [])
            new_data["messages"] = sliced_msgs
        else:
            new_data["messages"] = sliced_msgs

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        old_name = os.path.basename(path).replace('.json', '')
        new_fn = f"{old_name}_Branch_{ts}.json"
        new_path = os.path.join(FILES["conversations_dir"], new_fn)

        save_chat_data(new_path, new_data)
        write_json(FILES["active_meta"], {"filename": new_fn})

        return jsonify({"success": True, "new_filename": new_fn})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/consolidate_summaries', methods=['POST'])
def consolidate_summaries():
    data = request.json
    indices = data.get('indices', [])
    if not indices or len(indices) < 2:
        return jsonify({"error": "Need at least 2 summaries to consolidate."}), 400

    path = get_active_chat_path()
    chat_data = load_chat_data(path)
    sums = chat_data.get("summaries", [])

    if max(indices) >= len(sums) or min(indices) < 0:
        return jsonify({"error": "Invalid indices."}), 400

    selected_sums = [sums[i] for i in indices]

    s_set = read_json(FILES["summarizer_settings"], {})
    cons_model = s_set.get("consolidation_model", "venice-uncensored")
    cons_prompt = read_text(FILES["summary_consolidator_prompt"])

    text_to_condense = ""
    for i, s in enumerate(selected_sums):
        text_to_condense += f"--- SUMMARY PART {i+1} ---\n{s['content']}\n\n"

    h = {"Authorization": f"Bearer {VENICE_API_KEY}"}
    cache_key = os.path.basename(path).replace('.json', '') + "_cons"
    p = {
        "model": cons_model,
        "temperature": 0.3,
        "messages": apply_claude_caching([
            {"role": "system", "content": cons_prompt},
            {"role": "user", "content": f"Please condense the following summaries into a single seamless narrative block:\n\n{text_to_condense}"}
        ], cons_model),
        "prompt_cache_key": cache_key,
        "venice_parameters": {
            "include_venice_system_prompt": False,
            "strip_thinking_response": True
        }
    }

    try:
        r = requests.post(VENICE_URL, headers=h, json=p)
        r.raise_for_status()
        resp = r.json()
        if 'choices' not in resp:
            return jsonify({"error": f"API Error: {resp}"}), 500

        consolidated_text = resp['choices'][0]['message']['content'].strip()
        usage = resp.get('usage', {})

        new_sum = {
            "start_index": selected_sums[0]["start_index"],
            "end_index": selected_sums[-1]["end_index"],
            "content": consolidated_text,
            "usage": usage,
            "is_consolidated": True,
            "original_summaries": selected_sums
        }

        chat_data["backup_summaries"] = json.loads(json.dumps(sums))
        new_sums_list = sums[:indices[0]] + [new_sum] + sums[indices[-1]+1:]
        chat_data["summaries"] = new_sums_list
        save_chat_data(path, chat_data)

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/undo_consolidation', methods=['POST'])
def undo_consolidation():
    data = request.json
    idx = data.get('index')
    if idx is None:
        return jsonify({"error": "No index provided."}), 400

    path = get_active_chat_path()
    chat_data = load_chat_data(path)
    sums = chat_data.get("summaries", [])

    if idx < 0 or idx >= len(sums):
        return jsonify({"error": "Invalid index."}), 400

    target = sums[idx]
    if not target.get("is_consolidated"):
        return jsonify({"error": "Target is not a consolidated summary."}), 400

    originals = target.get("original_summaries", [])

    chat_data["backup_summaries"] = json.loads(json.dumps(sums))
    new_sums_list = sums[:idx] + originals + sums[idx+1:]
    chat_data["summaries"] = new_sums_list
    save_chat_data(path, chat_data)

    return jsonify({"success": True})

@app.route('/ai_refine_message', methods=['POST'])
def ai_refine_message():
    try:
        req = request.json
        idx = req.get('index')
        guidance = req.get('guidance', '').strip()
        path = get_active_chat_path()
        data = load_chat_data(path)

        if idx < 0 or idx >= len(data["messages"]):
            return jsonify({"error": "Invalid message index"}), 400

        msg_content = data["messages"][idx]["content"]
        text_to_refine = get_text_content(data["messages"][idx])

        ref_set = read_json(FILES["refiner_settings"], {"model": "venice-uncensored", "temperature": 0.3})
        system_prompt = read_text(FILES["refine_prompt"])

        # Extract context parameters
        include_context = req.get('include_context', False)
        context_depth = int(req.get('context_depth', 5))

        context_block = ""
        if include_context and idx > 0:
            start_ctx = max(0, idx - context_depth)
            # Slice history up to the target message
            ctx_msgs = data["messages"][start_ctx : idx]
            context_block = "### CONTEXT (DO NOT EDIT THESE MESSAGES):\n"
            for m in ctx_msgs:
                role = m.get('role', 'unknown').upper()
                txt = get_text_content(m)
                if txt:
                    context_block += f"[{role}]: {txt}\n\n"
            context_block += "-------------------\n\n"

        # Construct final prompt with labels for clarity
        user_content = f"{context_block}### MESSAGE TO EDIT (TARGET):\n{text_to_refine}\n\n### EDIT INSTRUCTIONS:\n{guidance if guidance else 'Polish the prose and fix any minor errors while keeping the meaning identical.'}"

        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        model_id = ref_set.get("model", "venice-uncensored")
        cache_key = os.path.basename(path).replace('.json', '') + "_refine"
        payload = {
            "model": model_id,
            "temperature": float(ref_set.get("temperature", 0.3)),
            "messages": apply_claude_caching([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ], model_id),
            "prompt_cache_key": cache_key,
            "venice_parameters": {
                "include_venice_system_prompt": False,
                "strip_thinking_response": True
            }
        }

        r = requests.post(VENICE_URL, headers=headers, json=payload)
        resp = r.json()

        if 'choices' in resp and resp['choices']:
            raw_response = resp['choices'][0]['message']['content'].strip()
            
            if raw_response == "NO CHANGES REQUIRED":
                return jsonify({"success": True, "no_changes": True})

            # Parser for [REPLACE]...[WITH]...[/REPLACE]
            import re
            pattern = re.compile(r'\[REPLACE\](.*?)\[WITH\](.*?)\[/REPLACE\]', re.DOTALL)
            matches = pattern.findall(raw_response)
            
            if not matches:
                # Fallback: if model failed to use tool and just gave full text
                refined_text = raw_response
            else:
                refined_text = text_to_refine
                for replace_text, with_text in matches:
                    replace_text = replace_text.strip('\n') # Allow some whitespace slack in tool use
                    with_text = with_text.strip('\n')
                    if replace_text in refined_text:
                        refined_text = refined_text.replace(replace_text, with_text)
                    else:
                        print(f"[REFINER ERROR] Could not find exact match for replacement: {replace_text}")

            data["messages"][idx]["original_content"] = msg_content
            data["messages"][idx]["content"] = refined_text
            data["messages"][idx]["refine_logic"] = raw_response
            save_chat_data(path, data)
            return jsonify({"success": True, "refined_text": refined_text, "refine_logic": raw_response})
        else:
            return jsonify({"error": resp.get('error', {}).get('message', 'API Error')}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/undo_ai_refine', methods=['POST'])
def undo_ai_refine():
    try:
        idx = request.json.get('index')
        path = get_active_chat_path()
        data = load_chat_data(path)

        if idx < 0 or idx >= len(data["messages"]):
            return jsonify({"error": "Invalid message index"}), 400

        msg = data["messages"][idx]
        if "original_content" in msg:
            msg["content"] = msg["original_content"]
            del msg["original_content"]
            save_chat_data(path, data)
            return jsonify({"success": True})
        else:
            return jsonify({"error": "No original content to revert to."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/find_replace_message', methods=['POST'])
def find_replace_message():
    try:
        req = request.json
        idx = req.get('index')
        find_str = req.get('find', '')
        replace_str = req.get('replace', '')

        if not find_str:
            return jsonify({"error": "Nothing to find."}), 400

        path = get_active_chat_path()
        data = load_chat_data(path)

        if idx < 0 or idx >= len(data["messages"]):
            return jsonify({"error": "Invalid message index"}), 400

        msg = data["messages"][idx]
        content = msg.get("content", "")

        if isinstance(content, str):
            if find_str in content:
                msg["original_content"] = content
                msg["content"] = content.replace(find_str, replace_str)
                save_chat_data(path, data)
                return jsonify({"success": True, "new_content": msg["content"]})
            else:
                return jsonify({"error": f"'{find_str}' not found in message."}), 404
        elif isinstance(content, list):
            # Handle vision content lists
            changed = False
            for part in content:
                if part.get("type") == "text" and find_str in part.get("text", ""):
                    if not changed:
                        msg["original_content"] = json.loads(json.dumps(content))
                    part["text"] = part["text"].replace(find_str, replace_str)
                    changed = True
            
            if changed:
                save_chat_data(path, data)
                return jsonify({"success": True, "new_content": content})
            else:
                return jsonify({"error": f"'{find_str}' not found in message text."}), 404

        return jsonify({"error": "Unsupported message format."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_banned_phrases', methods=['GET'])
def get_banned_phrases():
    return jsonify({"banned_phrases": read_text(FILES["banned_phrases"])})

@app.route('/save_banned_phrases', methods=['POST'])
def save_banned_phrases():
    text = request.json.get('banned_phrases', '')
    write_text(FILES["banned_phrases"], text)
    return jsonify({"success": True})

@app.route('/write_for_me', methods=['POST'])
def write_for_me():
    try:
        path = get_active_chat_path()
        data = load_chat_data(path)
        req_data = request.json or {}
        guidance = req_data.get('guidance', '').strip()

        wfm_set = read_json(FILES["wfm_settings"], {"model": "venice-uncensored", "temperature": 0.8, "context_depth": 10})
        depth = int(wfm_set.get("context_depth", 10))

        all_msgs = [m for m in data["messages"] if m['role'] != 'system' and not (isinstance(m.get('content', ''), str) and m.get('content', '').startswith('__IMG_JSON__'))]

        # Get only user messages for style mimicry
        user_msgs = [m for m in all_msgs if m['role'] == 'user']
        style_examples = user_msgs[-depth:] if depth > 0 else []
        style_block = "\n\n".join([f"EXAMPLE USER MESSAGE:\n{get_text_content(m)}" for m in style_examples])

        # Get the single most recent assistant message to respond to
        assistant_msgs = [m for m in all_msgs if m['role'] == 'assistant']
        last_assistant_msg = assistant_msgs[-1] if assistant_msgs else None

        situation_block = ""
        if last_assistant_msg:
            situation_block = f"### MESSAGE TO RESPOND TO:\n{get_text_content(last_assistant_msg)}\n\nInstruction: Respond to the message above using the exact same style, voice, and format as the EXAMPLE USER MESSAGES provided below."

        system_instr = read_text(FILES["user_mimic_prompt"])
        if guidance:
            system_instr += f"\n\nUse the following raw text to guide the content of your reply, enriching it even if it seems like it's already complete: {guidance}"

        prompt_content = f"{situation_block}\n\n### USER STYLE EXAMPLES (MIMIC THIS STYLE):\n{style_block}"

        messages = [
            {"role": "system", "content": system_instr},
            {"role": "user", "content": prompt_content}
        ]

        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        model_id = wfm_set.get("model", "venice-uncensored")
        cache_key = os.path.basename(path).replace('.json', '') + "_wfm"
        payload = {
            "model": model_id,
            "temperature": float(wfm_set.get("temperature", 0.8)),
            "messages": apply_claude_caching(messages, model_id),
            "prompt_cache_key": cache_key,
            "venice_parameters": {
                "include_venice_system_prompt": False,
                "strip_thinking_response": True
            }
        }

        resp = requests.post(VENICE_URL, headers=headers, json=payload)
        r = resp.json()

        if 'choices' in r and r['choices']:
            reply = r['choices'][0]['message']['content'].strip()
            if (reply.startswith('"') and reply.endswith('"')) or (reply.startswith("'") and reply.endswith("'")):
                reply = reply[1:-1]

            debug_info = {
                "full_prompt_sent": messages,
                "usage": r.get('usage', {})
            }

            return jsonify({
                "success": True, 
                "text": reply, 
                "debug": debug_info
            })
        else:
            return jsonify({"error": r.get('error', {}).get('message', 'Unknown API Error')}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_characters', methods=['GET'])
def get_characters():
    search = request.args.get('search', '')
    limit = request.args.get('limit', 50)

    cache = read_json(FILES["character_cache"], {"timestamp": 0, "data": []})
    now = time.time()

    if not search and cache["data"] and (now - cache["timestamp"] < 3600):
        return jsonify(cache["data"])

    try:
        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        params = {"limit": limit}
        if search: params["search"] = search

        r = requests.get(VENICE_CHARACTERS_URL, headers=headers, params=params)

        # Log IO
        res_data = r.json() if r.status_code == 200 else r.text
        update_last_io(VENICE_CHARACTERS_URL, params, res_data)

        r.raise_for_status()
        data = res_data.get("data", [])

        if not search:
            write_json(FILES["character_cache"], {"timestamp": now, "data": data})

        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/create_character', methods=['POST'])
def create_character():
    try:
        headers = {"Authorization": f"Bearer {VENICE_API_KEY}", "Content-Type": "application/json"}
        payload = request.json
        r = requests.post(VENICE_CHARACTERS_URL, headers=headers, json=payload)
        res_data = r.json() if r.status_code in [200, 201] else r.text
        update_last_io(f"POST {VENICE_CHARACTERS_URL}", payload, res_data)
        r.raise_for_status()
        # Clear cache so new character shows up
        write_json(FILES["character_cache"], {"timestamp": 0, "data": []})
        return jsonify({"success": True, "data": res_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/edit_character/<slug>', methods=['PUT'])
def edit_character(slug):
    try:
        headers = {"Authorization": f"Bearer {VENICE_API_KEY}", "Content-Type": "application/json"}
        payload = request.json
        url = f"{VENICE_CHARACTERS_URL}/{slug}"
        r = requests.put(url, headers=headers, json=payload)
        res_data = r.json() if r.status_code == 200 else r.text
        update_last_io(f"PUT {url}", payload, res_data)
        r.raise_for_status()
        write_json(FILES["character_cache"], {"timestamp": 0, "data": []})
        return jsonify({"success": True, "data": res_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_character/<slug>', methods=['DELETE'])
def delete_character(slug):
    try:
        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        url = f"{VENICE_CHARACTERS_URL}/{slug}"
        r = requests.delete(url, headers=headers)
        res_data = r.json() if r.status_code == 200 else r.text
        update_last_io(f"DELETE {url}", None, res_data)
        r.raise_for_status()
        write_json(FILES["character_cache"], {"timestamp": 0, "data": []})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_character_image', methods=['GET'])
def get_character_image():
    url = request.args.get('url')
    if not url: return Response("Missing URL", status=400)

    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            return Response(r.content, mimetype=r.headers.get('Content-Type', 'image/jpeg'))
        else:
            print(f"[IMAGE PROXY ERROR] {r.status_code} for URL: {url} - Resp: {r.text[:200]}")
            # Provide transparent pixel or redirect to placeholder instead of 500 error
            return app.send_static_file('error_placeholder.png') if os.path.exists('static/error_placeholder.png') else Response("Image failed", status=404)
    except Exception as e:
        print(f"[IMAGE PROXY EXCEPTION] {e}")
        return Response(str(e), status=500)

@app.route('/get_character_details/<slug>', methods=['GET'])
def get_character_details(slug):
    try:
        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        url = f"{VENICE_CHARACTERS_URL}/{slug}"
        r = requests.get(url, headers=headers)
        res_data = r.json() if r.status_code == 200 else r.text
        update_last_io(url, None, res_data)
        r.raise_for_status()
        return jsonify(res_data.get("data", {}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_last_io')
def get_last_io():
    return jsonify(LAST_IO)

@app.route('/get_balance')
def get_balance_route():
    return jsonify(get_persisted_balance())

@app.route('/open_audit', methods=['POST'])
def open_audit():
    parent_fn = request.json.get('filename')
    if not parent_fn: return jsonify({"error": "No filename provided"}), 400
    
    audit_fn = parent_fn.replace('.json', '.audit.json')
    if parent_fn.endswith('.audit.json'):
        audit_fn = parent_fn
        
    audit_path = os.path.join(FILES["conversations_dir"], audit_fn)
    
    if not os.path.exists(audit_path):
        write_json(audit_path, {
            "messages": [{"role": "system", "content": read_text(FILES["audit_prompt"])}],
            "parent_file": parent_fn
        })
        
    write_json(FILES["active_meta"], {"filename": audit_fn})
    return jsonify({"success": True, "filename": audit_fn})

@app.route('/get_parent_context', methods=['GET'])
def get_parent_context():
    path = get_active_chat_path()
    data = load_chat_data(path)
    if "parent_file" not in data:
        return jsonify({"error": "Not an audit chat"}), 400
        
    p_path = os.path.join(FILES["conversations_dir"], data["parent_file"])
    if not os.path.exists(p_path):
        return jsonify({"error": "Parent chat not found"}), 404
        
    p_data = load_chat_data(p_path)
    return jsonify({"success": True, "summaries": p_data.get("summaries", [])})

@app.route('/save_audit_context', methods=['POST'])
def save_audit_context():
    path = get_active_chat_path()
    data = load_chat_data(path)
    data["audit_context"] = request.json
    save_chat_data(path, data)
    return jsonify({"success": True})

@app.route('/apply_audit_fix', methods=['POST'])
def apply_audit_fix():
    req = request.json
    idx = int(req.get('index'))
    new_text = req.get('new_text')
    
    path = get_active_chat_path()
    data = load_chat_data(path)
    if "parent_file" not in data: return jsonify({"error": "Not an audit chat"}), 400
    
    p_path = os.path.join(FILES["conversations_dir"], data["parent_file"])
    if not os.path.exists(p_path): return jsonify({"error": "Parent chat not found"}), 404
    
    p_data = load_chat_data(p_path)
    
    if 0 <= idx < len(p_data.get("summaries", [])):
        p_data["backup_summaries"] = json.loads(json.dumps(p_data.get("summaries", [])))
        p_data["summaries"][idx]["content"] = new_text
        save_chat_data(p_path, p_data)
        return jsonify({"success": True})
    return jsonify({"error": "Invalid index"}), 400

@app.route('/update_character', methods=['POST'])
def update_character():
    slug = request.json.get('slug')
    model_id = request.json.get('modelId')
    path = get_active_chat_path()

    # Update current file
    data = load_chat_data(path)
    data["character_slug"] = slug
    save_chat_data(path, data)

    # Update metadata to preserve this character for the NEXT "new chat"
    meta = read_json(FILES["active_meta"], {})
    meta["pending_character"] = slug
    write_json(FILES["active_meta"], meta)

    # Model Affinity Sync
    if model_id:
        v_set = read_json(FILES["venice_settings"], {})
        v_set["model"] = model_id
        write_json(FILES["venice_settings"], v_set)

    return jsonify({"success": True})

@app.route('/clear_backups', methods=['POST'])
def clear_backups():
    path = get_active_chat_path()
    chat_data = load_chat_data(path)
    if "backup_summaries" in chat_data:
        del chat_data["backup_summaries"]
        save_chat_data(path, chat_data)
    return jsonify({"success": True})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    path = get_active_chat_path()
    chat_data = load_chat_data(path)

    custom_model = data.get('custom_model')
    v_set = read_json(FILES["venice_settings"], {})
    model_to_use = custom_model if custom_model else v_set.get("model", "venice-uncensored")

    if chat_data.get("chat_type") == "pipeline":
        if custom_model:
            model_to_use = custom_model
        else:
            # Use global model from v_set
            model_to_use = v_set.get("model", "venice-uncensored")

    if len(chat_data["messages"]) == 1:
        # Extract text for filename safely (handling vision list format)
        raw_text = get_text_content({"content": data.get('message', '')})
        clean = "".join([c for c in raw_text[:25] if c.isalnum() or c==' ']).strip().replace(" ", "_")
        if not clean: clean = "Image_Chat"

        ts = datetime.datetime.now().strftime("%m%d_%H%M")
        new_fn = f"{clean}_{ts}.json"
        new_path = os.path.join(FILES["conversations_dir"], new_fn)
        os.rename(path, new_path)
        write_json(FILES["active_meta"], {"filename": new_fn})
        path = new_path

    if data.get('message'):
        message_content = data['message']
        # If it's a vision message with base64, save image locally
        if isinstance(message_content, list):
            for part in message_content:
                if part.get('type') == 'image_url':
                    part['image_url']['url'] = save_base64_image(part['image_url']['url'])

        # Dual track saving
        if chat_data.get("chat_type") == "pipeline":
            phase = chat_data.get("pipeline_phase", "architect")
            track_key = "scribe_messages" if phase == "scribe" else "architect_messages"
            if track_key not in chat_data:
                chat_data[track_key] = []
            print(f"[PIPELINE USER APPEND] phase={phase}, track_key={track_key}, track_len_before_user_append={len(chat_data.get(track_key, []))}, fallback_messages_len={len(chat_data.get('messages', []))}")
            chat_data[track_key].append({"role": "user", "content": message_content, "timestamp": datetime.datetime.now().isoformat()})
        else:
            chat_data["messages"].append({"role": "user", "content": message_content, "timestamp": datetime.datetime.now().isoformat()})

    # Prepare assistant placeholder
    ast_msg = {"role": "assistant", "content": "", "model": model_to_use, "timestamp": datetime.datetime.now().isoformat()}
    if chat_data.get("chat_type") == "pipeline":
        phase = chat_data.get("pipeline_phase", "architect")
        track_key = "scribe_messages" if phase == "scribe" else "architect_messages"
        if track_key not in chat_data:
            chat_data[track_key] = []
        chat_data[track_key].append(ast_msg)
        idx = len(chat_data[track_key]) - 1
        print(f"[PIPELINE ASSISTANT PLACEHOLDER] phase={phase}, track_key={track_key}, idx={idx}, track_len_after_assistant_append={len(chat_data.get(track_key, []))}, fallback_messages_len={len(chat_data.get('messages', []))}")
    else:
        chat_data["messages"].append(ast_msg)
        idx = len(chat_data["messages"]) - 1
        
    save_chat_data(path, chat_data)

    def generate():
        nonlocal chat_data
        track_key = "messages"
        if chat_data.get("chat_type") == "pipeline":
            phase = chat_data.get("pipeline_phase", "architect")
            track_key = "scribe_messages" if phase == "scribe" else "architect_messages"
            print(f"[PIPELINE STREAM] phase={phase}, track_key={track_key}, idx={idx}, track_len={len(chat_data.get(track_key, []))}, fallback_messages_len={len(chat_data.get('messages', []))}")

        # Immediate yield to keep connection alive
        yield f"data: {json.dumps({'status': 'Connecting to Venice...'})}\n\n"

        # Build context from appropriate track
        context = build_context(chat_data, user_query=get_text_content({"content": data.get('message', '')}), current_model=model_to_use)

        # Handle Audit Context Injection
        audit_ctx_req = data.get('audit_context')
        if audit_ctx_req and chat_data.get('parent_file'):
            p_path = os.path.join(FILES["conversations_dir"], chat_data['parent_file'])
            if os.path.exists(p_path):
                p_data = load_chat_data(p_path)
                sel_batches = audit_ctx_req.get('batches', [])
                inc_raw = audit_ctx_req.get('include_raw', True)
                
                ctx_text = "--- SYSTEM: PROVIDED AUDIT CONTEXT FROM ORIGINAL CHAT ---\n\n"
                for b_idx in sel_batches:
                    if 0 <= b_idx < len(p_data.get("summaries", [])):
                        batch = p_data["summaries"][b_idx]
                        ctx_text += f"=== SUMMARY BATCH #{b_idx + 1} (index={b_idx}) ===\n{batch['content']}\n\n"
                        if inc_raw:
                            raw_msgs = p_data["messages"][batch["start_index"]:batch["end_index"]+1]
                            ctx_text += f"--- RAW MESSAGES FOR BATCH #{b_idx + 1} ---\n"
                            for rm in clean_for_api(raw_msgs):
                                if rm['role'] != 'system':
                                    ctx_text += f"{rm['role'].upper()}: {get_text_content(rm)}\n\n"
                ctx_text += "---------------------------------------------------------\n"
                
                # Insert right before the user's latest message
                if len(context) >= 2:
                    context.insert(-1, {"role": "system", "content": ctx_text})

        # Character Logic ...
        if chat_data.get("character_slug") and context and context[0]["role"] == "system":
            pass

        # Token Analytics ...
        char_system = 0
        char_summary = 0
        char_raw = 0
        for m in context:
            c = m.get('content', '')
            content_str = get_text_content(m) if isinstance(c, list) else str(c)
            c_len = len(content_str)
            role = m.get('role')
            if role == 'system':
                if "RECENT SUMMARY" in content_str or "CONSOLIDATED ARCHIVE" in content_str: char_summary += c_len
                else: char_system += c_len
            else: char_raw += c_len
        total_chars = char_system + char_summary + char_raw

        # 2. Start Main Generation
        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        cache_key = os.path.basename(path).replace('.json', '')

        reasoning_effort = v_set.get("reasoning_effort", "medium")
        disable_thinking = False
        if reasoning_effort == "none":
            disable_thinking = True
            reasoning_effort = "low" # Fallback depth, but thinking is disabled

        venice_params = {
            "include_venice_system_prompt": v_set.get("include_venice_system_prompt", True),
            "strip_thinking_response": False,
            "disable_thinking": disable_thinking,
            "reasoning_effort": reasoning_effort
        }

        if chat_data.get("character_slug"):
            venice_params["character_slug"] = chat_data["character_slug"]
            venice_params["include_venice_system_prompt"] = True

            # Pass web search/scraping settings if provided in v_set
            if v_set.get("enable_web_search"):
                venice_params["enable_web_search"] = v_set.get("enable_web_search", "off")
            if v_set.get("enable_web_scraping"):
                venice_params["enable_web_scraping"] = v_set.get("enable_web_scraping", False)

        payload = {
            "model": model_to_use,
            "temperature": float(v_set.get("temperature", 0.7)),
            "max_tokens": int(v_set.get("max_tokens", 4000)),
            "presence_penalty": float(v_set.get("presence_penalty", 0.0)),
            "frequency_penalty": float(v_set.get("frequency_penalty", 0.0)),
            "messages": apply_claude_caching(context, model_to_use), 
            "stream": True,
            "prompt_cache_key": cache_key,
            "venice_parameters": venice_params
        }

        if is_reasoning_model(model_to_use):
            payload["reasoning_effort"] = v_set.get("reasoning_effort", "medium")

        # Store for the Last IO debugger
        update_last_io(VENICE_URL, payload, None)

        # --- CACHE DEBUG LOGGING ---
        try:
            log_path = os.path.join(FILES["payload_logs_dir"], os.path.basename(path))
            existing_logs = read_json(log_path, {"prev": None, "last": None})
            write_json(log_path, {
                "prev": existing_logs.get("last"),
                "last": payload["messages"]
            })
        except Exception as e:
            print(f"Error saving payload log: {e}")

        full = ""
        reasoning = ""
        usage = None
        balance = None
        full_response_json = []

        # Memory parsing variables
        capturing_memory = False
        memory_buffer = ""
        
        capturing_live_sum = False
        live_sum_buffer = ""
        
        chunk_buffer = ""

        try:
            with requests.post(VENICE_URL, headers=headers, json=payload, stream=True) as r:
                if r.status_code != 200:
                    raise Exception(f"{r.status_code} Client Error: {r.text}")
                balance = r.headers.get('x-venice-balance-usd')
                save_persisted_balance(balance)
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if "[DONE]" in decoded: break
                        try:
                            chunk_raw = decoded[6:]
                            chunk = json.loads(chunk_raw)
                            full_response_json.append(chunk)
                            if "usage" in chunk: usage = chunk["usage"]
                            if len(chunk['choices'])>0:
                                delta = chunk['choices'][0]['delta']
                                
                                if 'reasoning_content' in delta and delta['reasoning_content']:
                                    r_part = delta['reasoning_content']
                                    reasoning += r_part
                                    chat_data[track_key][idx]["reasoning"] = reasoning
                                    yield f"data: {json.dumps({'reasoning': r_part})}\n\n"

                                if 'content' in delta and delta['content']:
                                    c = delta['content']
                                    chunk_buffer += c
                                    
                                    while True: # process buffer until no more tags can be resolved
                                        if capturing_live_sum:
                                            if '[/SUM]' in chunk_buffer:
                                                pre, post = chunk_buffer.split('[/SUM]', 1)
                                                live_sum_buffer += pre
                                                yield f"data: {json.dumps({'live_summary': pre})}\n\n"
                                                capturing_live_sum = False
                                                chunk_buffer = post
                                                continue
                                            else:
                                                safe_idx = len(chunk_buffer)
                                                for i in range(1, len('[/SUM]')):
                                                    if chunk_buffer.endswith('[/SUM]'[:i]):
                                                        safe_idx = len(chunk_buffer) - i
                                                        break
                                                if safe_idx > 0:
                                                    to_yield = chunk_buffer[:safe_idx]
                                                    live_sum_buffer += to_yield
                                                    yield f"data: {json.dumps({'live_summary': to_yield})}\n\n"
                                                    chunk_buffer = chunk_buffer[safe_idx:]
                                                break # need more chunks

                                        elif capturing_memory:
                                            if '[/MEMORY_ACTION]' in chunk_buffer:
                                                pre, post = chunk_buffer.split('[/MEMORY_ACTION]', 1)
                                                memory_buffer += pre
                                                capturing_memory = False
                                                chunk_buffer = post
                                                continue
                                            else:
                                                safe_idx = len(chunk_buffer)
                                                for i in range(1, len('[/MEMORY_ACTION]')):
                                                    if chunk_buffer.endswith('[/MEMORY_ACTION]'[:i]):
                                                        safe_idx = len(chunk_buffer) - i
                                                        break
                                                if safe_idx > 0:
                                                    memory_buffer += chunk_buffer[:safe_idx]
                                                    chunk_buffer = chunk_buffer[safe_idx:]
                                                break

                                        else:
                                            # check for start tags
                                            sum_idx = chunk_buffer.find('[SUM]')
                                            mem_idx = chunk_buffer.find('[MEMORY_ACTION]')
                                            
                                            first_tag = None
                                            first_idx = -1
                                            if sum_idx != -1 and mem_idx != -1:
                                                if sum_idx < mem_idx:
                                                    first_tag, first_idx = '[SUM]', sum_idx
                                                else:
                                                    first_tag, first_idx = '[MEMORY_ACTION]', mem_idx
                                            elif sum_idx != -1:
                                                first_tag, first_idx = '[SUM]', sum_idx
                                            elif mem_idx != -1:
                                                first_tag, first_idx = '[MEMORY_ACTION]', mem_idx
                                                
                                            if first_tag == '[SUM]':
                                                pre = chunk_buffer[:first_idx]
                                                post = chunk_buffer[first_idx + len('[SUM]'):]
                                                full += pre
                                                if pre:
                                                    yield f"data: {json.dumps({'content': pre, 'balance': balance})}\n\n"
                                                capturing_live_sum = True
                                                chunk_buffer = post
                                                continue
                                            elif first_tag == '[MEMORY_ACTION]':
                                                pre = chunk_buffer[:first_idx]
                                                post = chunk_buffer[first_idx + len('[MEMORY_ACTION]'):]
                                                full += pre
                                                if pre:
                                                    yield f"data: {json.dumps({'content': pre, 'balance': balance})}\n\n"
                                                capturing_memory = True
                                                chunk_buffer = post
                                                continue
                                            else:
                                                # no complete tags, check partials
                                                safe_idx = len(chunk_buffer)
                                                for t in ['[SUM]', '[MEMORY_ACTION]']:
                                                    for i in range(1, len(t)):
                                                        if chunk_buffer.endswith(t[:i]):
                                                            safe_idx = min(safe_idx, len(chunk_buffer) - i)
                                                if safe_idx > 0:
                                                    to_yield = chunk_buffer[:safe_idx]
                                                    full += to_yield
                                                    yield f"data: {json.dumps({'content': to_yield, 'balance': balance})}\n\n"
                                                    chunk_buffer = chunk_buffer[safe_idx:]
                                                break

                                chat_data[track_key][idx]["content"] = full
                                save_chat_data(path, chat_data)
                        except: pass

            # Flush remaining buffer
            if chunk_buffer:
                if capturing_live_sum:
                    live_sum_buffer += chunk_buffer
                    yield f"data: {json.dumps({'live_summary': chunk_buffer})}\n\n"
                elif capturing_memory:
                    memory_buffer += chunk_buffer
                else:
                    full += chunk_buffer
                    yield f"data: {json.dumps({'content': chunk_buffer, 'balance': balance})}\n\n"
                    
            # 1. Store Prose
            chat_data[track_key][idx]["content"] = full

            # 2. Process Tools (If Architect)
            if chat_data.get("chat_type") == "pipeline" and chat_data.get("pipeline_phase") == "architect":
                cleaned_prose, tool_display, new_bp, bp_updated, edit_errors = process_blueprint_tools(chat_data, full)
                
                chat_data[track_key][idx]["content"] = cleaned_prose
                if tool_display:
                    chat_data[track_key][idx]["blueprint_tool_calls"] = tool_display
                    yield f"data: {json.dumps({'tool_calls': tool_display})}\n\n"
                
                if bp_updated:
                    chat_data["blueprint"] = new_bp
                    yield f"data: {json.dumps({'blueprint_update': new_bp})}\n\n"
                
                if edit_errors:
                    err_msg = "\n".join(edit_errors)
                    chat_data[track_key].append({"role": "system", "content": f"Automated Notice: Your targeted blueprint edit failed.\n{err_msg}\nPlease ensure the <find> block exactly matches the text currently in the document. You may use [REWRITE_BLUEPRINT] if targeted editing fails."})
                
                yield f"data: {json.dumps({'content_overwrite': cleaned_prose})}\n\n"

            # 3. Process Live Summary
            if live_sum_buffer:
                chat_data[track_key][idx]["live_summary"] = live_sum_buffer.strip()
            
            save_chat_data(path, chat_data)

            # Post-stream surgical memory update logic
            if memory_buffer:
                raw_actions = memory_buffer
                if '[/MEMORY_ACTION]' in raw_actions:
                    raw_actions = raw_actions.split('[/MEMORY_ACTION]')[0]
                
                # Reload chat data right before update to ensure we aren't using a stale copy
                fresh_data = load_chat_data(path)
                updated_mem = fresh_data.get("self_memory", "")
                changes_made = []

                # 1. Handle [REPLACE]...[WITH]...[/REPLACE]
                replace_pattern = re.compile(r'\[REPLACE\]\s*(.*?)\s*\[WITH\]\s*(.*?)\s*\[/REPLACE\]', re.DOTALL)
                for match in replace_pattern.finditer(raw_actions):
                    old_text = match.group(1).strip()
                    new_text = match.group(2).strip()
                    if old_text and old_text in updated_mem:
                        updated_mem = updated_mem.replace(old_text, new_text)
                        snip = f"{new_text[:15]}...{new_text[-15:]}" if len(new_text) > 30 else new_text
                        changes_made.append(f"REPLACED text with: '{snip}'")
                    elif old_text:
                        print(f"[MEMORY TOOL ERROR] Could not find exact match to replace: {old_text[:50]}...")

                # 2. Handle [ADD]...[/ADD]
                add_pattern = re.compile(r'\[ADD\]\s*(.*?)\s*\[/ADD\]', re.DOTALL)
                for match in add_pattern.finditer(raw_actions):
                    addition = match.group(1).strip()
                    if addition:
                        if updated_mem:
                            updated_mem += "\n" + addition
                        else:
                            updated_mem = addition
                        snip = f"{addition[:15]}...{addition[-15:]}" if len(addition) > 30 else addition
                        changes_made.append(f"ADDED: '{snip}'")

                if changes_made:
                    fresh_data["self_memory"] = updated_mem.strip()
                    log_entry = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "model": f"AI Model ({model_to_use})",
                        "change": "; ".join(changes_made)
                    }
                    if "memory_logs" not in fresh_data: fresh_data["memory_logs"] = []
                    fresh_data["memory_logs"].insert(0, log_entry)
                    save_chat_data(path, fresh_data)
                    
                    # Update local state for subsequent summary processing in same turn if needed
                    chat_data = fresh_data
                    
                    yield f"data: {json.dumps({'memory_update': updated_mem.strip(), 'memory_logs': fresh_data['memory_logs']})}\n\n"

            # Store final combined response.
            # In pipeline architect mode, the content may already have been cleaned by process_blueprint_tools(),
            # so do not overwrite it with the raw full text containing blueprint tool tags.
            update_last_io(VENICE_URL, payload, full_response_json)
            is_pipeline_architect = (
                chat_data.get("chat_type") == "pipeline"
                and chat_data.get("pipeline_phase") == "architect"
            )
            if not is_pipeline_architect:
                chat_data[track_key][idx]["content"] = full
                save_chat_data(path, chat_data)
                yield f"data: {json.dumps({'content_overwrite': full})}\n\n"
            else:
                save_chat_data(path, chat_data)

            # Store final combined response
            update_last_io(VENICE_URL, payload, full_response_json)

            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                if total_chars > 0:
                    usage["breakdown"] = {
                        "summary": round(prompt_tokens * (char_summary / total_chars)),
                        "system": round(prompt_tokens * (char_system / total_chars)),
                        "raw": round(prompt_tokens * (char_raw / total_chars))
                    }

                chat_data[track_key][idx]["usage"] = usage
                save_chat_data(path, chat_data)
                yield f"data: {json.dumps({'usage': usage, 'balance': balance})}\n\n"

            # Post-stream: log to ledger and handle summarization in background
            def background_tasks(u, b_before, m_id, p_path, c_data):
                # 1. Throttle balance fetching (only fetch real balance every 5 calls)
                ledger_data = read_json(FILES["api_ledger"], {"calls": []})
                call_count = len(ledger_data.get("calls", []))
                
                real_bal = None
                if call_count % 5 == 0:
                    real_bal = fetch_real_balance()
                    if real_bal:
                        save_persisted_balance(real_bal)
                
                # 2. Log API call
                log_api_call(
                    feature="Chat Response",
                    model=m_id,
                    chat_file=os.path.basename(p_path),
                    usage=u,
                    balance_before=b_before,
                    balance_after=real_bal
                )
                
                # 3. Post-response Summarization (moved here to prevent blocking start of stream)
                if c_data.get("chat_type") != "pipeline":
                    s_set = read_json(FILES["summarizer_settings"], {"enabled": False})
                    if s_set.get("enabled"):
                        # Exhaust the generator to perform the summary work
                        for _ in process_summaries(c_data):
                            pass
                        save_chat_data(p_path, c_data)
            
            threading.Thread(target=background_tasks, args=(usage, balance, model_to_use, path, chat_data)).start()

        except Exception as e:
            target_messages = chat_data.get(track_key, [])
            if not full and not reasoning:
                if 0 <= idx < len(target_messages):
                    target_messages.pop(idx)
            else:
                if 0 <= idx < len(target_messages):
                    target_messages[idx]["content"] = full + f"\n\n*(Stream error: {str(e)})*"
            save_chat_data(path, chat_data)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/generate_image', methods=['POST'])
def generate_image():
    path = get_active_chat_path()
    chat_data = load_chat_data(path)

    req_data = request.json
    guidance = req_data.get('guidance', '')
    debug_mode = req_data.get('debug_mode', False)

    i_set = read_json(FILES["img_settings"], {})
    art_style = i_set.get("art_style", "None")
    neg_styles = i_set.get("negative_styles", "")

    prompt = ""
    debug_logs = {"prompt_generation": None, "image_request": None}

    # Step 1: Prompt Generation
    if debug_mode and guidance:
        prompt = guidance
        debug_logs["prompt_generation"] = "DEBUG MODE: Bypassed LLM. Using raw guidance as prompt."
    else:
        text_msgs = clean_for_api(chat_data["messages"])
        valid = [m for m in text_msgs if m['role'] != 'system']
        context_msgs = valid[-5:] if len(valid) >= 5 else valid

        context_str = "\n".join([f"{m['role'].upper()}: {get_text_content(m)}" for m in context_msgs])
        if guidance: 
            context_str += f"\n\nUSER GUIDANCE FOR IMAGE: {guidance}"

        if art_style and art_style != "None":
            context_str += f"\n\nREQUIRED ART STYLE: {art_style}"

        vis_mem = chat_data.get("visual_memory", "")
        if vis_mem: 
            context_str = f"--- CHARACTER VISUAL GUIDELINES ---\n{vis_mem}\n\n--- CURRENT SCENE CONTEXT ---\n" + context_str

        if neg_styles:
            context_str += f"\n\nDO NOT USE ANY OF THE FOLLOWING STYLES OR ELEMENTS: {neg_styles}"

        try:
            p_set = read_json(FILES["venice_img_settings"], {})
            h = {"Authorization": f"Bearer {VENICE_API_KEY}"}
            prompt_model = p_set.get("model", "venice-uncensored")

            p_res_raw = requests.post(VENICE_URL, headers=h, json={
                "model": prompt_model,
                "temperature": 0.1,
                "messages": [
                    {"role":"system","content":read_text(FILES["img_prompt_instr"])},
                    {"role":"user","content":context_str}
                ],
                "venice_parameters": {
                    "include_venice_system_prompt": False,
                    "strip_thinking_response": True
                }
            }, timeout=45)

            p_res = p_res_raw.json()
            debug_logs["prompt_generation"] = {"payload": context_str, "response": p_res}

            # Log image prompt generation to ledger in background
            def log_img_bg(u, b_before, m_id, p_path):
                log_api_call(
                    feature="Image Prompt Gen",
                    model=m_id,
                    chat_file=os.path.basename(p_path),
                    usage=u,
                    balance_before=b_before,
                    balance_after=fetch_real_balance()
                )
            
            img_usage = p_res.get('usage') if 'choices' in p_res else None
            img_balance = p_res_raw.headers.get('x-venice-balance-usd')
            threading.Thread(target=log_img_bg, args=(img_usage, img_balance, prompt_model, path)).start()

            if 'choices' not in p_res:
                error_msg = p_res.get('error', {}).get('message', 'Unknown Error')
                return jsonify({"error": f"LLM Prompt Gen Failed: {error_msg}", "debug": debug_logs if debug_mode else None}), 500

            prompt = p_res['choices'][0]['message']['content'].strip()
        except Exception as e:
            return jsonify({"error": f"LLM Prompt Gen Exception: {str(e)}", "debug": debug_logs if debug_mode else None}), 500

    # Step 2: Image Generation
    try:
        fh = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}

        # Z-Image Turbo specific optimizations: 8 steps is mandatory for quality/speed ratio.
        # It does not accept CFG or other standard SD parameters.
        payload = {
            "prompt": prompt, 
            "num_inference_steps": 8, 
            "enable_safety_checker": False, 
            "enable_prompt_expansion": False,
            "image_size": {
                "width": int(i_set.get("width", 1024)), 
                "height": int(i_set.get("height", 1024))
            },
            "output_format": "jpeg"
        }

        f_res_raw = requests.post(FAL_URL, headers=fh, json=payload, timeout=60)

        try:
            f_res = f_res_raw.json()
        except:
            f_res = {"raw_text": f_res_raw.text}

        debug_logs["image_request"] = {"payload": payload, "response": f_res, "status": f_res_raw.status_code}

        if 'images' not in f_res:
            error_detail = f_res.get('detail', f_res.get('raw_text', 'Unknown error'))
            return jsonify({"error": f"FAL Image Gen Error {f_res_raw.status_code}: {error_detail}", "debug": debug_logs if debug_mode else None}), 500

        url = f_res['images'][0]['url']

        img_msg = {"role": "assistant", "content": f"__IMG_JSON__{json.dumps({'url': url, 'prompt': prompt})}"}
        chat_data["messages"].append(img_msg)
        save_chat_data(path, chat_data)

        return jsonify({"success": True, "debug": debug_logs if debug_mode else None})

    except Exception as e:
        return jsonify({"error": f"Internal Server Exception: {str(e)}", "debug": debug_logs if debug_mode else None}), 500

# --- SCANNER BACKGROUND JOBS ---
SCAN_JOBS = {} # chat_filename -> {"status": "running"|"completed"|"error", "message": str}

def background_scan_task(chat_path, model_id, start_num, end_num):
    try:
        filename = os.path.basename(chat_path)
        SCAN_JOBS[filename] = {"status": "running", "message": "Initiating background scan..."}
        
        chat_data = load_chat_data(chat_path)
        msgs = clean_for_api(chat_data["messages"])
        content_msgs = [m for m in msgs if m["role"] != "system"]
        
        if start_num is not None or end_num is not None:
            start_idx = (int(start_num) - 1) if start_num else 0
            end_idx = int(end_num) if end_num else len(content_msgs)
            target_msgs = content_msgs[start_idx:end_idx]
        else:
            target_msgs = content_msgs[-50:]

        blob = "\n".join([f"{m['role'].upper()}: {get_text_content(m)}" for m in target_msgs])
        existing_mem = chat_data.get("visual_memory", "")
        
        system_instr = read_text(FILES["visual_prompt"])
        
        # Reinforced Payload: System Prompt at Start and End
        context = [
            {"role": "system", "content": system_instr},
            {"role": "user", "content": f"### EXISTING VISUAL MEMORY:\n{existing_mem if existing_mem else '[Empty]'}\n\n### RECENT CHAT LOG TO ANALYZE:\n{blob}"},
            {"role": "system", "content": f"REMINDER: Use the [ADD] and [REPLACE] tools to update the Visual Memory based on the log above. Do not roleplay. Output ONLY tool calls.\n\n### REITERATION OF ROLE:\n{system_instr}"}
        ]

        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        payload = {
            "model": model_id,
            "messages": apply_claude_caching(context, model_id),
            "temperature": 0.1,
            "venice_parameters": {
                "include_venice_system_prompt": False,
                "strip_thinking_response": True
            }
        }

        r = requests.post(VENICE_URL, headers=headers, json=payload, timeout=120)
        update_last_io(f"Background Visual Scan: {filename}", payload, r.text)
        r.raise_for_status()

        res = r.json()
        if 'choices' in res and res['choices']:
            raw_output = res['choices'][0]['message']['content'].strip()
            
            if raw_output == "NO CHANGES REQUIRED":
                SCAN_JOBS[filename] = {"status": "completed", "message": "Scan finished: No changes required."}
                return

            # Surgical Application Logic
            updated_mem = existing_mem
            
            # Process [REPLACE] blocks
            replace_pattern = re.compile(r'\[REPLACE\](.*?)\[WITH\](.*?)\[/REPLACE\]', re.DOTALL)
            for old_text, new_text in replace_pattern.findall(raw_output):
                old_text = old_text.strip('\n')
                new_text = new_text.strip('\n')
                if old_text in updated_mem:
                    updated_mem = updated_mem.replace(old_text, new_text)
                else:
                    print(f"[SCANNER] Replacement failed: '{old_text}' not found.")

            # Process [ADD] blocks
            add_pattern = re.compile(r'\[ADD\](.*?)\[/ADD\]', re.DOTALL)
            for addition in add_pattern.findall(raw_output):
                addition = addition.strip('\n')
                if updated_mem:
                    updated_mem += "\n" + addition
                else:
                    updated_mem = addition

            chat_data["visual_memory"] = updated_mem.strip()
            save_chat_data(chat_path, chat_data)
            SCAN_JOBS[filename] = {"status": "completed", "message": "Scan finished: Memory updated."}
        else:
            SCAN_JOBS[filename] = {"status": "error", "message": "API returned empty response."}
            
    except Exception as e:
        print(f"[SCANNER ERROR] {traceback.format_exc()}")
        SCAN_JOBS[os.path.basename(chat_path)] = {"status": "error", "message": str(e)}

@app.route('/scan_visuals', methods=['POST'])
def scan_visuals():
    req = request.json
    path = get_active_chat_path()
    
    # Check if a job is already running
    filename = os.path.basename(path)
    if SCAN_JOBS.get(filename, {}).get("status") == "running":
        return jsonify({"success": False, "error": "A scan is already in progress for this chat."}), 409

    start_num = req.get('start')
    end_num = req.get('end')
    model_id = req.get('model')
    
    if not model_id:
        p_set = read_json(FILES["venice_img_settings"], {})
        model_id = p_set.get("visual_scan_model", p_set.get("model", "venice-uncensored"))

    # Launch background thread
    thread = threading.Thread(target=background_scan_task, args=(path, model_id, start_num, end_num))
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "Scan started in background."})

@app.route('/check_scan_status', methods=['GET'])
def check_scan_status():
    path = get_active_chat_path()
    filename = os.path.basename(path)
    job = SCAN_JOBS.get(filename, {"status": "idle", "message": "No active scan."})
    return jsonify(job)

@app.route('/update_history', methods=['POST'])
def update_history():
    path = get_active_chat_path()
    data = load_chat_data(path)
    
    new_history = request.json.get('history')
    
    if data.get("chat_type") == "pipeline":
        phase = data.get("pipeline_phase", "architect")
        history_key = "scribe_messages" if phase == "scribe" else "architect_messages"
        data[history_key] = new_history
        # Keep 'messages' in sync if needed or just use it as fallback
        data["messages"] = new_history
    else:
        data["messages"] = new_history
        
    if len(new_history) < (data.get("summaries", [])[-1]["end_index"] if data.get("summaries") else 0):
        data["summaries"] = [] 
        
    save_chat_data(path, data)
    return jsonify({"success": True})

@app.route('/update_visual_memory', methods=['POST'])
def update_visual_memory():
    path = get_active_chat_path()
    data = load_chat_data(path)
    mem_type = request.json.get('type', 'visual')
    content = request.json.get('memory')
    
    if mem_type == 'self':
        data["self_memory"] = content
        snip = f"{content[:15]}...{content[-15:]}" if len(content) > 30 else content
        data["memory_logs"].insert(0, {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model": "User (Manual Edit)",
            "change": f"Manually edited to: '{snip}'"
        })
    else:
        data["visual_memory"] = content
        
    save_chat_data(path, data)
    return jsonify({"success": True})

@app.route('/sidebar_data')
def sidebar_data():
    files = []
    with os.scandir(FILES["conversations_dir"]) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".json"):
                files.append({"name": entry.name, "time": entry.stat().st_mtime, "is_audit": entry.name.endswith(".audit.json")})
    files.sort(key=lambda x: x["time"], reverse=True)
    meta = read_json(FILES["active_meta"], {})
    return jsonify({"chats": [f["name"] for f in files], "active_chat": meta.get("filename")})

@app.route('/rename_chat', methods=['POST'])
def rename_chat():
    old, new_base = request.json.get('old'), request.json.get('new')

    new_name = new_base + ".json"
    counter = 1
    while os.path.exists(os.path.join(FILES["conversations_dir"], new_name)):
        new_name = f"{new_base}_{counter}.json"
        counter += 1

    os.rename(
        os.path.join(FILES["conversations_dir"], old), 
        os.path.join(FILES["conversations_dir"], new_name)
    )

    meta = read_json(FILES["active_meta"], {})
    if meta.get("filename") == old: 
        write_json(FILES["active_meta"], {"filename": new_name})

    return jsonify({"success": True, "new_filename": new_name})

@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    fn = request.json.get('filename')
    path = os.path.join(FILES["conversations_dir"], fn)

    try:
        if os.path.exists(path):
            data = load_chat_data(path)

            # Delete associated images
            for m in data.get("messages", []):
                c = m.get('content', '')
                # Handle standard image messages
                if isinstance(c, str) and c.startswith('__IMG_JSON__'):
                    try:
                        img_info = json.loads(c.replace('__IMG_JSON__', ''))
                        url = img_info.get('url')
                        if url and url.startswith('/static/uploads/'):
                            img_path = os.path.join(FILES["uploads_dir"], os.path.basename(url))
                            if os.path.exists(img_path): os.remove(img_path)
                    except: pass
                # Handle vision input images
                elif isinstance(c, list):
                    for part in c:
                        if part.get('type') == 'image_url':
                            url = part['image_url']['url']
                            if url.startswith('/static/uploads/'):
                                img_path = os.path.join(FILES["uploads_dir"], os.path.basename(url))
                                if os.path.exists(img_path): os.remove(img_path)

            # Delete the conversation file
            os.remove(path)

            # Cleanup payload logs
            log_path = os.path.join(FILES["payload_logs_dir"], fn)
            if os.path.exists(log_path): os.remove(log_path)

            return jsonify({"success": True})
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_chat_title', methods=['POST'])
def generate_chat_title():
    try:
        data = request.json
        model_id = data.get('model', 'venice-uncensored')
        range_data = data.get('range', {"start": 1, "end": 15})
        
        path = get_active_chat_path()
        chat_data = load_chat_data(path)

        # Get text messages
        all_text_msgs = [m for m in chat_data["messages"] if m["role"] != "system" and not (isinstance(m.get('content', ''), str) and m.get('content', '').startswith("__IMG_JSON__"))]
        
        start = max(0, range_data.get('start', 1) - 1)
        end = min(len(all_text_msgs), range_data.get('end', 15))
        
        history = all_text_msgs[start:end]
        history_text = "\n".join([f"{m['role'].upper()}: {get_text_content(m)[:500]}" for m in history])

        prompt = f"Based on the conversation snippet below (messages {start+1} to {end}), output a short, descriptive four-word title that summarizes the topic. Output ONLY the four-word title. Do not include any formatting, preamble, or additional text.\n\nCONVERSATION SNIPPET:\n{history_text}"

        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "venice_parameters": {"include_venice_system_prompt": False, "strip_thinking_response": True}
        }

        r = requests.post(VENICE_URL, headers=headers, json=payload)
        resp = r.json()

        if 'choices' in resp:
            title = resp['choices'][0]['message']['content'].strip()
            # Basic cleanup for filename safety
            title = title.replace(" ", "_")
            title = "".join([c for c in title if c.isalnum() or c == '_'])
            return jsonify({"success": True, "title": title, "old_filename": os.path.basename(path)})
        return jsonify({"error": "API Error"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/venice_models')
def venice_models():
    return jsonify(read_json('data/venice_models.json', {}))

@app.route('/rebuild_index', methods=['POST'])
def rebuild_index_route():
    success, msg = rebuild_lore_index()
    return jsonify({"success": success, "message": msg})

@app.route('/extract_lore', methods=['POST'])
def extract_lore():
    try:
        req = request.json
        start_num = req.get('start', 1)
        end_num = req.get('end', 999999)
        batch_size = req.get('batch_size', 20)

        path = get_active_chat_path()
        data = load_chat_data(path)
        msgs = data.get("messages", [])

        text_msgs = []
        msg_counter = 0
        for m in msgs:
            if m["role"] == "system": continue
            if isinstance(m.get('content', ''), str) and m.get('content', '').startswith("__IMG_JSON__"): continue
            msg_counter += 1
            if start_num <= msg_counter <= end_num:
                text_msgs.append(m)

        if not text_msgs:
            return jsonify({"error": "No text messages found in that range."}), 400

        rag_set = read_json(FILES["rag_settings"], {})
        use_lorebook = rag_set.get("extraction_send_lorebook", True)
        ext_model = rag_set.get("extraction_model", "venice-uncensored")
        ext_temp = float(rag_set.get("extraction_temp", 0.3))

        full_prompt = read_text(FILES["lore_extractor_prompt"])

        # Split prompt into base instructions and lorebook specific instructions
        prompt_parts = full_prompt.split("--- LOREBOOK CONTEXTUAL INSTRUCTIONS ---")
        base_prompt = prompt_parts[0].strip()
        context_instructions = prompt_parts[1].strip() if len(prompt_parts) > 1 else ""

        # Construct final system prompt based on toggle
        final_system_prompt = base_prompt
        if use_lorebook:
            final_system_prompt += "\n\n" + context_instructions
            existing_lore = read_text(FILES["lorebook"])
            user_prefix = f"EXISTING LORE:\n{existing_lore}\n\n" if existing_lore else ""
        else:
            user_prefix = ""

        h = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        extracted_blocks = []

        for i in range(0, len(text_msgs), batch_size):
            batch = text_msgs[i:i+batch_size]
            history_text = "\n".join([f"{m['role'].upper()}: {get_text_content(m)}" for m in batch])

            cache_key = os.path.basename(path).replace('.json', '') + "_lore"
            p = {
                "model": ext_model,
                "temperature": ext_temp,
                "messages": apply_claude_caching([
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": f"{user_prefix}CHAT HISTORY BATCH:\n\n{history_text}"}
                ], ext_model),
                "prompt_cache_key": cache_key,
                "venice_parameters": {
                    "include_venice_system_prompt": False,
                    "strip_thinking_response": True
                }
            }

            r = requests.post(VENICE_URL, headers=h, json=p)
            r.raise_for_status()
            resp = r.json()

            content = resp['choices'][0]['message']['content'].strip()
            if content.upper() != "NO NEW LORE." and content.upper() != "NO NEW LORE":
                extracted_blocks.append(content)

        final_lore = "\n\n".join(extracted_blocks)
        return jsonify({"success": True, "lore": final_lore})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_lorebook', methods=['GET'])
def get_lorebook():
    return jsonify({"text": read_text(FILES["lorebook"])})

@app.route('/save_lorebook', methods=['POST'])
def save_lorebook():
    text = request.json.get('text', '')
    write_text(FILES["lorebook"], text)
    return jsonify({"success": True})

def chunk_text_for_tts(text, max_chars=2000):
    """Consolidated chunking logic. Splits by sentence to avoid mid-word breaks."""
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = re.sub(r'\[.*?\]\(.*?\)', '', clean_text)
    clean_text = re.sub(r'[*_`#]', '', clean_text)
    clean_text = clean_text.replace('\n', ' ').strip()

    sentences = re.split('(?<=[.!?]) +', clean_text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += (" " if current else "") + s
        else:
            if current: chunks.append(current.strip())
            current = s
    if current: chunks.append(current.strip())
    return chunks

@app.route('/tts', methods=['POST'])
def tts():
    """Generates and returns a URL to a concatenated MP3 file for a message."""
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "No text provided"}), 400

        t_set = read_json(FILES["tts_settings"], {"enabled": False, "model": "tts-kokoro", "voice": "af_sky", "speed": 1.0})

        is_mistral = t_set.get("model", "").startswith("voxtral")
        ref_audio = data.get('ref_audio')

        # Fallback: Read sample.wav from server disk if Mistral and no UI upload
        if is_mistral and not ref_audio:
            try:
                if os.path.exists("sample.wav"):
                    with open("sample.wav", "rb") as f:
                        ref_audio = base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                print(f"Error reading sample.wav: {e}")

        # Check Cache first
        cache_fn = get_tts_cache_path(text, t_set.get("model"), t_set.get("voice"), t_set.get("speed"), ref_audio)
        cache_path = os.path.join(FILES["audio_cache_dir"], cache_fn)

        if os.path.exists(cache_path):
            return jsonify({
                "url": f"/static/audio_cache/{cache_fn}",
                "cached": True
            })

        text_chunks = chunk_text_for_tts(text)
        log_tts_event("job_started", {
            "input_len": len(text),
            "chunks": len(text_chunks),
            "settings": t_set,
            "has_ref_audio": bool(ref_audio)
        })

        api_key = MISTRAL_API_KEY if is_mistral else VENICE_API_KEY
        endpoint = MISTRAL_SPEECH_URL if is_mistral else VENICE_SPEECH_URL
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        success_count = 0
        last_balance = "0"

        wav_params = None
        frames = bytearray()

        chunk_errors = []

        for i, chunk in enumerate(text_chunks):
            if not chunk: continue

            if is_mistral:
                payload = {
                    "model": t_set.get("model", "voxtral-mini-tts-2603"),
                    "input": chunk,
                    "response_format": "wav"
                }
                # Priority 1: One-off clone from uploaded file in UI
                if ref_audio:
                    payload["ref_audio"] = ref_audio
                # Priority 2: Use saved Voice ID from settings
                else:
                    payload["voice_id"] = t_set.get("voice")
            else:
                payload = {
                    "model": t_set.get("model", "tts-kokoro"),
                    "input": chunk,
                    "voice": t_set.get("voice", "af_sky"),
                    "response_format": "wav",
                    "speed": float(t_set.get("speed", 1.0))
                }

            try:
                r = requests.post(endpoint, headers=headers, json=payload, timeout=90)

                # Store Last IO for TTS debugging
                update_last_io(endpoint, payload, {"status": r.status_code, "length": len(r.content) if r.status_code == 200 else 0})

                if r.status_code == 200:
                    audio_content = r.content

                    # Handle Mistral's JSON-wrapped base64 response
                    if is_mistral:
                        try:
                            json_res = r.json()
                            if isinstance(json_res, dict) and 'audio_data' in json_res:
                                audio_content = base64.b64decode(json_res['audio_data'])
                        except Exception as json_e:
                            print(f"Mistral JSON decode error: {json_e}")
                            # Fallback if it's already binary for some reason
                            pass

                    try:
                        with wave.open(io.BytesIO(audio_content), 'rb') as w:
                            if wav_params is None:
                                wav_params = w.getparams()
                            frames.extend(w.readframes(w.getnframes()))
                        success_count += 1
                    except Exception as wave_e:
                        chunk_errors.append({
                            "chunk": i,
                            "type": "wave_decode_error",
                            "error": str(wave_e),
                            "content_preview": str(audio_content[:50])
                        })

                    if r.headers.get('x-venice-balance-usd'):
                        last_balance = r.headers.get('x-venice-balance-usd')
                        save_persisted_balance(last_balance)
                else:
                    chunk_errors.append({
                        "chunk": i,
                        "type": "http_error",
                        "status": r.status_code,
                        "error": r.text,
                        "payload": payload
                    })
            except Exception as e:
                chunk_errors.append({"chunk": i, "type": "exception", "error": str(e)})

        log_tts_event("job_finished", {
            "total": len(text_chunks), 
            "success": success_count, 
            "bytes": len(frames),
            "errors": chunk_errors
        })

        if success_count == 0 or wav_params is None:
            return jsonify({
                "error": "Failed to generate any valid audio chunks.",
                "details": chunk_errors
            }), 500

        # Save to disk as a properly formatted WAV file
        with wave.open(cache_path, "wb") as w:
            w.setparams(wav_params)
            w.writeframes(frames)

        return jsonify({
            "url": f"/static/audio_cache/{cache_fn}",
            "balance": last_balance,
            "cached": False
        })

    except Exception as e:
        log_tts_event("global_error", {"error": str(e)})
        return jsonify({"error": str(e)}), 500

@app.route('/get_last_tts_log')
def get_last_tts_log():
    return jsonify(LAST_TTS_LOG)

@app.route('/get_settings')
def get_settings():
    v_set = read_json(FILES["venice_settings"], {"model": "venice-uncensored", "temperature": 0.9, "max_tokens": 4000, "reasoning_effort": "medium"})
    return jsonify({
        "main_prompt": read_text(FILES["main_prompt"]),
        "venice": v_set,
        "evaluator_model": v_set.get("evaluator_model", "venice-uncensored"),
        "venice_img": read_json(FILES["venice_img_settings"], {}),
        "refiner": read_json(FILES["refiner_settings"], {"model": "venice-uncensored", "temperature": 0.3}),
        "image_gen": read_json(FILES["img_settings"], {"model": "lustify-v7", "steps": 40, "cfg_scale": 7.5, "width": 864, "height": 1152}),
        "summarizer": read_json(FILES["summarizer_settings"], {
            "enabled": False, 
            "trigger_threshold_turns": 12, 
            "batch_size": 4, 
            "recent_turns_to_keep": 12, 
            "system_prompt": "Summarize.", 
            "consolidation_model": "venice-uncensored", 
            "consolidation_prompt": "You are a meticulous archivist. Condense the provided sequential story summaries into a highly factual, detailed, and objective overarching summary. Focus heavily on concrete events, lore, world-building facts, character arcs, and exact details. Strip away all flowery language, purple prose, and stylistic embellishments. Write as a pure, dense factual record in a single narrative block. Do not use lists."
        }),
        "rag": read_json(FILES["rag_settings"], {"enabled": False, "k": 3, "max_chars": 1200, "min_chars": 200}),
        "tts": read_json(FILES["tts_settings"], {"enabled": False, "model": "tts-kokoro", "voice": "af_sky", "speed": 1.0}),
        "wfm": read_json(FILES["wfm_settings"], {"model": "venice-uncensored", "temperature": 0.8, "context_depth": 10}),
        "interface": read_json(FILES["interface_settings"], {"font_size": 16, "bg_color": "#121212"}),
        "model_history": read_json(FILES["model_history"], [])
    })

@app.route('/save_settings', methods=['POST'])
def save_settings():
    d = request.json
    if "main_prompt" in d: write_text(FILES["main_prompt"], d["main_prompt"])
    if "venice" in d: write_json(FILES["venice_settings"], d["venice"])
    if "venice_img" in d: write_json(FILES["venice_img_settings"], d["venice_img"])
    if "refiner" in d: write_json(FILES["refiner_settings"], d["refiner"])
    if "summarizer" in d: write_json(FILES["summarizer_settings"], d["summarizer"])
    if "rag" in d: write_json(FILES["rag_settings"], d["rag"])
    if "evaluator_model" in d:
        v_set = read_json(FILES["venice_settings"], {})
        v_set["evaluator_model"] = d["evaluator_model"]
        write_json(FILES["venice_settings"], v_set)
    if "tts" in d:
        write_json(FILES["tts_settings"], d["tts"])
    if "wfm" in d: write_json(FILES["wfm_settings"], d["wfm"])
    if "interface" in d: write_json(FILES["interface_settings"], d["interface"])
    if "image_gen" in d:
        current = read_json(FILES["img_settings"], {})
        # Merge presets if they exist in file but not in incoming payload (partial update safety)
        if "presets" in current and "presets" not in d["image_gen"]:
            d["image_gen"]["presets"] = current["presets"]
        write_json(FILES["img_settings"], d["image_gen"])

    if "venice" in d and "model" in d["venice"]:
        h = read_json(FILES["model_history"], [])
        if d["venice"]["model"] not in h:
            h.insert(0, d["venice"]["model"])
            write_json(FILES["model_history"], h[:15])
    return jsonify({"success": True})

@app.route('/load_chat', methods=['POST'])
def load_chat():
    write_json(FILES["active_meta"], {"filename": request.json.get('filename')})
    return jsonify({"success": True})

@app.route('/new_chat', methods=['POST'])
def new_chat():
    write_json(FILES["active_meta"], {"filename": None})
    get_active_chat_path()
    return jsonify({"success": True})

@app.route('/analyze_chat', methods=['POST'])
def analyze_chat():
    try:
        req = request.json
        start_idx = req.get('start', 0)
        end_idx = req.get('end', 999999)
        role = req.get('role', 'both')

        path = get_active_chat_path()
        data = load_chat_data(path)
        msgs = data.get("messages", [])

        target_msgs = []
        for i, m in enumerate(msgs):
            if m["role"] == "system": continue
            if isinstance(m.get('content', ''), str) and m.get('content', '').startswith("__IMG_JSON__"): continue

            if i >= start_idx and i <= end_idx:
                if role == 'both' or m["role"] == role:
                    target_msgs.append(get_text_content(m))

        if not target_msgs:
            return jsonify({"error": "No text messages found in this range."}), 400

        full_text = " ".join(target_msgs)

        clean_text = re.sub(r'\[.*?\]\(.*?\)', '', full_text)
        clean_text = clean_text.lower()
        clean_text = clean_text.replace("â€™", "'").replace("â€˜", "'")
        clean_text = re.sub(r'[^a-z0-9\s\'.,!?;]', ' ', clean_text)
        words = clean_text.split()

        if not words:
            return jsonify({"error": "No words found after cleaning."}), 400

        stopwords = {"the","and","to","a","of","in","it","is","that","was","i","for","on","you","he","she","with","as","his","her","at","be","this","have","from","or","had","by","but","not","what","all","were","we","when","your","can","said","there","use","an","each","which","do","how","their","if","will","up","other","about","out","many","then","them","these","so","some","would","make","like","him","into","time","has","look","two","more","write","go","see","number","no","way","could","people","my","than","first","water","been","call","who","oil","its","now","find","long","down","day","did","get","come","made","may","part","me","they","are","just","very","also","because","only","even","well","any"}

        def clean_word(w):
            return w.strip(".,!?;")

        filtered_words = [clean_word(w) for w in words if clean_word(w) not in stopwords and len(clean_word(w)) > 1]

        def get_ngrams(w_list, n):
            return [" ".join(w_list[i:i+n]) for i in range(len(w_list)-n+1)]

        unigrams = Counter(filtered_words).most_common(15)
        bigrams = Counter(get_ngrams(words, 2)).most_common(15)
        trigrams = Counter(get_ngrams(words, 3)).most_common(15)
        quadgrams = Counter(get_ngrams(words, 4)).most_common(15)
        pentagrams = Counter(get_ngrams(words, 5)).most_common(15)

        total_words = len(words)
        unique_words = len(set(words))
        lexical_richness = round((unique_words / total_words) * 100, 1) if total_words > 0 else 0
        avg_words = round(total_words / len(target_msgs))

        return jsonify({
            "success": True,
            "stats": {
                "msg_count": len(target_msgs),
                "total_words": total_words,
                "unique_words": unique_words,
                "lexical_richness": lexical_richness,
                "avg_words_per_msg": avg_words
            },
            "unigrams": unigrams,
            "bigrams": bigrams,
            "trigrams": trigrams,
            "quadgrams": quadgrams,
            "pentagrams": pentagrams
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_chat_stats', methods=['GET'])
def get_chat_stats():
    path = get_active_chat_path()
    data = load_chat_data(path)
    msgs = data.get("messages", [])
    sums = data.get("summaries", [])

    def count_words(txt): return len(re.findall(r'\b\w+\b', txt))

    raw_words = sum(count_words(get_text_content(m)) for m in msgs if m['role']!='system' and not (isinstance(m.get('content',''), str) and m.get('content','').startswith('__IMG_')))
    sum1_words = sum(count_words(s['content']) for s in sums if not s.get('is_consolidated'))
    sum2_words = sum(count_words(s['content']) for s in sums if s.get('is_consolidated'))

    ctx = build_context(data)
    char_sys = 0
    char_sum = 0
    char_raw = 0
    char_mem = 0

    for m in ctx:
        c = m.get('content', '')
        if isinstance(c, str):
            c_len = len(c)
            content_str = c
        elif isinstance(c, list):
            content_str = get_text_content(m)
            c_len = len(content_str)
        else:
            c_len = 0
            content_str = ""

        role = m.get('role')
        if role == 'system':
            if "RECENT SUMMARY" in content_str or "CONSOLIDATED ARCHIVE" in content_str:
                char_sum += c_len
            elif "AI INTERNAL NOTEPAD" in content_str:
                char_mem += c_len
            else:
                char_sys += c_len
        else:
            char_raw += c_len

    return jsonify({
        "raw_words": raw_words,
        "sum1_words": sum1_words,
        "sum2_words": sum2_words,
        "proj_sys": char_sys // 4,
        "proj_sum": char_sum // 4,
        "proj_raw": char_raw // 4,
        "proj_mem": char_mem // 4,
        "proj_tot": (char_sys + char_sum + char_raw + char_mem) // 4
    })

@app.route('/get_ledger', methods=['GET'])
def get_ledger():
    """Returns the API call ledger with optional filtering."""
    ledger = read_json(FILES["api_ledger"], {"calls": []})
    calls = ledger.get("calls", [])

    # Optional filters
    feature = request.args.get('feature')
    chat_file = request.args.get('chat_file')
    limit = request.args.get('limit', type=int)

    if feature:
        calls = [c for c in calls if c.get("feature") == feature]
    if chat_file:
        calls = [c for c in calls if c.get("chat_file") == chat_file]

    # Summary stats
    total_estimated = sum(c.get("estimated_cost", 0) or 0 for c in calls)
    total_actual = sum(c.get("actual_cost", 0) or 0 for c in calls)
    avg_cache = 0
    cache_calls = [c for c in calls if c.get("prompt_tokens", 0) > 0]
    if cache_calls:
        avg_cache = round(sum(c.get("cache_hit_rate", 0) for c in cache_calls) / len(cache_calls), 1)

    # Feature breakdown
    feature_costs = {}
    for c in calls:
        f = c.get("feature", "Unknown")
        if f not in feature_costs:
            feature_costs[f] = {"count": 0, "estimated": 0, "actual": 0}
        feature_costs[f]["count"] += 1
        feature_costs[f]["estimated"] += c.get("estimated_cost", 0) or 0
        feature_costs[f]["actual"] += c.get("actual_cost", 0) or 0

    if limit:
        calls = calls[-limit:]

    return jsonify({
        "calls": calls,
        "summary": {
            "total_calls": len(ledger.get("calls", [])),
            "total_estimated": round(total_estimated, 4),
            "total_actual": round(total_actual, 4),
            "avg_cache_hit_rate": avg_cache,
            "feature_breakdown": feature_costs
        }
    })

@app.route('/clear_ledger', methods=['POST'])
def clear_ledger():
    write_json(FILES["api_ledger"], {"calls": []})
    return jsonify({"success": True})

import threading
import queue

@app.route('/compare_chat', methods=['POST'])
def compare_chat():
    data = request.json
    path = get_active_chat_path()
    chat_data = load_chat_data(path)

    history_A = data.get('history_A', [])
    history_B = data.get('history_B', [])

    v_set = read_json(FILES["venice_settings"], {})
    current_model = v_set.get("model", "venice-uncensored")

    temp_data_A = {"messages": chat_data["messages"] + history_A, "summaries": chat_data.get("summaries", []), "visual_memory": chat_data.get("visual_memory", "")}
    context_A = build_context(temp_data_A, current_model=current_model)

    backup_sums = chat_data.get("backup_summaries")
    if backup_sums is None:
        backup_sums = chat_data.get("summaries", [])
    temp_data_B = {"messages": chat_data["messages"] + history_B, "summaries": backup_sums, "visual_memory": chat_data.get("visual_memory", "")}
    context_B = build_context(temp_data_B, current_model=current_model)

    v_set = read_json(FILES["venice_settings"], {})
    headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
    payload_base = {
        "model": v_set.get("model", "venice-uncensored"),
        "temperature": float(v_set.get("temperature", 0.7)),
        "max_tokens": int(v_set.get("max_tokens", 4000)),
        "presence_penalty": float(v_set.get("presence_penalty", 0.0)),
        "frequency_penalty": float(v_set.get("frequency_penalty", 0.0)),
        "stream": True,
        "venice_parameters": {
            "include_venice_system_prompt": v_set.get("include_venice_system_prompt", True)
        }
    }

    q = queue.Queue()

    def fetch(idx, ctx):
        cache_key = os.path.basename(path).replace('.json', '') + f"_comp_{idx}"
        payload = payload_base.copy()
        payload["messages"] = apply_claude_caching(ctx, payload["model"])
        payload["prompt_cache_key"] = cache_key
        try:
            with requests.post(VENICE_URL, headers=headers, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if "[DONE]" in decoded: break
                        try:
                            chunk = json.loads(decoded[6:])
                            if len(chunk['choices'])>0:
                                c = chunk['choices'][0]['delta'].get('content', '')
                                if c: q.put((idx, {"content": c}))
                            if "usage" in chunk:
                                q.put((idx, {"usage": chunk["usage"]}))
                        except: pass
        except Exception as e:
            q.put((idx, {"error": str(e)}))
        q.put((idx, "[DONE]"))

    threading.Thread(target=fetch, args=('A', context_A)).start()
    threading.Thread(target=fetch, args=('B', context_B)).start()

    def generate():
        done_count = 0
        while done_count < 2:
            idx, d = q.get()
            if d == "[DONE]":
                done_count += 1
            else:
                yield f"data: {json.dumps({idx: d})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/resolve_comparison', methods=['POST'])
def resolve_comparison():
    data = request.json
    choice = data.get('choice')
    chosen_history = data.get('chosen_history', [])

    path = get_active_chat_path()
    chat_data = load_chat_data(path)

    if choice == 'B':
        if "backup_summaries" in chat_data:
            chat_data["summaries"] = chat_data["backup_summaries"]

    if "backup_summaries" in chat_data:
        del chat_data["backup_summaries"]

    first_ast = True
    for m in chosen_history:
        entry = {"role": m['role'], "content": m['content']}
        if m.get('usage'): entry['usage'] = m['usage']
        if m['role'] == 'assistant' and first_ast:
            entry['comparison_choice'] = choice
            first_ast = False
        chat_data["messages"].append(entry)

    save_chat_data(path, chat_data)
    chat_data = process_summaries(chat_data)
    save_chat_data(path, chat_data)

    return jsonify({"success": True})

@app.route('/debug_cache', methods=['GET'])
def debug_cache():
    path = get_active_chat_path()
    fn = os.path.basename(path)
    log_path = os.path.join(FILES["payload_logs_dir"], fn)
    data = read_json(log_path, {})
    prev = data.get("prev")
    last = data.get("last")

    if not prev or not last:
        return jsonify({"error": "Not enough history to compare. Make another chat turn first."})

    divergence = None

    for i in range(max(len(prev), len(last))):
        if i >= len(prev):
            break
        if i >= len(last):
            divergence = {"type": "deletion", "message_index": i, "reason": "Messages were removed from the context."}
            break

        m1 = prev[i]
        m2 = last[i]

        if m1.get("role") != m2.get("role"):
            divergence = {"type": "role_change", "message_index": i, "expected": m1.get("role"), "got": m2.get("role")}
            break

        c1 = json.dumps(m1.get("content", ""), sort_keys=True)
        c2 = json.dumps(m2.get("content", ""), sort_keys=True)

        if c1 != c2:
            diff_idx = 0
            for j in range(min(len(c1), len(c2))):
                if c1[j] != c2[j]:
                    diff_idx = j
                    break
            else:
                diff_idx = min(len(c1), len(c2))

            start_idx = max(0, diff_idx - 50)
            p_snip = c1[start_idx : diff_idx + 50]
            l_snip = c2[start_idx : diff_idx + 50]

            pad_len = 50 - (diff_idx - start_idx)
            p_snip = (" " * pad_len) + p_snip
            l_snip = (" " * pad_len) + l_snip

            divergence = {
                "type": "content_change",
                "message_index": i,
                "role": m1.get("role"),
                "diff_idx": diff_idx,
                "snippet_prev": p_snip,
                "snippet_last": l_snip
            }
            break

    return jsonify({
        "success": True,
        "divergence": divergence,
        "prev_len": len(prev),
        "last_len": len(last)
    })

@app.route('/create_arena', methods=['POST'])
def create_arena():
    models = request.json.get('models', [])
    if len(models) < 2: return jsonify({"error": "Need at least 2 models"}), 400
    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"Arena_{ts}.json"
    path = os.path.join(FILES["conversations_dir"], fn)
    
    sys_prompt = read_text(FILES["main_prompt"])
    eval_prompt = read_text(FILES["evaluator_prompt"])
    
    threads = {}
    for m in models:
        threads[m['id']] = [{"role": "system", "content": sys_prompt}]
        
    write_json(path, {
        "is_arena": True,
        "models": models,
        "threads": threads,
        "evaluator": [{"role": "system", "content": eval_prompt}]
    })
    
    write_json(FILES["active_meta"], {"filename": fn})
    return jsonify({"success": True, "filename": fn})

@app.route('/branch_to_arena', methods=['POST'])
def branch_to_arena():
    try:
        req = request.json
        idx = req.get('index')
        models = req.get('models', [])
        
        if idx is None or not models:
            return jsonify({"error": "Missing index or models"}), 400
            
        path = get_active_chat_path()
        data = load_chat_data(path)

        sliced_msgs = data["messages"][:idx + 1]
        sliced_sums = [s for s in data.get("summaries", []) if s["end_index"] <= idx]
        vis_mem = data.get("visual_memory", "")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        old_name = os.path.basename(path).replace('.json', '')
        new_fn = f"Arena_Branch_{old_name}_{ts}.json"
        new_path = os.path.join(FILES["conversations_dir"], new_fn)

        eval_prompt = read_text(FILES["evaluator_prompt"])
        
        threads = {}
        for m in models:
            # Deep copy history into each model thread
            threads[m['id']] = json.loads(json.dumps(sliced_msgs))

        write_json(new_path, {
            "is_arena": True,
            "models": models,
            "threads": threads,
            "summaries": sliced_sums, # Persist summaries for context building
            "visual_memory": vis_mem,
            "evaluator": [{"role": "system", "content": eval_prompt}],
            "parent_file": os.path.basename(path)
        })

        write_json(FILES["active_meta"], {"filename": new_fn})

        return jsonify({"success": True, "filename": new_fn})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/arena_chat', methods=['POST'])
def arena_chat():
    data = request.json
    path = get_active_chat_path()
    chat_data = read_json(path, {})
    if not chat_data.get('is_arena'): return jsonify({"error": "Not an arena chat"}), 400
    
    msg = data.get('message')
    target = data.get('target', 'all')
    
    models_to_run = [m['id'] for m in chat_data['models']] if target == 'all' else [target]
    send_time = datetime.datetime.now().isoformat()
    
    for m_id in models_to_run:
        chat_data['threads'][m_id].append({"role": "user", "content": msg, "timestamp": send_time})
        chat_data['threads'][m_id].append({"role": "assistant", "content": "", "model": m_id})
        
    write_json(path, chat_data)
    
    q = queue.Queue()
    
    def fetch_model(m_id):
        # Correctly build context for Arena model based on its specific thread
        # This allows summaries and visual memory to be used even in Arena
        temp_thread_data = {
            "messages": chat_data['threads'][m_id][:-1],
            "summaries": chat_data.get("summaries", []),
            "visual_memory": chat_data.get("visual_memory", "")
        }
        
        # Determine query text for RAG etc
        query_text = msg
        if isinstance(query_text, list):
            query_text = get_text_content({"content": query_text})

        ctx = build_context(temp_thread_data, user_query=query_text, current_model=m_id)
        
        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        payload = {
            "model": m_id,
            "messages": apply_claude_caching(ctx, m_id),
            "stream": True,
            "venice_parameters": {"include_venice_system_prompt": False}
        }
        try:
            with requests.post(VENICE_URL, headers=headers, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if "[DONE]" in decoded: break
                        try:
                            chunk = json.loads(decoded[6:])
                            if "usage" in chunk:
                                q.put((m_id, {"usage": chunk["usage"]}))
                            if len(chunk['choices'])>0:
                                delta = chunk['choices'][0]['delta']
                                res_chunk = {}
                                if 'content' in delta and delta['content']:
                                    res_chunk['content'] = delta['content']
                                if 'reasoning_content' in delta and delta['reasoning_content']:
                                    res_chunk['reasoning'] = delta['reasoning_content']
                                if res_chunk:
                                    q.put((m_id, res_chunk))
                        except: pass
        except Exception as e:
            q.put((m_id, {"error": str(e)}))
        q.put((m_id, "[DONE]"))
        
    for m_id in models_to_run:
        threading.Thread(target=fetch_model, args=(m_id,)).start()
        
    def generate():
        done_count = 0
        contents = {m_id: "" for m_id in models_to_run}
        reasonings = {m_id: "" for m_id in models_to_run}
        usages = {m_id: None for m_id in models_to_run}
        while done_count < len(models_to_run):
            m_id, d = q.get()
            if d == "[DONE]": done_count += 1
            else:
                if "content" in d: 
                    contents[m_id] += d["content"]
                if "reasoning" in d:
                    reasonings[m_id] += d["reasoning"]
                if "usage" in d:
                    usages[m_id] = d["usage"]
                yield f"data: {json.dumps({m_id: d})}\n\n"
                
        for m_id in models_to_run:
            chat_data['threads'][m_id][-1]["content"] = contents[m_id]
            chat_data['threads'][m_id][-1]["reasoning"] = reasonings[m_id]
            if usages[m_id]:
                chat_data['threads'][m_id][-1]["usage"] = usages[m_id]
        write_json(path, chat_data)
        
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/arena_eval', methods=['POST'])
def arena_eval():
    data = request.json
    path = get_active_chat_path()
    chat_data = read_json(path, {})
    
    msg = data.get('message')
    hide_names = data.get('hide_names', True)
    eval_model = data.get('model')
    
    if not eval_model:
        v_set = read_json(FILES["venice_settings"], {})
        eval_model = v_set.get("model", "venice-uncensored")

    # Construct the context from all comparison threads
    context_report = "--- ARENA COMPARISON CONTEXT (START) ---\n\n"
    for m_info in chat_data['models']:
        m_id = m_info['id']
        m_name = m_info['name']
        display_name = "Model [REDACTED]" if hide_names else f"Model: {m_name} ({m_id})"
        
        context_report += f"=== CONVERSATION THREAD FOR {display_name} ===\n"
        thread = chat_data['threads'].get(m_id, [])
        # Filter out system messages for the report
        roleplay_thread = [m for m in thread if m['role'] != 'system']
        
        for i, m in enumerate(roleplay_thread):
            is_latest = (i == len(roleplay_thread) - 1)
            prefix = "Latest " if is_latest else ""
            
            if m['role'] == 'user':
                label = f"{prefix}User Prompt"
            else:
                label = f"{prefix}Comparison Model Response"
            
            content = get_text_content(m)
            context_report += f"[{label}]: {content}\n\n"
        context_report += f"=== END THREAD FOR {display_name} ===\n\n"
    
    context_report += "--- ARENA COMPARISON CONTEXT (END) ---\n\n"

    # Now label the evaluator's own previous thoughts clearly
    eval_history_str = ""
    eval_roleplay = [m for m in chat_data['evaluator'] if m['role'] != 'system']
    # We don't include the message we just added (the last two) in this specific history block
    for i, m in enumerate(eval_roleplay[:-2]):
        is_latest = (i == len(eval_roleplay) - 3) # -2 is the new user msg, -3 is the previous assistant msg
        prefix = "Latest " if is_latest else ""
        
        if m['role'] == 'user':
            label = f"{prefix}User's Evaluator Query"
        else:
            label = f"{prefix}Evaluator Previous Response"
            
        eval_history_str += f"[{label}]: {get_text_content(m)}\n\n"

    user_prompt_str = f"{context_report}\n--- PREVIOUS EVALUATOR HISTORY ---\n{eval_history_str}\n\nUSER'S CURRENT EVALUATION REQUEST:\n{msg}"
    
    # 1. Persist the user message and a placeholder assistant message before streaming
    send_time = datetime.datetime.now().isoformat()
    chat_data['evaluator'].append({"role": "user", "content": msg, "timestamp": send_time})
    chat_data['evaluator'].append({"role": "assistant", "content": "", "model": eval_model, "timestamp": datetime.datetime.now().isoformat()})
    write_json(path, chat_data)

    def generate(final_prompt):
        # We use a copy of the evaluator thread up to the user message we just added
        # to construct the API context. The actual chat_data['evaluator'] remains 
        # as the permanent record.
        api_msgs = chat_data['evaluator'][:-1].copy()
        # Replace the last user message content with the enriched context report
        api_msgs[-1] = {"role": "user", "content": final_prompt}
        
        headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        payload = {
            "model": eval_model,
            "messages": apply_claude_caching(api_msgs, eval_model),
            "stream": True,
            "venice_parameters": {"include_venice_system_prompt": False}
        }
        
        full = ""
        reasoning = ""
        usage = None
        try:
            with requests.post(VENICE_URL, headers=headers, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if "[DONE]" in decoded: break
                        try:
                            chunk = json.loads(decoded[6:])
                            if "usage" in chunk:
                                usage = chunk["usage"]
                                yield f"data: {json.dumps({'usage': chunk['usage']})}\n\n"
                            if len(chunk['choices'])>0:
                                delta = chunk['choices'][0]['delta']
                                res_chunk = {}
                                if 'content' in delta and delta['content']:
                                    c = delta['content']
                                    full += c
                                    res_chunk['content'] = c
                                if 'reasoning_content' in delta and delta['reasoning_content']:
                                    r_part = delta['reasoning_content']
                                    reasoning += r_part
                                    res_chunk['reasoning'] = r_part
                                
                                if res_chunk:
                                    yield f"data: {json.dumps(res_chunk)}\n\n"
                        except: pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
        chat_data['evaluator'][-1]["content"] = full
        chat_data['evaluator'][-1]["reasoning"] = reasoning
        if usage:
            chat_data['evaluator'][-1]["usage"] = usage
        write_json(path, chat_data)
        
    return Response(stream_with_context(generate(user_prompt_str)), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)