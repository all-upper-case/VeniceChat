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

# --- CONFIG ---
# Using Venice.ai as the primary inference provider
VENICE_API_KEY = os.environ.get('VENICE_API_KEY')
FAL_KEY = os.environ.get('FAL_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
VENICE_URL = 'https://api.venice.ai/api/v1/chat/completions'
VENICE_EMBED_URL = 'https://api.venice.ai/api/v1/embeddings'
VENICE_CHARACTERS_URL = 'https://api.venice.ai/api/v1/characters'
FAL_URL = "https://fal.run/fal-ai/z-image/turbo"

FILES = {
    "active_meta": 'active_chat_meta.json',
    "venice_settings": 'venice_settings.json',
    "venice_img_settings": 'venice_img_settings.json',
    "refiner_settings": 'refiner_settings.json',
    "img_settings": 'image_settings.json',
    "summarizer_settings": 'summarizer_settings.json',
    "wfm_settings": 'wfm_settings.json',
    "model_history": 'model_history.json',
    "main_prompt": 'system_prompt_main.txt',
    "img_prompt_instr": 'system_prompt_imgprompt.txt',
    "visual_prompt": 'system_prompt_visual.txt',
    "architect_prompt": 'system_prompt_architect.txt',
    "user_mimic_prompt": 'system_prompt_user_mimic.txt',
    "refine_prompt": 'system_prompt_refine.txt',
    "summary_note_prompt": 'system_prompt_summary_note.txt',
    "rag_note_prompt": 'system_prompt_rag_note.txt',
    "lore_extractor_prompt": 'system_prompt_lore_extractor.txt',
    "summary_consolidator_prompt": 'system_prompt_summary_consolidator.txt',
    "venice_dupe_prompt": 'system_prompt_venice_dupe.txt',
    "banned_phrases": 'banned_phrases.txt',
    "interface_settings": 'interface_settings.json',
    "conversations_dir": 'conversations',
    "uploads_dir": 'static/uploads',
    "lorebook": 'lorebook.txt',
    "lorebook_index": 'lorebook.index',
    "lorebook_chunks": 'lorebook_chunks.json',
    "rag_settings": 'rag_settings.json',
    "tts_settings": 'tts_settings.json',
    "payload_logs_dir": 'payload_logs',
    "tts_logs_dir": 'tts_logs',
    "character_cache": 'character_cache.json',
    "audio_cache_dir": 'static/audio_cache',
    "balance": 'balance.json'
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

def read_json(path, default):
    if not os.path.exists(path): return default
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except: return default

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4)

def read_text(path, default=""):
    if not os.path.exists(path): return default
    with open(path, 'r', encoding='utf-8') as f: return f.read().strip()

def write_text(path, content):
    with open(path, 'w', encoding='utf-8') as f: f.write(content)

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
    if isinstance(raw, list): return {"messages": raw, "summaries": [], "visual_memory": "", "character_slug": None}
    if "summaries" not in raw: raw["summaries"] = []
    if "visual_memory" not in raw: raw["visual_memory"] = ""
    if "character_slug" not in raw: raw["character_slug"] = None
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
            "character_slug": pending_character
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
    models_data = read_json('venice_models.json', {})
    for group in models_data.values():
        for m in group:
            if m.get('id') == model_id and m.get('vision'):
                return True
    return False

# --- SUMMARIZATION ENGINE ---
def process_summaries(chat_data):
    s_set = read_json(FILES["summarizer_settings"], {"enabled": False})
    if not s_set.get("enabled", False): return

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
                    "prompt_cache_retention": "24h",
                    "venice_parameters": {
                        "include_venice_system_prompt": False,
                        "strip_thinking_response": True
                    }
                }
                r_sum = requests.post(VENICE_URL, headers=h, json=p)
                r_sum.raise_for_status()
                resp = r_sum.json()

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
    msgs = chat_data.get("messages", [])
    vision_capable = is_vision_model(current_model) if current_model else False

    if not msgs: return []

    if not user_query:
        for m in reversed(msgs):
            if m.get("role") == "user" and not (isinstance(m.get('content', ''), str) and m.get('content', '').startswith('__IMG_JSON__')):
                user_query = get_text_content(m)
                break

    context = [msgs[0]] # System

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

    # --- STABLE HISTORY BLOCK (Summaries & Raw History) ---
    # We process history first so the large prefix remains identical across turns
    if s_set.get("enabled", False):
        sums = chat_data.get("summaries", [])
        active_sums = [s for s in sums if not s.get("disabled", False)]
        active_sums.sort(key=lambda x: x["start_index"])

        if active_sums:
            context.append({"role": "system", "content": read_text(FILES["summary_note_prompt"])})

        msg_idx = 1
        sum_idx = 0

        v_set = read_json(FILES["venice_settings"], {})
        vision_detail = "high" if v_set.get("vision_high_res", True) else "low"

        while msg_idx < len(msgs):
            if sum_idx < len(active_sums) and msg_idx == active_sums[sum_idx]["start_index"]:
                s = active_sums[sum_idx]
                prefix = "CONSOLIDATED ARCHIVE (Distant Past Context):" if s.get("is_consolidated") else "RECENT SUMMARY (Immediate Past Context):"
                clean_sum = re.sub(r'<think>.*?</think>', '', s['content'], flags=re.DOTALL).strip()
                context.append({"role": "system", "content": f"--- {prefix} ---\n{clean_sum}"})
                msg_idx = s["end_index"] + 1
                sum_idx += 1
            elif sum_idx < len(active_sums) and msg_idx > active_sums[sum_idx]["start_index"]:
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

# --- ROUTES ---

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
            "prompt_cache_retention": "24h",
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
            "prompt_cache_retention": "24h",
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

@app.route('/get_history', methods=['GET'])
def get_history():
    path = get_active_chat_path()
    data = load_chat_data(path)
    if len(data["messages"]) > 0 and data["messages"][0]["role"] == "system":
        data["messages"][0]["content"] = read_text(FILES["main_prompt"])

    has_backup = False
    if "backup_summaries" in data and data["backup_summaries"] is not None:
        if data["backup_summaries"] != data.get("summaries", []):
            has_backup = True

    return jsonify({
        "history": data["messages"], 
        "summaries": data["summaries"], 
        "visual_memory": data["visual_memory"], 
        "character_slug": data.get("character_slug"),
        "has_backup": has_backup
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
                    "prompt_cache_retention": "24h",
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
                        "prompt_cache_retention": "24h",
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

        sliced_msgs = data["messages"][:idx + 1]
        sliced_sums = [s for s in data.get("summaries", []) if s["end_index"] <= idx]

        new_data = {
            "messages": sliced_msgs,
            "summaries": sliced_sums,
            "visual_memory": data.get("visual_memory", "")
        }

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
        "prompt_cache_retention": "24h",
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

        user_content = f"### ORIGINAL MESSAGE:\n{text_to_refine}\n\n### EDIT INSTRUCTIONS:\n{guidance if guidance else 'Polish the prose and fix any minor errors while keeping the meaning identical.'}"

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
            "prompt_cache_retention": "24h",
            "venice_parameters": {
                "include_venice_system_prompt": False,
                "strip_thinking_response": True
            }
        }

        r = requests.post(VENICE_URL, headers=headers, json=payload)
        resp = r.json()

        if 'choices' in resp and resp['choices']:
            refined_text = resp['choices'][0]['message']['content'].strip()
            data["messages"][idx]["original_content"] = msg_content
            data["messages"][idx]["content"] = refined_text
            save_chat_data(path, data)
            return jsonify({"success": True, "refined_text": refined_text})
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
            "prompt_cache_retention": "24h",
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

        chat_data["messages"].append({"role": "user", "content": message_content})

    chat_data["messages"].append({"role": "assistant", "content": "", "model": model_to_use})
    save_chat_data(path, chat_data)

    idx = len(chat_data["messages"]) - 1

    def generate():
        nonlocal chat_data

        # 1. Process Summaries and stream status REAL-TIME
        s_set = read_json(FILES["summarizer_settings"], {"enabled": False})
        if s_set.get("enabled"):
            for status_msg in process_summaries(chat_data):
                yield f"data: {json.dumps({'status': status_msg})}\n\n"

        save_chat_data(path, chat_data)

        # Build context AFTER summarization is complete
        temp_data = {"messages": chat_data["messages"][:-1], "summaries": chat_data["summaries"], "visual_memory": chat_data.get("visual_memory", "")}
        query_text = data.get('message', '')
        if isinstance(query_text, list):
            query_text = get_text_content({"content": query_text})

        context = build_context(temp_data, user_query=query_text, current_model=model_to_use)

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

        venice_params = {
            "include_venice_system_prompt": v_set.get("include_venice_system_prompt", True),
            "strip_thinking_response": True
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
            "reasoning_effort": v_set.get("reasoning_effort", "medium"),
            "messages": apply_claude_caching(context, model_to_use), 
            "stream": True,
            "prompt_cache_key": cache_key,
            "prompt_cache_retention": "24h",
            "venice_parameters": venice_params
        }

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
        try:
            with requests.post(VENICE_URL, headers=headers, json=payload, stream=True) as r:
                r.raise_for_status()
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
                                # ... standard yield logic ...
                                if 'reasoning_content' in delta and delta['reasoning_content']:
                                    r_part = delta['reasoning_content']
                                    reasoning += r_part
                                    chat_data["messages"][idx]["reasoning"] = reasoning
                                    yield f"data: {json.dumps({'reasoning': r_part})}\n\n"

                                if 'content' in delta and delta['content']:
                                    c = delta['content']
                                    full += c
                                    chat_data["messages"][idx]["content"] = full
                                    yield f"data: {json.dumps({'content': c, 'balance': balance})}\n\n"

                                save_chat_data(path, chat_data)
                        except: pass

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

                chat_data["messages"][idx]["usage"] = usage
                save_chat_data(path, chat_data)
                yield f"data: {json.dumps({'usage': usage, 'balance': balance})}\n\n"

        except Exception as e:
            if not full and not reasoning:
                chat_data["messages"].pop()
            else:
                chat_data["messages"][idx]["content"] = full + f"\n\n*(Stream error: {str(e)})*"
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

@app.route('/scan_visuals', methods=['POST'])
def scan_visuals():
    try:
        depth = int(request.json.get('depth', 50))
        path = get_active_chat_path()
        chat_data = load_chat_data(path)

        msgs = clean_for_api(chat_data["messages"])
        target_msgs = msgs[-depth:]
        blob = "\n".join([f"{m['role']}: {get_text_content(m)}" for m in target_msgs])

        existing = chat_data.get("visual_memory", "")
        prompt = f"EXISTING VISUAL MEMORY:\n{existing}\n\nRECENT CHAT LOG:\n{blob}"

        # Use the model configured for image prompting, falling back to venice-uncensored
        p_set = read_json(FILES["venice_img_settings"], {})
        model_to_use = p_set.get("model", "venice-uncensored")

        h = {"Authorization": f"Bearer {VENICE_API_KEY}"}
        cache_key = get_cache_key(path, "_scan")

        payload = {
            "model": model_to_use,
            "messages": apply_claude_caching([
                {"role": "system", "content": read_text(FILES["visual_prompt"])}, 
                {"role": "user", "content": prompt}
            ], model_to_use),
            "prompt_cache_key": cache_key,
            "prompt_cache_retention": "24h",
            "venice_parameters": {
                "include_venice_system_prompt": False,
                "strip_thinking_response": True
            }
        }

        r = requests.post(VENICE_URL, headers=h, json=payload)
        update_last_io(VENICE_URL, payload, r.text)
        r.raise_for_status()

        res = r.json()
        if 'choices' in res and res['choices']:
            new_mem = res['choices'][0]['message']['content'].strip()
            chat_data["visual_memory"] = new_mem
            save_chat_data(path, chat_data)
            return jsonify({"success": True, "memory": new_mem})
        else:
            return jsonify({"error": "No content returned from API", "raw": res}), 500

    except Exception as e:
        print(f"[SCAN VISUALS ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/update_history', methods=['POST'])
def update_history():
    path = get_active_chat_path()
    data = load_chat_data(path)
    data["messages"] = request.json.get('history')
    if len(data["messages"]) < (data.get("summaries", [])[-1]["end_index"] if data.get("summaries") else 0):
        data["summaries"] = [] 
    save_chat_data(path, data)
    return jsonify({"success": True})

@app.route('/update_visual_memory', methods=['POST'])
def update_visual_memory():
    path = get_active_chat_path()
    data = load_chat_data(path)
    data["visual_memory"] = request.json.get('memory')
    save_chat_data(path, data)
    return jsonify({"success": True})

@app.route('/sidebar_data')
def sidebar_data():
    files = []
    for f in os.listdir(FILES["conversations_dir"]):
        if f.endswith(".json"):
            fp = os.path.join(FILES["conversations_dir"], f)
            files.append({"name": f, "time": os.path.getmtime(fp)})
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
    return jsonify(read_json('venice_models.json', {}))

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
                "prompt_cache_retention": "24h",
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
    return jsonify({
        "main_prompt": read_text(FILES["main_prompt"]),
        "venice": read_json(FILES["venice_settings"], {"model": "venice-uncensored", "temperature": 0.9, "max_tokens": 4000, "reasoning_effort": "medium"}),
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
        clean_text = clean_text.replace("’", "'").replace("‘", "'")
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
        "proj_tot": (char_sys + char_sum + char_raw) // 4
    })

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
        payload["prompt_cache_retention"] = "24h"
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)