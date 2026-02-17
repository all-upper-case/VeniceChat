# Last Updated: 2026-02-17 13:00:00
import os
import json
import time
import datetime
import requests
from flask import Flask, request, jsonify, render_template, Response, stream_with_context

app = Flask(__name__)

# --- CONFIG ---
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
FAL_KEY = os.environ.get('FAL_KEY')
MISTRAL_URL = 'https://api.mistral.ai/v1/chat/completions'
FAL_URL = "https://fal.run/fal-ai/z-image/turbo"

FILES = {
    "active_meta": 'active_chat_meta.json',
    "mistral_settings": 'mistral_settings.json',
    "mistral_img_settings": 'mistral_img_settings.json',
    "img_settings": 'image_settings.json',
    "summarizer_settings": 'summarizer_settings.json',
    "model_history": 'model_history.json',
    "main_prompt": 'system_prompt_main.txt',
    "img_prompt_instr": 'system_prompt_imgprompt.txt',
    "visual_prompt": 'system_prompt_visual.txt',
    "conversations_dir": 'conversations'
}

os.makedirs(FILES["conversations_dir"], exist_ok=True)

# --- UTILS ---
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

# --- CHAT DATA HELPERS ---
def load_chat_data(path):
    raw = read_json(path, [])
    if isinstance(raw, list): return {"messages": raw, "summaries": [], "visual_memory": ""}
    if "summaries" not in raw: raw["summaries"] = []
    if "visual_memory" not in raw: raw["visual_memory"] = ""
    return raw

def save_chat_data(path, data):
    write_json(path, data)
    os.utime(path, None)

def get_active_chat_path():
    meta = read_json(FILES["active_meta"], {"filename": None})
    fn = meta.get("filename")
    if not fn or not os.path.exists(os.path.join(FILES["conversations_dir"], fn)):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"New_Chat_{ts}.json"
        write_json(FILES["active_meta"], {"filename": fn})
        write_json(os.path.join(FILES["conversations_dir"], fn), {
            "messages": [{"role": "system", "content": read_text(FILES["main_prompt"])}],
            "summaries": [],
            "visual_memory": ""
        })
    return os.path.join(FILES["conversations_dir"], fn)

def clean_for_api(messages):
    cleaned = []
    for m in messages:
        c = str(m.get('content', ''))
        if c.startswith('__IMG_JSON__'): continue
        cleaned.append({"role": m.get("role"), "content": c})
    return cleaned

# --- SUMMARIZATION ENGINE ---
def process_summaries(chat_data):
    s_set = read_json(FILES["summarizer_settings"], {"enabled": False})
    if not s_set.get("enabled", False): return chat_data

    msgs = chat_data["messages"]
    sums = chat_data["summaries"]

    threshold = int(s_set.get("trigger_threshold_turns", 10))
    keep = int(s_set.get("recent_turns_to_keep", 4))
    batch_size = int(s_set.get("batch_size", 4))

    while True:
        last_sum_idx = sums[-1]["end_index"] if sums else 0

        # Identify all text messages (non-system, non-image)
        valid_indices = [i for i, m in enumerate(msgs) if i > 0 and not str(m.get('content', '')).startswith('__IMG_JSON__')]

        # Find valid messages that haven't been summarized yet
        available_valid = [i for i in valid_indices if i > last_sum_idx]

        # How many valid messages are we forced to keep in raw context?
        # We can only summarize if we have more than 'keep' valid messages total
        if len(available_valid) <= keep:
            break

        # We can summarize everything except the 'keep' most recent valid messages
        summarizable_valid = available_valid[:-keep]

        # Check if we have at least 'batch_size' to process, or if we've crossed the 'threshold'
        # Note: threshold is checked against 'available_valid' count
        if len(available_valid) >= threshold and len(summarizable_valid) >= batch_size:
            # Take exactly 'batch_size' from the start of summarizable_valid
            batch_indices = summarizable_valid[:batch_size]
            start_idx = batch_indices[0]
            actual_end = batch_indices[-1]

            # The range [start_idx : actual_end + 1] might include images, 
            # but clean_for_api will strip them out of the summary content.
            batch = msgs[start_idx : actual_end + 1]
            batch_cleaned = clean_for_api(batch)
            text_to_sum = "\n".join([f"{m['role'].upper()}: {m['content'][:1000]}" for m in batch_cleaned])

            # Provide the last 3 summaries as context to help the model maintain continuity
            context_summaries = sums[-3:]
            context_str = "\n\n".join([f"PREVIOUS SUMMARY BLOCK:\n{s['content']}" for s in context_summaries])

            user_content = ""
            if context_str:
                user_content += f"--- CONTEXT: PREVIOUS SUMMARIES (Do not re-summarize these) ---\n{context_str}\n\n"
            user_content += f"--- NEW MESSAGES TO SUMMARIZE (Summarize the following only) ---\n{text_to_sum}"

            try:
                h = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
                p = {
                    "model": s_set.get("model", "mistral-small-latest"),
                    "temperature": 0.3,
                    "messages": [
                        {"role": "system", "content": s_set.get("system_prompt", "Summarize.")},
                        {"role": "user", "content": user_content}
                    ],
                    "safe_prompt": False
                }
                resp = requests.post(MISTRAL_URL, headers=h, json=p).json()
                summary_text = resp['choices'][0]['message']['content']

                sums.append({
                    "start_index": start_idx,
                    "end_index": actual_end,
                    "content": summary_text
                })
                print(f"DEBUG: Batch summarized indices {start_idx} to {actual_end} (Contains {batch_size} text msgs)")
            except Exception as e:
                print(f"Summarizer failed: {e}")
                break
        else:
            break

    return chat_data

def build_context(chat_data):
    s_set = read_json(FILES["summarizer_settings"], {"enabled": False})
    msgs = chat_data["messages"]

    if not s_set.get("enabled", False):
        return clean_for_api(msgs)

    sums = chat_data["summaries"]
    context = [msgs[0]] # System
    last_covered = 0

    # Inject Summaries
    for s in sums:
        prefix = s_set.get("assistant_prefix", "SUMMARY OF PREVIOUS TURNS:")
        context.append({"role": "system", "content": f"{prefix}\n{s['content']}"})
        last_covered = s["end_index"]

    # Append Unsummarized (Recent) messages
    # Any images interleaved between text messages that were summarized 
    # are naturally skipped here because of last_covered.
    raw_remainder = msgs[last_covered + 1:]
    context.extend(clean_for_api(raw_remainder))

    return context

# --- ROUTES ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/get_history', methods=['GET'])
def get_history():
    path = get_active_chat_path()
    data = load_chat_data(path)
    # Sync system prompt
    if len(data["messages"]) > 0 and data["messages"][0]["role"] == "system":
        data["messages"][0]["content"] = read_text(FILES["main_prompt"])
    return jsonify({"history": data["messages"], "summaries": data["summaries"], "visual_memory": data["visual_memory"]})

@app.route('/check_summary_status', methods=['POST'])
def check_summary_status():
    """Calculates how many batches would be processed if enabled now."""
    path = get_active_chat_path()
    data = load_chat_data(path)
    msgs = data["messages"]
    sums = data["summaries"]

    req_set = request.json.get('settings', {})
    keep = int(req_set.get("recent_turns_to_keep", 4))
    batch_size = int(req_set.get("batch_size", 4))

    last_idx = sums[-1]["end_index"] if sums else 0

    valid_indices = [i for i, m in enumerate(msgs) if i > 0 and not str(m.get('content', '')).startswith('__IMG_JSON__')]
    available_valid = [i for i in valid_indices if i > last_idx]

    summarizable_count = len(available_valid) - keep
    if summarizable_count < 0: summarizable_count = 0

    batches_needed = summarizable_count // batch_size

    return jsonify({
        "total_messages": len(msgs),
        "unsummarized_text_messages": len(available_valid),
        "batches_pending": batches_needed
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    path = get_active_chat_path()
    chat_data = load_chat_data(path)

    if len(chat_data["messages"]) == 1:
        clean = "".join([c for c in data['message'][:25] if c.isalnum() or c==' ']).strip().replace(" ", "_")
        ts = datetime.datetime.now().strftime("%m%d_%H%M")
        new_fn = f"{clean}_{ts}.json"
        new_path = os.path.join(FILES["conversations_dir"], new_fn)
        os.rename(path, new_path)
        write_json(FILES["active_meta"], {"filename": new_fn})
        path = new_path

    chat_data["messages"].append({"role": "user", "content": data['message']})
    chat_data["messages"].append({"role": "assistant", "content": ""})
    save_chat_data(path, chat_data)

    idx = len(chat_data["messages"]) - 1

    # Run Summarizer (Catches up if needed)
    chat_data = process_summaries(chat_data)
    save_chat_data(path, chat_data)

    # Build Context (excluding the empty placeholder we just added)
    temp_data = {"messages": chat_data["messages"][:-1], "summaries": chat_data["summaries"]}
    context = build_context(temp_data)

    def generate():
        m_set = read_json(FILES["mistral_settings"], {})
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
        payload = {
            "model": m_set.get("model", "mistral-large-latest"),
            "temperature": float(m_set.get("temperature", 0.7)),
            "max_tokens": int(m_set.get("max_tokens", 4000)),
            "presence_penalty": float(m_set.get("presence_penalty", 0.0)),
            "frequency_penalty": float(m_set.get("frequency_penalty", 0.0)),
            "messages": context, "stream": True, "safe_prompt": False
        }

        full = ""
        usage = None
        try:
            with requests.post(MISTRAL_URL, headers=headers, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if "[DONE]" in decoded: break
                        try:
                            chunk = json.loads(decoded[6:])
                            if "usage" in chunk: usage = chunk["usage"]
                            if len(chunk['choices'])>0:
                                c = chunk['choices'][0]['delta'].get('content', '')
                                full += c
                                if len(c)>0:
                                    chat_data["messages"][idx]["content"] = full
                                    save_chat_data(path, chat_data)
                                yield f"data: {json.dumps({'content': c})}\n\n"
                        except: pass

            if usage:
                chat_data["messages"][idx]["usage"] = usage
                save_chat_data(path, chat_data)
                yield f"data: {json.dumps({'usage': usage})}\n\n"

        except Exception as e:
            chat_data["messages"].pop()
            save_chat_data(path, chat_data)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/generate_image', methods=['POST'])
def generate_image():
    path = get_active_chat_path()
    chat_data = load_chat_data(path)

    text_msgs = clean_for_api(chat_data["messages"])
    valid = [m for m in text_msgs if m['role'] != 'system']
    context_msgs = valid[-3:]

    context_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in context_msgs])
    if request.json.get('guidance'): context_str += f"\nINSTRUCTION: {request.json.get('guidance')}"

    vis_mem = chat_data.get("visual_memory", "")
    if vis_mem: context_str = f"CHARACTER VISUAL DESCRIPTIONS:\n{vis_mem}\n\n" + context_str

    try:
        p_set = read_json(FILES["mistral_img_settings"], {})
        h = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
        print(f"DEBUG: Generating image prompt for context: {context_str[:100]}...")

        p_res = requests.post(MISTRAL_URL, headers=h, json={
            "model": p_set.get("model", "mistral-medium-latest"),
            "messages": [{"role":"system","content":read_text(FILES["img_prompt_instr"])},{"role":"user","content":context_str}],
            "safe_prompt": False
        }, timeout=45).json()

        prompt = p_res['choices'][0]['message']['content'].strip()
        print(f"DEBUG: Optimized Prompt: {prompt}")

        i_set = read_json(FILES["img_settings"], {})
        fh = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}
        f_res = requests.post(FAL_URL, headers=fh, json={
            "prompt": prompt, "num_inference_steps": 8, "enable_safety_checker": False,
            "image_size": {"width": int(i_set.get("width", 1024)), "height": int(i_set.get("height", 1024))}
        }, timeout=60).json()

        if 'images' not in f_res:
            raise Exception(f"FAL Error: {f_res.get('detail', 'Unknown error')}")

        url = f_res['images'][0]['url']
        chat_data["messages"].append({"role": "assistant", "content": f"__IMG_JSON__{json.dumps({'url': url, 'prompt': prompt})}"})
        save_chat_data(path, chat_data)
        return jsonify({"success": True})
    except Exception as e:
        print(f"ERROR in generate_image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/scan_visuals', methods=['POST'])
def scan_visuals():
    depth = int(request.json.get('depth', 50))
    path = get_active_chat_path()
    chat_data = load_chat_data(path)

    msgs = clean_for_api(chat_data["messages"])
    target_msgs = msgs[-depth:]
    blob = "\n".join([f"{m['role']}: {m['content']}" for m in target_msgs])

    existing = chat_data.get("visual_memory", "")
    prompt = f"EXISTING VISUAL MEMORY:\n{existing}\n\nRECENT CHAT LOG:\n{blob}"

    try:
        h = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
        res = requests.post(MISTRAL_URL, headers=h, json={
            "model": "mistral-large-latest",
            "messages": [{"role":"system","content":read_text(FILES["visual_prompt"])}, {"role":"user","content":prompt}],
            "safe_prompt": False
        }).json()
        new_mem = res['choices'][0]['message']['content']
        chat_data["visual_memory"] = new_mem
        save_chat_data(path, chat_data)
        return jsonify({"success": True, "memory": new_mem})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/update_history', methods=['POST'])
def update_history():
    path = get_active_chat_path()
    data = load_chat_data(path)
    data["messages"] = request.json.get('history')
    # Invalidate summaries if history truncated
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
    return jsonify({"chats": [f["name"] for f in files]})

@app.route('/rename_chat', methods=['POST'])
def rename_chat():
    old, new = request.json.get('old'), request.json.get('new')
    os.rename(os.path.join(FILES["conversations_dir"], old), os.path.join(FILES["conversations_dir"], new+".json"))
    meta = read_json(FILES["active_meta"], {})
    if meta.get("filename") == old: write_json(FILES["active_meta"], {"filename": new+".json"})
    return jsonify({"success": True})

@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    fn = request.json.get('filename')
    os.remove(os.path.join(FILES["conversations_dir"], fn))
    return jsonify({"success": True})

@app.route('/get_settings')
def get_settings():
    return jsonify({
        "main_prompt": read_text(FILES["main_prompt"]),
        "mistral": read_json(FILES["mistral_settings"], {}),
        "mistral_img": read_json(FILES["mistral_img_settings"], {}),
        "fal": read_json(FILES["img_settings"], {}),
        "summarizer": read_json(FILES["summarizer_settings"], {}),
        "model_history": read_json(FILES["model_history"], [])
    })

@app.route('/save_settings', methods=['POST'])
def save_settings():
    d = request.json
    if "main_prompt" in d: write_text(FILES["main_prompt"], d["main_prompt"])
    if "mistral" in d: write_json(FILES["mistral_settings"], d["mistral"])
    if "summarizer" in d: write_json(FILES["summarizer_settings"], d["summarizer"])
    if "fal" in d:
        f = d["fal"]; f["num_inference_steps"] = 8; f["enable_safety_checker"] = False
        write_json(FILES["img_settings"], f)

    if "mistral" in d and "model" in d["mistral"]:
        h = read_json(FILES["model_history"], [])
        if d["mistral"]["model"] not in h:
            h.insert(0, d["mistral"]["model"])
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# Last Updated: 2026-02-17 13:00:00