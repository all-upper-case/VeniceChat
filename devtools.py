import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from flask import Blueprint, jsonify, request

devtools_bp = Blueprint("devtools", __name__)

ROOT = Path(__file__).resolve().parent
ALLOWED_SCRIPTS = {"fetch.py", "edit.py"}


def _token_ok() -> bool:
    expected = os.environ.get("DEVTOOLS_TOKEN", "").strip()
    supplied = (
        request.args.get("token")
        or request.headers.get("X-Devtools-Token")
        or (request.json or {}).get("token") if request.is_json else None
    )
    return bool(expected) and supplied == expected


def _run(args, input_text=None, timeout=30):
    proc = subprocess.run(
        args,
        cwd=ROOT,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "combined": (proc.stdout or "") + (("\n--- STDERR ---\n" + proc.stderr) if proc.stderr else ""),
    }


@devtools_bp.route("/devtools", methods=["GET"])
def devtools_page():
    token = request.args.get("token", "")
    if not _token_ok():
        return "Forbidden or DEVTOOLS_TOKEN is not set.", 403

    return f"""<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dev Tools</title>
<style>
body {{
    font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    background: #111;
    color: #eee;
    margin: 0;
    padding: 16px;
}}
textarea, input {{
    width: 100%;
    box-sizing: border-box;
    background: #1f1f1f;
    color: #eee;
    border: 1px solid #444;
    border-radius: 10px;
    padding: 12px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 14px;
}}
textarea {{
    min-height: 210px;
}}
button {{
    width: 100%;
    margin-top: 10px;
    padding: 14px;
    border: 0;
    border-radius: 12px;
    background: #2563eb;
    color: white;
    font-weight: 700;
    font-size: 16px;
}}
button.secondary {{
    background: #444;
}}
button.danger {{
    background: #dc2626;
}}
pre {{
    white-space: pre-wrap;
    word-break: break-word;
    background: #050505;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 12px;
    min-height: 160px;
}}
.small {{
    color: #aaa;
    font-size: 13px;
    line-height: 1.4;
}}
</style>
</head>
<body>
<h2>Dev Tools</h2>
<p class="small">
Paste either a fetch command like <code>python fetch.py main.py --contains "create_pipeline_chat" --context 80</code>,
or paste a full <code>---PATCH---</code> block.
</p>

<textarea id="input" placeholder="Paste fetch command or patch block here..."></textarea>

<button onclick="runFetch()">Run Fetch Command</button>
<button class="secondary" onclick="dryRunPatch()">Dry Run Patch</button>
<button class="danger" onclick="applyPatch()">Apply Patch</button>

<h3>Output</h3>
<button class="secondary" onclick="copyOutput()">Copy Output</button>
<pre id="output"></pre>

<script>
const TOKEN = {token!r};

async function postJson(url, body) {{
    body.token = TOKEN;
    const res = await fetch(url, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(body)
    }});
    const data = await res.json().catch(() => ({{ combined: "Could not parse JSON response." }}));
    document.getElementById("output").textContent =
        data.combined || JSON.stringify(data, null, 2);
}}

function getInput() {{
    return document.getElementById("input").value;
}}

async function runFetch() {{
    await postJson("/devtools/run", {{ command: getInput() }});
}}

async function dryRunPatch() {{
    await postJson("/devtools/patch", {{ patch_text: getInput(), dry_run: true }});
}}

async function applyPatch() {{
    if (!confirm("Apply this patch to the codebase?")) return;
    await postJson("/devtools/patch", {{ patch_text: getInput(), dry_run: false }});
}}

async function copyOutput() {{
    const text = document.getElementById("output").textContent;
    await navigator.clipboard.writeText(text);
    alert("Copied output.");
}}
</script>
</body>
</html>"""


@devtools_bp.route("/devtools/run", methods=["POST"])
def devtools_run():
    if not _token_ok():
        return jsonify({"combined": "Forbidden or DEVTOOLS_TOKEN is not set."}), 403

    command = (request.json or {}).get("command", "").strip()
    if not command:
        return jsonify({"combined": "No command provided."}), 400

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return jsonify({"combined": f"Could not parse command: {exc}"}), 400

    if len(parts) < 2:
        return jsonify({"combined": "Command must look like: python fetch.py ..."}), 400

    if parts[0] not in {"python", "python3"}:
        return jsonify({"combined": "Only python/python3 commands are allowed."}), 400

    script = Path(parts[1]).name
    if script not in ALLOWED_SCRIPTS:
        return jsonify({"combined": "Only fetch.py and edit.py are allowed."}), 400

    # Do not allow edit.py from this generic command route.
    # Patches should go through /devtools/patch so they get temp-file handling.
    if script == "edit.py":
        return jsonify({"combined": "Use the Dry Run Patch or Apply Patch button for edit.py patches."}), 400

    result = _run(parts, timeout=30)
    return jsonify(result)


@devtools_bp.route("/devtools/patch", methods=["POST"])
def devtools_patch():
    if not _token_ok():
        return jsonify({"combined": "Forbidden or DEVTOOLS_TOKEN is not set."}), 403

    data = request.json or {}
    patch_text = data.get("patch_text", "")
    dry_run = bool(data.get("dry_run", True))

    if "---PATCH---" not in patch_text:
        return jsonify({"combined": "No ---PATCH--- block found."}), 400

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        prefix="devtools_patch_",
        delete=False,
        dir=ROOT,
    ) as tmp:
        tmp.write(patch_text)
        tmp_path = Path(tmp.name)

    try:
        args = ["python", "edit.py", str(tmp_path.name)]
        if dry_run:
            args.append("--dry-run")
        result = _run(args, timeout=60)
        return jsonify(result)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass