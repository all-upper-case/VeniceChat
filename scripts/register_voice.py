import os
import requests
import base64
import json
from pathlib import Path

MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
VOICES_URL = 'https://api.mistral.ai/v1/audio/voices'

def interactive_register():
    print("=== Mistral Voxtral: Permanent Voice Registration ===")

    # 1. File Handling
    filename = input("Enter the audio sample filename (e.g., sample.wav): ").strip()
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    # 2. Metadata Gathering
    name = input("Voice Name (required): ").strip()
    if not name:
        print("Error: Name is required.")
        return

    gender = input("Gender (e.g., male, female): ").strip().lower() or None

    age_input = input("Age (integer, e.g., 25): ").strip()
    age = int(age_input) if age_input.isdigit() else None

    langs = input("Languages (comma separated, e.g., en, fr): ").strip().split(',')
    tags = input("Tags (comma separated, e.g., warm, raspy): ").strip().split(',')

    # 3. Base64 Encoding
    print(f"Encoding {filename}...")
    sample_b64 = base64.b64encode(Path(filename).read_bytes()).decode()

    # 4. API Request Construction
    payload = {
        "name": name,
        "sample_audio": sample_b64,
        "sample_filename": filename,
        "languages": [l.strip() for l in langs if l.strip()] or None,
        "gender": gender,
        "age": age,
        "tags": [t.strip() for t in tags if t.strip()] or None
    }

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    print("Registering voice with Mistral AI...")
    try:
        r = requests.post(VOICES_URL, headers=headers, json=payload)
        if r.status_code in [200, 201]:
            voice_data = r.json()
            print(f"\n✅ SUCCESS!")
            print(f"VOICE ID: {voice_data.get('id')}")

            # Persistent Local Record
            record_file = "my_mistral_voices.json"
            history = []
            if os.path.exists(record_file):
                try:
                    with open(record_file, "r") as f:
                        history = json.load(f)
                except: pass

            history.append(voice_data)
            with open(record_file, "w") as f:
                json.dump(history, f, indent=4)
            print(f"Voice details saved to {record_file}")
        else:
            print(f"❌ API Error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    if not MISTRAL_API_KEY:
        print("ERROR: MISTRAL_API_KEY environment variable is not set.")
    else:
        interactive_register()