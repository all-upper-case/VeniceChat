import os
import requests
import json

MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
BASE_URL = 'https://api.mistral.ai/v1/audio/voices'

def update_voice_metadata():
    print("=== Mistral Voxtral: Update Voice Metadata ===")

    # 1. Get the Target Voice ID
    voice_id = input("Enter the Voice ID (UUID) you want to update: ").strip()
    if not voice_id:
        print("Error: Voice ID is required.")
        return

    print("\n--- Leave blank to keep existing value ---")

    # 2. Collect New Metadata
    name = input("New Name: ").strip() or None
    gender = input("New Gender (male/female/neutral): ").strip().lower() or None

    age_in = input("New Age (integer): ").strip()
    age = int(age_in) if age_in.isdigit() else None

    langs_in = input("New Languages (comma separated, e.g. en,fr): ").strip()
    languages = [l.strip() for l in langs_in.split(',')] if langs_in else None

    tags_in = input("New Tags (comma separated, e.g. soft, narrator): ").strip()
    tags = [t.strip() for t in tags_in.split(',')] if tags_in else None

    # 3. Construct Payload (Only including non-null fields)
    payload = {}
    if name: payload["name"] = name
    if gender: payload["gender"] = gender
    if age is not None: payload["age"] = age
    if languages: payload["languages"] = languages
    if tags: payload["tags"] = tags

    if not payload:
        print("No changes provided. Exiting.")
        return

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    # 4. Send PATCH request to the specific voice ID
    print(f"\nUpdating voice {voice_id}...")
    try:
        url = f"{BASE_URL}/{voice_id}"
        # Note: Mistral uses PATCH for partial updates
        r = requests.patch(url, headers=headers, json=payload)

        if r.status_code == 200:
            updated_data = r.json()
            print(f"✅ SUCCESS! Voice metadata updated.")
            print(json.dumps(updated_data, indent=2))

            # 5. Sync with local JSON file
            sync_local_record(updated_data)
        else:
            print(f"❌ Error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

def sync_local_record(updated_voice):
    """Updates the entry in my_mistral_voices.json if it exists."""
    filename = "my_mistral_voices.json"
    if not os.path.exists(filename):
        return

    try:
        with open(filename, "r") as f:
            history = json.load(f)

        # Find and replace the specific voice record
        updated_history = []
        found = False
        for v in history:
            if v.get('id') == updated_voice.get('id'):
                updated_history.append(updated_voice)
                found = True
            else:
                updated_history.append(v)

        if found:
            with open(filename, "w") as f:
                json.dump(updated_history, f, indent=4)
            print(f"Local record '{filename}' has been synchronized.")
    except Exception as e:
        print(f"Warning: Could not sync local JSON file: {e}")

if __name__ == "__main__":
    if not MISTRAL_API_KEY:
        print("ERROR: MISTRAL_API_KEY environment variable not set.")
    else:
        update_voice_metadata()