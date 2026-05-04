import os
import requests
import json

MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
VOICES_URL = 'https://api.mistral.ai/v1/audio/voices'

def list_voices():
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}

    print("--- Fetching Mistral Voice Library ---")
    try:
        # Mistral uses 'items' in the response, not 'data'
        r = requests.get(f"{VOICES_URL}?limit=20", headers=headers)

        if r.status_code == 200:
            resp = r.json()
            voices = resp.get('items', []) # FIX: Changed from 'data' to 'items'
            total = resp.get('total', 0)

            print(f"Total Registered Voices: {total}\n")
            print(f"{'ID':<40} | {'Name':<20} | {'Gender'}")
            print("-" * 75)

            for v in voices:
                vid = v.get('id')
                name = v.get('name', 'N/A')
                gender = v.get('gender', 'null')
                print(f"{vid:<40} | {name:<20} | {gender}")
        else:
            print(f"Error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    if not MISTRAL_API_KEY:
        print("ERROR: MISTRAL_API_KEY not set.")
    else:
        list_voices()