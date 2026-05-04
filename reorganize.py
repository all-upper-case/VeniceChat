import os
import shutil
import glob

# Define the target directories and the files that belong in them
MAPPING = {
    "prompts": [
        "system_prompt_*.txt",
        "banned_phrases.txt",
        "zzztagprompt.txt",
        "ORIGINAL-system_prompt_main.txt"
    ],
    "settings": [
        "venice_settings.json",
        "venice_img_settings.json",
        "refiner_settings.json",
        "image_settings.json",
        "summarizer_settings.json",
        "wfm_settings.json",
        "interface_settings.json",
        "rag_settings.json",
        "tts_settings.json"
    ],
    "scripts": [
        "list_voices.py",
        "update_voice.py",
        "recover_chat.py",
        "register_voice.py",
        "apply_updates.py"
    ],
    "data": [
        "active_chat_meta.json",
        "balance.json",
        "character_cache.json",
        "model_history.json",
        "my_mistral_voices.json",
        "venice_models.json",
        "lorebook.txt",
        "lorebook.index",
        "lorebook_chunks.json"
    ]
}

def reorganize():
    files_moved = 0
    for folder, patterns in MAPPING.items():
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        for pattern in patterns:
            # Find all files matching the pattern (handles wildcards like system_prompt_*.txt)
            for filepath in glob.glob(pattern):
                if os.path.isfile(filepath):
                    dest = os.path.join(folder, os.path.basename(filepath))
                    try:
                        shutil.move(filepath, dest)
                        print(f"Moved {filepath} -> {dest}")
                        files_moved += 1
                    except Exception as e:
                        print(f"Failed to move {filepath}: {e}")
                        
    print(f"\nSUCCESS! Reorganization complete! Moved {files_moved} files.")

if __name__ == "__main__":
    reorganize()