import os
import re

def apply_updates():
    if not os.path.exists('input.txt'):
        print("Error: input.txt not found.")
        return

    with open('input.txt', 'r') as f:
        content = f.read()

    parts = re.split(r'---FILE:(.*?)---', content)

    for i in range(1, len(parts), 2):
        filepath = parts[i].strip()
        code = parts[i+1].strip()

        # This part extracts the folder path (e.g., 'static/')
        folder = os.path.dirname(filepath)

        # If there is a folder in the path and it doesn't exist, create it!
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

        with open(filepath, 'w') as target_file:
            target_file.write(code)
            print(f"Updated: {filepath}")

if __name__ == "__main__":
    apply_updates()