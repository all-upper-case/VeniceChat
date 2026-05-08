#!/usr/bin/env python3
from pathlib import Path
import shutil
import datetime

FILES_TO_FIX = [
    Path("static/script.js"),
    Path("main.py"),
]

REPLACEMENTS = {
    # Common UI arrows / controls
    "â–¼": "▼",
    "â–²": "▲",
    "â–¶": "▶",
    "â¹": "⏹",
    "â—€": "←",
    "â†’": "→",
    "â†": "←",
    "â€¢": "•",
    "âœ“": "✓",
    "âœ…": "✅",
    "âŒ": "❌",
    "âŒ›": "⌛",

    # Emoji mojibake seen in your fetched results
    "ðŸ‘ï¸": "👁️",
    "ðŸ‘": "👁️",
    "ðŸ’­": "💭",
    "ðŸ“": "📝",
    "ðŸ—‘ï¸": "🗑️",
    "ðŸ› ï¸": "🛠️",
    "ðŸ”§": "🔧",
    "ðŸ”„": "🔄",
    "âž•": "➕",
    "âœï¸": "✏️",
}

backup_root = Path(".edit_backups") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_mojibake")
backup_root.mkdir(parents=True, exist_ok=True)

total_replacements = 0

for path in FILES_TO_FIX:
    if not path.exists():
        print(f"Skipping missing file: {path}")
        continue

    original = path.read_text(encoding="utf-8", errors="replace")
    fixed = original

    print(f"\nChecking {path}...")
    for bad, good in REPLACEMENTS.items():
        count = fixed.count(bad)
        if count:
            print(f"  {bad!r} -> {good!r}: {count}")
            fixed = fixed.replace(bad, good)
            total_replacements += count

    if fixed != original:
        backup_path = backup_root / "__".join(path.parts)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
        path.write_text(fixed, encoding="utf-8")
        print(f"  Updated. Backup: {backup_path}")
    else:
        print("  No changes needed.")

print(f"\nDone. Total replacements: {total_replacements}")