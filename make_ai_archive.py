from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime

ROOT = Path.cwd()

OUTPUT_ZIP = ROOT / "ai_code_archive.zip"
PATH_SEPARATOR = "___PATH___"

TEXT_EXTENSIONS = {
    ".txt",
    ".json",
    ".py",
    ".css",
    ".js",
    ".html",
    ".htm",
    ".md",
}

SKIP_DIRS = {
    ".git",
    ".cache",
    ".config",
    ".upm",
    ".pythonlibs",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
}

SKIP_FILES = {
    OUTPUT_ZIP.name,
    "make_ai_archive.py",
}

def should_skip_path(path: Path) -> bool:
    parts = set(path.parts)

    if any(part in SKIP_DIRS for part in parts):
        return True

    if path.name in SKIP_FILES:
        return True

    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return True

    return False

def archive_name_for(relative_path: Path) -> str:
    """
    Converts:
        static/style.css
    into:
        static___PATH___style.css

    Converts:
        main.py
    into:
        main.py
    """
    return PATH_SEPARATOR.join(relative_path.parts)

def read_text_safely(path: Path) -> str:
    """
    Try UTF-8 first. If that fails, fall back to a more forgiving read.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")

def main():
    files_added = 0

    with ZipFile(OUTPUT_ZIP, "w", compression=ZIP_DEFLATED) as zipf:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue

            relative_path = path.relative_to(ROOT)

            if should_skip_path(relative_path):
                continue

            original_relative_path = relative_path.as_posix()
            header = f"---FILE:{original_relative_path}---"

            original_content = read_text_safely(path)

            if original_content.startswith("---FILE:"):
                zipped_content = original_content
            else:
                zipped_content = header + "\n" + original_content

            zip_entry_name = archive_name_for(relative_path)

            zipf.writestr(zip_entry_name, zipped_content)
            files_added += 1

            print(f"Added: {original_relative_path} -> {zip_entry_name}")

    print()
    print(f"Done. Created: {OUTPUT_ZIP}")
    print(f"Files added: {files_added}")

if __name__ == "__main__":
    main()
