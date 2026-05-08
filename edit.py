#!/usr/bin/env python3
"""
edit.py - exact, safe code replacement tool.

Usage:
    python edit.py patches.txt
    python edit.py patches.txt --dry-run
    python edit.py patches.txt --no-compile

Patch format:

---PATCH---
NAME: optional human-readable patch name
FILE: path/to/file.py
COUNT: 1
---OLD---
exact old code here
---NEW---
exact new code here
---END---

Rules:
- The OLD block must match the file exactly.
- COUNT defaults to 1.
- If OLD is found zero times, the script refuses to edit.
- If OLD is found more/less than COUNT times, the script refuses to edit.
- Backups are created automatically in .edit_backups/.
- Python files are syntax-checked after editing unless --no-compile is used.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import py_compile
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


PATCH_START = "---PATCH---"
OLD_MARKER = "---OLD---"
NEW_MARKER = "---NEW---"
END_MARKER = "---END---"


@dataclass
class Patch:
    name: str
    file: Path
    old: str
    new: str
    count: int = 1


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def parse_header(lines: list[str]) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"Bad patch header line: {line!r}")
        key, value = stripped.split(":", 1)
        data[key.strip().upper()] = value.strip()
    return data


def parse_patches(patch_text: str) -> list[Patch]:
    lines = patch_text.splitlines()
    patches: list[Patch] = []
    i = 0

    while i < len(lines):
        if lines[i].strip() != PATCH_START:
            i += 1
            continue

        i += 1
        header_lines: list[str] = []

        while i < len(lines) and lines[i].strip() != OLD_MARKER:
            header_lines.append(lines[i])
            i += 1

        if i >= len(lines) or lines[i].strip() != OLD_MARKER:
            raise ValueError("Patch is missing ---OLD--- marker.")

        header = parse_header(header_lines)

        if "FILE" not in header:
            raise ValueError("Patch is missing FILE: header.")

        name = header.get("NAME", header["FILE"])
        file_path = Path(header["FILE"])

        try:
            count = int(header.get("COUNT", "1"))
        except ValueError as exc:
            raise ValueError(f"COUNT must be an integer in patch {name!r}.") from exc

        i += 1
        old_lines: list[str] = []

        while i < len(lines) and lines[i].strip() != NEW_MARKER:
            old_lines.append(lines[i])
            i += 1

        if i >= len(lines) or lines[i].strip() != NEW_MARKER:
            raise ValueError(f"Patch {name!r} is missing ---NEW--- marker.")

        i += 1
        new_lines: list[str] = []

        while i < len(lines) and lines[i].strip() != END_MARKER:
            new_lines.append(lines[i])
            i += 1

        if i >= len(lines) or lines[i].strip() != END_MARKER:
            raise ValueError(f"Patch {name!r} is missing ---END--- marker.")

        old = "\n".join(old_lines)
        new = "\n".join(new_lines)

        patches.append(Patch(name=name, file=file_path, old=old, new=new, count=count))
        i += 1

    if not patches:
        raise ValueError("No patches found. Expected at least one ---PATCH--- block.")

    return patches


def make_backup(path: Path, backup_root: Path) -> Path:
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "__".join(path.parts)
    backup_path = backup_root / timestamp / safe_name
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)
    return backup_path


def compile_python_file(path: Path) -> None:
    py_compile.compile(str(path), doraise=True)


def apply_patches(patches: list[Patch], dry_run: bool, no_compile: bool) -> int:
    backup_root = Path(".edit_backups")
    touched_files: set[Path] = set()
    backups: dict[Path, Path] = {}

    for patch in patches:
        path = patch.file

        if not path.exists():
            print(f"ERROR: File not found for patch {patch.name!r}: {path}", file=sys.stderr)
            return 1

        text = read_text_file(path)
        actual_count = text.count(patch.old)

        print(f"\nPatch: {patch.name}")
        print(f"File:  {path}")
        print(f"Expected matches: {patch.count}")
        print(f"Actual matches:   {actual_count}")

        if actual_count != patch.count:
            print("ERROR: Match count mismatch. Refusing to edit this file.", file=sys.stderr)
            print("Tip: use fetch.py to copy the exact current code block again.", file=sys.stderr)
            return 1

        updated = text.replace(patch.old, patch.new, patch.count)

        if updated == text:
            print("ERROR: Replacement produced no change. Refusing to continue.", file=sys.stderr)
            return 1

        if dry_run:
            print("DRY RUN: patch would apply cleanly.")
            continue

        if path not in backups:
            backups[path] = make_backup(path, backup_root)
            print(f"Backup created: {backups[path]}")

        write_text_file(path, updated)
        touched_files.add(path)
        print("Applied.")

    if dry_run:
        print("\nDry run complete. No files changed.")
        return 0

    if not no_compile:
        py_files = sorted(path for path in touched_files if path.suffix == ".py")
        for path in py_files:
            print(f"\nSyntax check: python -m py_compile {path}")
            try:
                compile_python_file(path)
            except Exception as exc:
                print(f"ERROR: Python syntax check failed for {path}.", file=sys.stderr)
                print(str(exc), file=sys.stderr)

                print("\nRestoring backups because syntax check failed...")
                for original, backup in backups.items():
                    shutil.copy2(backup, original)
                    print(f"Restored {original} from {backup}")

                return 1

            print("OK.")

    print("\nAll patches applied successfully.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply exact code replacement patches safely.")
    parser.add_argument("patch_file", help="Patch file to apply, e.g. patches.txt")
    parser.add_argument("--dry-run", action="store_true", help="Check whether patches match without editing files.")
    parser.add_argument("--no-compile", action="store_true", help="Skip py_compile syntax check for touched .py files.")
    args = parser.parse_args()

    patch_path = Path(args.patch_file)

    if not patch_path.exists():
        print(f"ERROR: Patch file not found: {patch_path}", file=sys.stderr)
        return 1

    try:
        patch_text = read_text_file(patch_path)
        patches = parse_patches(patch_text)
    except Exception as exc:
        print(f"ERROR: Could not parse patch file: {exc}", file=sys.stderr)
        return 1

    return apply_patches(patches, dry_run=args.dry_run, no_compile=args.no_compile)


if __name__ == "__main__":
    raise SystemExit(main())