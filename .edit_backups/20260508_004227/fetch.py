#!/usr/bin/env python3
"""
fetch.py - print exact pieces of code for review or patching.

Examples:

Print lines 2200 through 2350:
    python fetch.py main.py --lines 2200:2350

Print 40 lines around the first matching phrase:
    python fetch.py static/script.js --contains "Please acknowledge these settings" --context 40

Print 40 lines around the second matching phrase:
    python fetch.py main.py --contains "create_pipeline_chat" --context 40 --nth 2

Print all matching phrase locations with context:
    python fetch.py main.py --contains "create_pipeline_chat" --context 20 --all

Print the Python function named chat:
    python fetch.py main.py --def chat

Print with line numbers:
    python fetch.py main.py --lines 2200:2350 --numbered
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def print_block(path: Path, lines: list[str], start_line: int, end_line: int, numbered: bool) -> None:
    print("---FETCHED---")
    print(f"FILE: {path}")
    print(f"LINES: {start_line}:{end_line}")
    print("---CODE---")

    if numbered:
        width = len(str(end_line))
        for offset, line in enumerate(lines, start=start_line):
            print(f"{offset:>{width}} | {line}")
    else:
        print("\n".join(lines))

    print("---END FETCHED---")


def parse_line_range(spec: str, total_lines: int) -> tuple[int, int]:
    if ":" not in spec:
        raise ValueError("Line range must look like START:END, for example 120:180.")

    left, right = spec.split(":", 1)
    start = int(left)
    end = int(right)

    if start < 1:
        start = 1
    if end > total_lines:
        end = total_lines
    if end < start:
        raise ValueError("Line range end must be greater than or equal to start.")

    return start, end


def find_contains_matches(lines: list[str], needle: str, context: int) -> list[tuple[int, int, int]]:
    matches: list[tuple[int, int, int]] = []

    for index, line in enumerate(lines):
        if needle in line:
            match_line = index + 1
            start = max(1, match_line - context)
            end = min(len(lines), match_line + context)
            matches.append((match_line, start, end))

    return matches


def find_python_def(path: Path, text: str, name: str, nth: int) -> tuple[int, int]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse Python file because it has a syntax error: {exc}") from exc

    matches: list[ast.AST] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            matches.append(node)

    matches.sort(key=lambda n: getattr(n, "lineno", 0))

    if not matches:
        raise ValueError(f"No Python function/class named {name!r} found in {path}.")

    if nth < 1 or nth > len(matches):
        raise ValueError(f"Requested --nth {nth}, but only found {len(matches)} definition(s) named {name!r}.")

    node = matches[nth - 1]
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)

    if start is None or end is None:
        raise ValueError("Could not determine line numbers for that definition.")

    return int(start), int(end)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch exact code blocks from a file.")
    parser.add_argument("file", help="File to inspect, e.g. main.py or static/script.js")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lines", help="Line range START:END, e.g. 100:150")
    group.add_argument("--contains", help="Find occurrence(s) of this text and print nearby lines")
    group.add_argument("--def", dest="def_name", help="Print a Python function/class by name")

    parser.add_argument("--context", type=int, default=30, help="Lines before/after --contains match. Default: 30")
    parser.add_argument("--numbered", action="store_true", help="Include line numbers in output")
    parser.add_argument("--nth", type=int, default=1, help="Which match to print. Default: 1")
    parser.add_argument("--all", action="store_true", help="Print all matches for --contains")
    args = parser.parse_args()

    path = Path(args.file)

    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 1

    text = read_text_file(path)
    lines = text.splitlines()

    try:
        if args.lines:
            start, end = parse_line_range(args.lines, len(lines))
            selected = lines[start - 1:end]
            print_block(path, selected, start, end, args.numbered)

        elif args.contains:
            matches = find_contains_matches(lines, args.contains, args.context)

            if not matches:
                raise ValueError(f"Text not found: {args.contains!r}")

            print(f"FOUND {len(matches)} MATCH(ES) for {args.contains!r} in {path}:")
            for idx, (match_line, start, end) in enumerate(matches, start=1):
                print(f"  {idx}. line {match_line}, fetch range {start}:{end}")
            print()

            if args.all:
                for idx, (match_line, start, end) in enumerate(matches, start=1):
                    print(f"=== MATCH {idx} at line {match_line} ===")
                    selected = lines[start - 1:end]
                    print_block(path, selected, start, end, args.numbered)
                    print()
            else:
                if args.nth < 1 or args.nth > len(matches):
                    raise ValueError(f"Requested --nth {args.nth}, but only found {len(matches)} match(es).")

                match_line, start, end = matches[args.nth - 1]
                selected = lines[start - 1:end]
                print_block(path, selected, start, end, args.numbered)

        elif args.def_name:
            start, end = find_python_def(path, text, args.def_name, args.nth)
            selected = lines[start - 1:end]
            print_block(path, selected, start, end, args.numbered)

        else:
            raise ValueError("No fetch mode selected.")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())