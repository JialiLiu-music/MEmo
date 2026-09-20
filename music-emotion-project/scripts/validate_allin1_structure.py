#!/usr/bin/env python3
"""Validate All-In-One JSON files and their independent structure index."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate(project_root: Path, index_path: Path) -> dict[str, int]:
    rows = read_rows(index_path)
    missing_json = 0
    invalid_json = 0
    invalid_segments = 0
    blank_bounds = 0
    indexed = 0

    for row in rows:
        path = project_root / row.get("structure_path", "")
        if not path.is_file():
            missing_json += 1
            continue
        indexed += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_json += 1
            continue
        segments = payload.get("segments")
        if not isinstance(segments, list):
            invalid_segments += 1
            continue
        previous_end = None
        for segment in segments:
            if not isinstance(segment, dict) or not {"start", "end", "label"}.issubset(segment):
                invalid_segments += 1
                break
            start = float(segment["start"])
            end = float(segment["end"])
            if end < start or (previous_end is not None and start < previous_end):
                invalid_segments += 1
                break
            previous_end = end
        if not row.get("structure_start_sec") or not row.get("structure_end_sec"):
            blank_bounds += 1

    return {
        "index_rows": len(rows),
        "indexed_json": indexed,
        "missing_json": missing_json,
        "invalid_json": invalid_json,
        "invalid_segments": invalid_segments,
        "blank_bounds": blank_bounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    for key, value in validate(args.project_root, args.index).items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()