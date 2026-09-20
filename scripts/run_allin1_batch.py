#!/usr/bin/env python3
"""Run All-In-One once per manifest row and preserve one JSON per track."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def find_json(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*.json") if path.is_file())


def run_track(cli: str, row: dict[str, str], project_root: Path, output_root: Path) -> dict[str, str]:
    track_id = row["track_id"].strip()
    audio = project_root / row["audio_path"].strip()
    target = output_root / f"{track_id}.json"
    track_dir = output_root / ".runs" / track_id
    track_dir.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        return {"track_id": track_id, "status": "skipped", "output": str(target)}
    if not audio.is_file():
        return {"track_id": track_id, "status": "failed", "error": f"missing_audio:{audio}"}

    command = [cli, str(audio), "-o", str(track_dir)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    json_files = find_json(track_dir)
    if completed.returncode != 0:
        return {
            "track_id": track_id,
            "status": "failed",
            "returncode": str(completed.returncode),
            "error": completed.stderr[-1000:],
        }
    if not json_files:
        return {"track_id": track_id, "status": "failed", "error": "json_output_missing"}

    shutil.copyfile(json_files[0], target)
    return {"track_id": track_id, "status": "written", "output": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cli", default="allin1")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "batch_log.jsonl"
    counts = {"written": 0, "skipped": 0, "failed": 0}

    with log_path.open("a", encoding="utf-8") as log:
        for row in rows:
            result = run_track(args.cli, row, args.project_root, args.output_dir)
            result["audio_path"] = row.get("audio_path", "")
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            counts[result["status"]] += 1
            print(f"track_id={result['track_id']} status={result['status']}")

    for key, value in counts.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()