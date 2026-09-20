#!/usr/bin/env python3
"""Build an independent index for frozen All-In-One JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "track_id",
    "audio_path",
    "structure_path",
    "structure_version",
    "structure_start_sec",
    "structure_end_sec",
    "structure_segment_count",
    "bpm",
    "analysis_utc",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def numeric(value: object) -> str:
    return "" if value is None else str(value)


def build_row(row: dict[str, str], structure_root: Path, project_root: Path, version: str) -> dict[str, str]:
    track_id = row["track_id"].strip()
    structure = structure_root / f"{track_id}.json"
    result = {field: "" for field in FIELDS}
    result.update(
        {
            "track_id": track_id,
            "audio_path": row.get("audio_path", ""),
            "structure_path": str(structure.relative_to(project_root)),
            "structure_version": version,
            "analysis_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    if not structure.is_file():
        return result

    payload = json.loads(structure.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    starts = [float(item["start"]) for item in segments if "start" in item]
    ends = [float(item["end"]) for item in segments if "end" in item]
    result["structure_start_sec"] = numeric(min(starts) if starts else None)
    result["structure_end_sec"] = numeric(max(ends) if ends else None)
    result["structure_segment_count"] = str(len(segments))
    result["bpm"] = numeric(payload.get("bpm"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(build_row(row, args.structure_dir, args.project_root, args.version))
    print(f"index_rows={len(rows)}")
    print(f"index_path={args.output}")


if __name__ == "__main__":
    main()