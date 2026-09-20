#!/usr/bin/env python
"""Backfill DEAM manifest rows with audio duration and sample rate.

The script uses mutagen so the remote environment does not need ffmpeg or
ffprobe. It keeps the manifest schema stable and rewrites the file atomically.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

from mutagen.mp3 import MP3

REQUIRED_FIELDS = [
    "track_id",
    "source_dataset",
    "audio_path",
    "duration_sec",
    "sample_rate",
    "label_path",
    "label_hz",
    "label_start_sec",
    "label_end_sec",
    "structure_path",
    "structure_version",
    "split",
    "license_status",
    "quality_status",
    "exclude_reason",
]


def load_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest is missing a header row")
        missing = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest is missing fields: {', '.join(missing)}")
        return list(reader.fieldnames), list(reader)


def enrich_rows(project_root: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    updated = 0
    skipped = 0
    for row in rows:
        audio_path = project_root / row["audio_path"]
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        audio = MP3(audio_path)
        duration = f"{audio.info.length:.3f}"
        sample_rate = str(int(audio.info.sample_rate))
        if row.get("duration_sec", "").strip() != duration or row.get("sample_rate", "").strip() != sample_rate:
            updated += 1
        else:
            skipped += 1
        row["duration_sec"] = duration
        row["sample_rate"] = sample_rate
    return {"updated_rows": updated, "already_aligned_rows": skipped}


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False, dir=path.parent) as tmp:
        tmp_path = Path(tmp.name)
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fieldnames, rows = load_manifest(args.manifest)
    stats = enrich_rows(args.project_root, rows)
    output_path = args.output or args.manifest
    write_manifest(output_path, fieldnames, rows)
    for key, value in stats.items():
        print(f"{key}={value}")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()