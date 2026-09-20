#!/usr/bin/env python
"""Validate the DEAM manifest produced by prepare_deam.py.

The validator stays in the standard library so it can run in the same minimal
remote environment. It checks schema, path reachability, and the current label
protocol assumptions used by the manifest builder.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

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


def load_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest is missing a header row")
        missing = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest is missing fields: {', '.join(missing)}")
        return list(reader)


def validate(project_root: Path, manifest_path: Path) -> dict[str, int]:
    rows = load_rows(manifest_path)
    data_root = project_root / "data"
    missing_audio = 0
    missing_label_dir = 0
    excluded_rows = 0
    blank_duration = 0
    blank_sample_rate = 0

    for row in rows:
        audio_path = project_root / row["audio_path"]
        label_path = project_root / row["label_path"]
        if not audio_path.is_file():
            missing_audio += 1
        if not label_path.exists():
            missing_label_dir += 1
        if not row["duration_sec"].strip():
            blank_duration += 1
        if not row["sample_rate"].strip():
            blank_sample_rate += 1
        if row["quality_status"].strip().lower() == "excluded":
            excluded_rows += 1

    return {
        "row_count": len(rows),
        "missing_audio": missing_audio,
        "missing_label_dir": missing_label_dir,
        "excluded_rows": excluded_rows,
        "blank_duration_rows": blank_duration,
        "blank_sample_rate_rows": blank_sample_rate,
        "data_root_exists": int(data_root.exists()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    stats = validate(args.project_root, args.manifest)
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()