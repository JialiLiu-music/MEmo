#!/usr/bin/env python
"""Validate model-ready assets referenced by manifest.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SPLITS = ("train", "val", "test")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest is missing a header row")
        return list(reader)


def label_file(project_root: Path, row: dict[str, str]) -> Path:
    label_path = row.get("label_path", "").strip()
    track_id = row.get("track_id", "").strip()
    path = project_root / label_path
    if path.is_dir():
        return path / f"{track_id}.csv"
    return path


def count_valid_label_rows(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        total = 0
        valid = 0
        for row in reader:
            total += 1
            valid += int(row.get("valid", "").strip() == "1")
    return total, valid


def validate_assets(project_root: Path, manifest_path: Path) -> dict[str, int]:
    rows = read_manifest(manifest_path)
    missing_audio = 0
    missing_raw_audio = 0
    missing_label_file = 0
    invalid_split = 0
    wav_audio_rows = 0
    total_label_rows = 0
    valid_label_rows = 0

    split_counts = {split: 0 for split in SPLITS}

    for row in rows:
        audio = project_root / row.get("audio_path", "")
        raw_audio = project_root / row.get("raw_audio_path", "") if row.get("raw_audio_path") else None
        labels = label_file(project_root, row)
        split = row.get("split", "").strip()

        if not audio.is_file():
            missing_audio += 1
        if audio.suffix.lower() == ".wav":
            wav_audio_rows += 1
        if raw_audio is not None and not raw_audio.is_file():
            missing_raw_audio += 1
        if split not in SPLITS:
            invalid_split += 1
        else:
            split_counts[split] += 1
        if not labels.is_file():
            missing_label_file += 1
            continue
        label_rows, label_valid = count_valid_label_rows(labels)
        total_label_rows += label_rows
        valid_label_rows += label_valid

    return {
        "row_count": len(rows),
        "wav_audio_rows": wav_audio_rows,
        "missing_audio": missing_audio,
        "missing_raw_audio": missing_raw_audio,
        "missing_label_file": missing_label_file,
        "invalid_split_rows": invalid_split,
        "train_rows": split_counts["train"],
        "val_rows": split_counts["val"],
        "test_rows": split_counts["test"],
        "total_label_rows": total_label_rows,
        "valid_label_rows": valid_label_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    stats = validate_assets(args.project_root, args.manifest)
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()