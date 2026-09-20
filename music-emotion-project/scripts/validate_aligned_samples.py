#!/usr/bin/env python3
"""Validate the aligned one-second sample table before model training."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

REQUIRED_FIELDS = {
    "track_id",
    "split",
    "time_sec",
    "valence",
    "arousal",
    "valid",
    "segment_index",
}
VALID_SPLITS = {"train", "val", "test", "validation"}


def validate(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_FIELDS.difference(fields)
        if missing:
            raise ValueError(f"aligned sample table missing fields: {sorted(missing)}")

        rows = 0
        valid_rows = 0
        invalid_rows = 0
        blank_segment_rows = 0
        invalid_split_rows = 0
        non_monotonic_rows = 0
        previous_time: dict[str, float] = {}
        tracks: set[str] = set()
        splits: defaultdict[str, int] = defaultdict(int)

        for row in reader:
            rows += 1
            track_id = row["track_id"].strip()
            tracks.add(track_id)
            splits[row["split"].strip()] += 1
            if row["split"].strip() not in VALID_SPLITS:
                invalid_split_rows += 1
            time_sec = float(row["time_sec"])
            if track_id in previous_time and time_sec <= previous_time[track_id]:
                non_monotonic_rows += 1
            previous_time[track_id] = time_sec

            is_valid = row["valid"].strip() == "1"
            if is_valid:
                valid_rows += 1
                if not row["valence"].strip() or not row["arousal"].strip():
                    invalid_rows += 1
            if not row["segment_index"].strip():
                blank_segment_rows += 1

    return {
        "rows": rows,
        "tracks": len(tracks),
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "blank_segment_rows": blank_segment_rows,
        "invalid_split_rows": invalid_split_rows,
        "non_monotonic_rows": non_monotonic_rows,
        "split_count": len(splits),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    args = parser.parse_args()
    stats = validate(args.samples)
    for key, value in stats.items():
        print(f"{key}={value}")
    if stats["invalid_rows"] or stats["invalid_split_rows"] or stats["non_monotonic_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()