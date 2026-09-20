#!/usr/bin/env python3
"""Validate 1 Hz MERT timestamps against aligned VA sample rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate(aligned_samples: Path, feature_dir: Path) -> dict[str, int]:
    sample_rows = rows(aligned_samples)
    valid_rows = [row for row in sample_rows if row.get("valid", "0").strip() == "1"]
    missing_files = 0
    missing_timestamps = 0
    invalid_files = 0
    checked_tracks: set[str] = set()
    matched_rows = 0

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in valid_rows:
        grouped.setdefault(row["track_id"].strip(), []).append(row)

    for track_id, track_rows in sorted(grouped.items()):
        feature_path = feature_dir / f"{track_id}.npz"
        checked_tracks.add(track_id)
        if not feature_path.exists():
            missing_files += 1
            continue
        try:
            with np.load(feature_path) as payload:
                features = payload["features"]
                timestamps = payload["time_sec"]
            if features.ndim != 2 or timestamps.ndim != 1 or features.shape[0] != timestamps.shape[0]:
                raise ValueError("feature/time shape mismatch")
            if not np.isfinite(timestamps).all():
                raise ValueError("non-finite timestamps")
            for row in track_rows:
                time_sec = float(row["time_sec"])
                if np.any(np.isclose(timestamps, time_sec, rtol=0.0, atol=1e-6)):
                    matched_rows += 1
                else:
                    missing_timestamps += 1
        except (OSError, KeyError, ValueError) as exc:
            invalid_files += 1
            print(f"invalid track_id={track_id}: {exc}")

    result = {
        "valid_sample_rows": len(valid_rows),
        "checked_tracks": len(checked_tracks),
        "missing_feature_files": missing_files,
        "invalid_feature_files": invalid_files,
        "missing_timestamps": missing_timestamps,
        "matched_rows": matched_rows,
    }
    print(json.dumps(result, ensure_ascii=False))
    if missing_files or invalid_files or missing_timestamps:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-samples", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    args = parser.parse_args()
    validate(args.aligned_samples, args.feature_dir)


if __name__ == "__main__":
    main()