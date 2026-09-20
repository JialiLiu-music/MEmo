#!/usr/bin/env python3
"""Validate frozen MERT feature files and their 1 Hz time contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EXPECTED_MODEL = "m-a-p/MERT-v1-95M"
EXPECTED_DIM = 768


def manifest_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["track_id"].strip() for row in csv.DictReader(handle)}


def validate(project_root: Path, manifest: Path, output_dir: Path) -> dict[str, int]:
    expected = manifest_ids(manifest)
    found = {path.stem for path in output_dir.glob("*.npz")}
    missing = expected - found
    extra = found - expected
    invalid = 0
    total_frames = 0
    for track_id in sorted(expected & found):
        feature_path = output_dir / f"{track_id}.npz"
        metadata_path = output_dir / f"{track_id}.json"
        try:
            with np.load(feature_path) as payload:
                features = payload["features"]
                times = payload["time_sec"]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if features.ndim != 2 or features.shape[1] != EXPECTED_DIM:
                raise ValueError(f"feature shape={features.shape}")
            if times.ndim != 1 or times.shape[0] != features.shape[0]:
                raise ValueError("time and feature lengths differ")
            if not np.isfinite(features).all() or not np.isfinite(times).all():
                raise ValueError("non-finite feature or timestamp")
            if times.shape[0] and not np.allclose(times, np.arange(times.shape[0]) + 0.5):
                raise ValueError("timestamps are not 1 Hz bin centers")
            if metadata.get("model_name") != EXPECTED_MODEL:
                raise ValueError("unexpected model name")
            if metadata.get("feature_shape") != list(features.shape):
                raise ValueError("metadata feature shape mismatch")
            total_frames += features.shape[0]
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            invalid += 1
            print(f"invalid track_id={track_id}: {exc}")
    result = {
        "manifest_tracks": len(expected),
        "feature_files": len(found),
        "missing": len(missing),
        "extra": len(extra),
        "invalid": invalid,
        "total_frames": total_frames,
    }
    print(json.dumps(result, ensure_ascii=False))
    if missing or extra or invalid:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    validate(args.project_root, args.manifest, args.output_dir)


if __name__ == "__main__":
    main()