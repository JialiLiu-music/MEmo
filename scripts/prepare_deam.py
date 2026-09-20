#!/usr/bin/env python
"""Build a reproducible DEAM manifest and validate raw dynamic labels.

This first stage deliberately uses only the standard library. Audio duration and
sample rate remain unknown until an audio decoder is installed; those fields are
left empty instead of being guessed.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

MANIFEST_FIELDS = [
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


def parse_track_id(value: str) -> str:
    match = re.fullmatch(r"(\d+)", value)
    if not match:
        raise ValueError(f"unsupported DEAM track name: {value}")
    return f"deam_{int(match.group(1)):04d}"


def dynamic_label_index(path: Path) -> tuple[set[str], list[str], int]:
    """Return IDs, ordered sample columns, and first sample time in ms."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        sample_columns = [column for column in header[1:] if column.startswith("sample_")]
        sample_times = [int(column.removeprefix("sample_").removesuffix("ms")) for column in sample_columns]
        ids = {row[0].strip() for row in reader if row and row[0].strip()}
    if not sample_times:
        raise ValueError(f"no sample columns found in {path}")
    return ids, sample_columns, min(sample_times)


def build_manifest(project_root: Path, output_path: Path) -> dict[str, int]:
    data_root = project_root / "data"
    audio_root = data_root / "DEAM_audio"
    labels_root = data_root / "annotations_raw" / "annotations averaged per song" / "dynamic (per second annotations)"
    valence_path = labels_root / "valence.csv"
    arousal_path = labels_root / "arousal.csv"

    if not audio_root.is_dir():
        raise FileNotFoundError(audio_root)
    if not valence_path.is_file() or not arousal_path.is_file():
        raise FileNotFoundError("DEAM dynamic valence/arousal CSV is missing")

    valence_ids, valence_samples, valence_first_ms = dynamic_label_index(valence_path)
    arousal_ids, arousal_samples, arousal_first_ms = dynamic_label_index(arousal_path)
    common_samples = [column for column in valence_samples if column in set(arousal_samples)]
    if not common_samples:
        raise ValueError("valence/arousal sample grids have no shared columns")
    shared_first_ms = min(
        int(column.removeprefix("sample_").removesuffix("ms")) for column in common_samples
    )

    rows = []
    for audio_path in sorted(audio_root.glob("*.mp3"), key=lambda path: int(path.stem)):
        numeric_id = audio_path.stem
        track_id = parse_track_id(numeric_id)
        label_id = str(int(numeric_id))
        reasons = []
        if label_id not in valence_ids:
            reasons.append("missing_valence")
        if label_id not in arousal_ids:
            reasons.append("missing_arousal")
        quality = "ok" if not reasons else "excluded"
        rows.append(
            {
                "track_id": track_id,
                "source_dataset": "DEAM",
                "audio_path": str(audio_path.relative_to(project_root)),
                "duration_sec": "",
                "sample_rate": "",
                "label_path": str(labels_root.relative_to(project_root)),
                "label_hz": "2.0",
                "label_start_sec": f"{shared_first_ms / 1000:.1f}",
                "label_end_sec": "",
                "structure_path": "",
                "structure_version": "",
                "split": "unassigned",
                "license_status": "unknown",
                "quality_status": quality,
                "exclude_reason": ";".join(reasons),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "audio_count": len(rows),
        "valence_track_count": len(valence_ids),
        "arousal_track_count": len(arousal_ids),
        "valence_sample_count": len(valence_samples),
        "arousal_sample_count": len(arousal_samples),
        "shared_sample_count": len(common_samples),
        "first_shared_sample_ms": shared_first_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stats = build_manifest(args.project_root, args.output)
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()