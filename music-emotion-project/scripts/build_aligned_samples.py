#!/usr/bin/env python3
"""Create one aligned training-row table from labels and structure index."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "track_id",
    "split",
    "time_sec",
    "audio_path",
    "label_path",
    "valence",
    "arousal",
    "valid",
    "segment_index",
    "segment_label",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def structure_segments(project_root: Path, path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"structure segments missing: {path}")
    normalized = []
    previous_end = None
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= start or (previous_end is not None and start < previous_end):
            raise ValueError(f"invalid segment order at {path}:{index}")
        normalized.append({"start": start, "end": end, "label": str(segment.get("label", "unknown"))})
        previous_end = end
    return normalized


def segment_at(segments: list[dict[str, object]], time_sec: float) -> tuple[str, str]:
    for index, segment in enumerate(segments):
        if float(segment["start"]) <= time_sec < float(segment["end"]):
            return str(index), str(segment["label"])
    return "", ""


def build(project_root: Path, manifest_path: Path, structure_index: Path, output: Path) -> dict[str, int]:
    manifest = {row["track_id"].strip(): row for row in rows(manifest_path)}
    index = {row["track_id"].strip(): row for row in rows(structure_index)}
    if set(manifest) != set(index):
        raise ValueError("manifest and structure index track_id sets differ")

    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    valid = 0
    aligned = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for track_id, manifest_row in manifest.items():
            structure_path = project_root / index[track_id]["structure_path"]
            segments = structure_segments(project_root, structure_path)
            label_root = project_root / manifest_row["label_path"]
            label_path = label_root / f"{track_id}.csv" if label_root.is_dir() else label_root
            for label_row in rows(label_path):
                time_sec = float(label_row["time_sec"])
                segment_index, segment_label = segment_at(segments, time_sec)
                row = {
                    "track_id": track_id,
                    "split": manifest_row.get("split", ""),
                    "time_sec": label_row["time_sec"],
                    "audio_path": manifest_row.get("audio_path", ""),
                    "label_path": manifest_row["label_path"],
                    "valence": label_row.get("valence", ""),
                    "arousal": label_row.get("arousal", ""),
                    "valid": label_row.get("valid", "0"),
                    "segment_index": segment_index,
                    "segment_label": segment_label,
                }
                writer.writerow(row)
                total += 1
                valid += int(row["valid"].strip() == "1")
                aligned += int(bool(segment_index))

    return {"rows_written": total, "valid_rows": valid, "structure_aligned_rows": aligned}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--structure-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key, value in build(args.project_root, args.manifest, args.structure_index, args.output).items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()