#!/usr/bin/env python
"""Aggregate DEAM dynamic labels from 2 Hz to 1 Hz and update the manifest."""

from __future__ import annotations

import argparse
import csv
import re
import tempfile
from pathlib import Path

LABEL_FIELDS = ["time_sec", "valence", "arousal", "valid"]
SAMPLE_PATTERN = re.compile(r"^sample_(\d+)ms$")


def parse_sample_columns(header: list[str]) -> list[tuple[int, str]]:
    samples: list[tuple[int, str]] = []
    for column in header[1:]:
        match = SAMPLE_PATTERN.fullmatch(column)
        if match:
            samples.append((int(match.group(1)), column))
    samples.sort(key=lambda item: item[0])
    return samples


def read_dynamic_table(path: Path) -> tuple[list[tuple[int, str]], dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing header row in {path}")
        samples = parse_sample_columns(reader.fieldnames)
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            track_id = (row.get(reader.fieldnames[0]) or "").strip()
            if not track_id:
                continue
            rows[track_id] = {key: (value or "").strip() for key, value in row.items()}
    if not samples:
        raise ValueError(f"no sample columns found in {path}")
    return samples, rows


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed != parsed:
        return None
    return parsed


def mean_pair(left: str, right: str) -> tuple[str, int]:
    left_value = parse_float(left)
    right_value = parse_float(right)
    if left_value is None or right_value is None:
        return "", 0
    return f"{(left_value + right_value) / 2:.7f}", 1


def atomic_write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", delete=False, dir=path.parent
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def aggregate_track(
    track_id: str,
    valence_row: dict[str, str],
    arousal_row: dict[str, str],
    shared_samples: list[tuple[int, str]],
) -> tuple[list[dict[str, str]], int]:
    shared_ms = {ms for ms, _ in shared_samples}
    if not shared_samples:
        raise ValueError(f"no shared dynamic samples for {track_id}")

    start_ms = shared_samples[0][0]
    end_ms = shared_samples[-1][0]
    rows: list[dict[str, str]] = []
    valid_rows = 0

    for left_ms in range(start_ms, end_ms, 1000):
        right_ms = left_ms + 500
        if left_ms not in shared_ms or right_ms not in shared_ms:
            continue
        left_column = f"sample_{left_ms}ms"
        right_column = f"sample_{right_ms}ms"
        mean_valence, valid_valence = mean_pair(valence_row.get(left_column, ""), valence_row.get(right_column, ""))
        mean_arousal, valid_arousal = mean_pair(arousal_row.get(left_column, ""), arousal_row.get(right_column, ""))
        valid = int(valid_valence and valid_arousal)
        if valid:
            valid_rows += 1
        rows.append(
            {
                "time_sec": f"{(left_ms + 500) / 1000:.1f}",
                "valence": mean_valence,
                "arousal": mean_arousal,
                "valid": str(valid),
            }
        )

    if not rows:
        raise ValueError(f"no 1 Hz label rows could be generated for {track_id}")
    return rows, valid_rows


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest is missing a header row")
        return list(reader)


def write_manifest(manifest_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    atomic_write_csv(manifest_path, rows, fieldnames)


def aggregate(project_root: Path, manifest_path: Path, labels_root: Path) -> dict[str, int | str]:
    data_root = project_root / "data"
    source_root = data_root / "annotations_raw" / "annotations averaged per song" / "dynamic (per second annotations)"
    valence_path = source_root / "valence.csv"
    arousal_path = source_root / "arousal.csv"

    valence_samples, valence_rows = read_dynamic_table(valence_path)
    arousal_samples, arousal_rows = read_dynamic_table(arousal_path)

    valence_ms = [ms for ms, _ in valence_samples]
    arousal_ms = {ms for ms, _ in arousal_samples}
    shared_samples = [(ms, column) for ms, column in valence_samples if ms in arousal_ms]
    shared_ms = [ms for ms, _ in shared_samples]

    if not shared_samples:
        raise ValueError("valence/arousal sample grids have no shared columns")
    if any(right - left != 500 for left, right in zip(shared_ms, shared_ms[1:])):
        raise ValueError("shared dynamic samples are not uniformly spaced at 500 ms")

    manifest_rows = read_manifest(manifest_path)
    manifest_fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    updated_manifest_rows: list[dict[str, str]] = []

    generated_tracks = 0
    generated_rows = 0
    valid_rows = 0

    for row in manifest_rows:
        audio_path = row.get("audio_path", "").strip()
        stem = Path(audio_path).stem
        label_id = stem
        if label_id.isdigit():
            track_id = f"deam_{int(label_id):04d}"
        else:
            track_id = row.get("track_id", "").strip() or stem
        if not label_id.isdigit():
            raise ValueError(f"unexpected DEAM audio path: {audio_path}")

        valence_row = valence_rows.get(label_id)
        arousal_row = arousal_rows.get(label_id)
        if valence_row is None or arousal_row is None:
            raise ValueError(f"missing dynamic labels for track {label_id}")

        track_rows, valid_count = aggregate_track(track_id, valence_row, arousal_row, shared_samples)
        atomic_write_csv(labels_root / f"{track_id}.csv", track_rows, LABEL_FIELDS)

        updated = dict(row)
        updated["track_id"] = track_id
        updated["label_path"] = str(labels_root.relative_to(project_root))
        updated["label_hz"] = "1.0"
        updated["label_start_sec"] = track_rows[0]["time_sec"]
        updated["label_end_sec"] = track_rows[-1]["time_sec"]
        updated_manifest_rows.append(updated)

        generated_tracks += 1
        generated_rows += len(track_rows)
        valid_rows += valid_count

    if manifest_fieldnames:
        write_manifest(manifest_path, updated_manifest_rows, manifest_fieldnames)

    return {
        "tracks_written": generated_tracks,
        "rows_written": generated_rows,
        "valid_rows": valid_rows,
        "shared_sample_count": len(shared_samples),
        "shared_start_ms": shared_ms[0],
        "shared_end_ms": shared_ms[-1],
        "manifest_path": str(manifest_path),
        "labels_root": str(labels_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--labels-root", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root
    manifest_path = args.manifest or project_root / "data" / "manifest.csv"
    labels_root = args.labels_root or project_root / "data" / "labels_1hz"

    stats = aggregate(project_root, manifest_path, labels_root)
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()