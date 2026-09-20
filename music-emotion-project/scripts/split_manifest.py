#!/usr/bin/env python
"""Assign deterministic track-level train/validation/test splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import tempfile
from pathlib import Path

SPLITS = ("train", "val", "test")


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest is missing a header row")
        if "split" not in reader.fieldnames:
            raise ValueError("manifest is missing split field")
        if "track_id" not in reader.fieldnames:
            raise ValueError("manifest is missing track_id field")
        return list(reader.fieldnames), list(reader)


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False, dir=path.parent) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def stable_score(track_id: str, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}:{track_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def assign_splits(rows: list[dict[str, str]], seed: str, train_ratio: float, val_ratio: float) -> dict[str, str]:
    track_ids = sorted({row["track_id"].strip() for row in rows if row.get("track_id", "").strip()})
    ordered = sorted(track_ids, key=lambda track_id: stable_score(track_id, seed))

    train_count = round(len(ordered) * train_ratio)
    val_count = round(len(ordered) * val_ratio)
    if train_count + val_count >= len(ordered):
        raise ValueError("split ratios leave no test tracks")

    split_by_track: dict[str, str] = {}
    for index, track_id in enumerate(ordered):
        if index < train_count:
            split_by_track[track_id] = "train"
        elif index < train_count + val_count:
            split_by_track[track_id] = "val"
        else:
            split_by_track[track_id] = "test"
    return split_by_track


def split_manifest(
    manifest_path: Path,
    seed: str,
    train_ratio: float,
    val_ratio: float,
    overwrite: bool,
) -> dict[str, int | str]:
    fieldnames, rows = read_manifest(manifest_path)
    existing = {row.get("split", "").strip() for row in rows}
    already_fixed = existing <= set(SPLITS) and "unassigned" not in existing and "" not in existing
    if already_fixed and not overwrite:
        counts = {name: sum(1 for row in rows if row.get("split") == name) for name in SPLITS}
        return {
            "row_count": len(rows),
            "updated_rows": 0,
            "train_rows": counts["train"],
            "val_rows": counts["val"],
            "test_rows": counts["test"],
            "seed": seed,
        }

    split_by_track = assign_splits(rows, seed, train_ratio, val_ratio)
    updated_rows = []
    for row in rows:
        updated = dict(row)
        updated["split"] = split_by_track[updated["track_id"].strip()]
        updated_rows.append(updated)

    write_manifest(manifest_path, fieldnames, updated_rows)
    counts = {name: sum(1 for row in updated_rows if row.get("split") == name) for name in SPLITS}
    return {
        "row_count": len(rows),
        "updated_rows": len(updated_rows),
        "train_rows": counts["train"],
        "val_rows": counts["val"],
        "test_rows": counts["test"],
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", default="deam-structure-emotion-v1")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stats = split_manifest(
        manifest_path=args.manifest,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        overwrite=args.overwrite,
    )
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()