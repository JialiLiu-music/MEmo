#!/usr/bin/env python3
"""Merge a validated All-In-One structure index into manifest.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

STRUCTURE_FIELDS = (
    "structure_path",
    "structure_version",
    "structure_start_sec",
    "structure_end_sec",
    "structure_segment_count",
)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "track_id" not in reader.fieldnames:
            raise ValueError("manifest or index must contain track_id")
        return list(reader.fieldnames), list(reader)


def merge(manifest: Path, index: Path, output: Path) -> dict[str, int]:
    manifest_fields, manifest_rows = read_rows(manifest)
    index_fields, index_rows = read_rows(index)
    required = {"track_id", *STRUCTURE_FIELDS}
    missing = required.difference(index_fields)
    if missing:
        raise ValueError(f"structure index missing fields: {sorted(missing)}")

    by_track = {row["track_id"].strip(): row for row in index_rows}
    if len(by_track) != len(index_rows):
        raise ValueError("structure index contains duplicate track_id")
    manifest_ids = {row["track_id"].strip() for row in manifest_rows}
    unknown = set(by_track).difference(manifest_ids)
    if unknown:
        raise ValueError(f"structure index contains unknown track_id: {sorted(unknown)[:3]}")

    output_fields = list(manifest_fields)
    for field in STRUCTURE_FIELDS:
        if field not in output_fields:
            output_fields.append(field)

    updated = 0
    complete = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in manifest_rows:
            structure = by_track.get(row["track_id"].strip())
            if structure is not None:
                for field in STRUCTURE_FIELDS:
                    row[field] = structure.get(field, "")
                updated += 1
                complete += int(all(row.get(field, "").strip() for field in STRUCTURE_FIELDS))
            writer.writerow({field: row.get(field, "") for field in output_fields})

    return {"manifest_rows": len(manifest_rows), "updated_rows": updated, "complete_rows": complete}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key, value in merge(args.manifest, args.index, args.output).items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()