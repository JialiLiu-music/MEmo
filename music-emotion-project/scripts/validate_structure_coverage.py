#!/usr/bin/env python3
"""Validate structure index coverage before manifest integration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "track_id" not in reader.fieldnames:
            raise ValueError(f"missing track_id: {path}")
        values = [row["track_id"].strip() for row in reader]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate track_id: {path}")
    return set(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    manifest_ids = ids(args.manifest)
    index_ids = ids(args.index)
    missing = manifest_ids.difference(index_ids)
    extra = index_ids.difference(manifest_ids)
    print(f"manifest_tracks={len(manifest_ids)}")
    print(f"index_tracks={len(index_ids)}")
    print(f"missing_index_tracks={len(missing)}")
    print(f"extra_index_tracks={len(extra)}")
    if missing or extra:
        raise SystemExit(1)


if __name__ == "__main__":
    main()