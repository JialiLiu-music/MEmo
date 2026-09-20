#!/usr/bin/env python
"""Convert manifest audio entries to normalized WAV assets.

The script keeps the original MP3 path in raw_audio_path and points audio_path to
model-ready WAV files. WAV output is mono PCM at the selected sample rate.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

TARGET_FIELD = "audio_path"
RAW_FIELD = "raw_audio_path"


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest is missing a header row")
        return list(reader.fieldnames), list(reader)


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False, dir=path.parent) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def ensure_field(fieldnames: list[str], field: str, after: str) -> list[str]:
    if field in fieldnames:
        return fieldnames
    if after not in fieldnames:
        return [*fieldnames, field]
    index = fieldnames.index(after) + 1
    return [*fieldnames[:index], field, *fieldnames[index:]]


def convert_one(ffmpeg: str, source: Path, target: Path, sample_rate: int, overwrite: bool) -> bool:
    if target.exists() and not overwrite:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    subprocess.run(command, check=True)
    return True


def convert_manifest(
    project_root: Path,
    manifest_path: Path,
    output_root: Path,
    sample_rate: int,
    ffmpeg: str,
    overwrite: bool,
) -> dict[str, int | str]:
    fieldnames, rows = read_manifest(manifest_path)
    fieldnames = ensure_field(fieldnames, RAW_FIELD, TARGET_FIELD)

    converted = 0
    reused = 0
    missing = 0
    updated_rows: list[dict[str, str]] = []

    for row in rows:
        current_audio = row.get(TARGET_FIELD, "").strip()
        raw_audio = row.get(RAW_FIELD, "").strip() or current_audio
        source = project_root / raw_audio
        track_id = row.get("track_id", "").strip() or source.stem
        target = output_root / f"{track_id}.wav"

        if not source.is_file():
            missing += 1
            updated_rows.append(row)
            continue

        did_convert = convert_one(ffmpeg, source, target, sample_rate, overwrite)
        converted += int(did_convert)
        reused += int(not did_convert)

        updated = dict(row)
        updated[RAW_FIELD] = raw_audio
        updated[TARGET_FIELD] = str(target.relative_to(project_root))
        updated["sample_rate"] = str(sample_rate)
        updated_rows.append(updated)

    write_manifest(manifest_path, fieldnames, updated_rows)
    return {
        "row_count": len(rows),
        "converted_rows": converted,
        "reused_rows": reused,
        "missing_source_rows": missing,
        "sample_rate": sample_rate,
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root
    manifest_path = args.manifest or project_root / "data" / "manifest.csv"
    output_root = args.output_root or project_root / "data" / "audio_wav"

    stats = convert_manifest(
        project_root=project_root,
        manifest_path=manifest_path,
        output_root=output_root,
        sample_rate=args.sample_rate,
        ffmpeg=args.ffmpeg,
        overwrite=args.overwrite,
    )
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()