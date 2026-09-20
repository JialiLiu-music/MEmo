#!/usr/bin/env python3
"""Build paper-ready summaries from frozen validation and test artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODELS = ("b0", "b1", "b2", "b3", "s1")
SEEDS = (20260903, 20260904, 20260905)
METRICS = ("valence_ccc", "arousal_ccc", "ccc_mean", "valence_mae", "arousal_mae", "mae_mean", "valence_rmse", "arousal_rmse", "rmse_mean")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def seeds_for(model: str) -> tuple[int, ...]:
    return (20260903,) if model == "b0" else SEEDS


def run_dir(root: Path, model: str, seed: int) -> Path:
    return root / f"runs/formal-{model}-seed{seed}-valonly"


def load_track_metrics(root: Path, model: str, seed: int, split: str) -> dict[str, dict[str, float]]:
    path = run_dir(root, model, seed) / f"{split}_predictions.json"
    result = {}
    for item in load_json(path):
        result[item["track_id"]] = item["metrics"]
    return result


def summarize(root: Path, split: str) -> dict[str, dict[str, object]]:
    result = {}
    for model in MODELS:
        values_by_seed = []
        for seed in seeds_for(model):
            per_track = load_track_metrics(root, model, seed, split)
            values_by_seed.append({metric: float(np.mean([row[metric] for row in per_track.values()])) for metric in METRICS})
        result[model] = {}
        for metric in METRICS:
            values = [row[metric] for row in values_by_seed]
            result[model][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "seed_values": values,
            }
    return result


def load_predictions(root: Path, model: str, seed: int, split: str) -> dict[str, dict[str, object]]:
    path = run_dir(root, model, seed) / f"{split}_predictions.json"
    return {item["track_id"]: item for item in load_json(path)}


def error_list(root: Path, split: str) -> list[dict[str, object]]:
    s1 = [load_predictions(root, "s1", seed, split) for seed in SEEDS]
    b2 = [load_predictions(root, "b2", seed, split) for seed in SEEDS]
    b3 = [load_predictions(root, "b3", seed, split) for seed in SEEDS]
    track_ids = sorted(s1[0])
    rows = []
    for track_id in track_ids:
        s1_mae = float(np.mean([s1[index][track_id]["metrics"]["mae_mean"] for index in range(3)]))
        b2_mae = float(np.mean([b2[index][track_id]["metrics"]["mae_mean"] for index in range(3)]))
        b3_mae = float(np.mean([b3[index][track_id]["metrics"]["mae_mean"] for index in range(3)]))
        rows.append({
            "track_id": track_id,
            "s1_mae_mean": s1_mae,
            "b2_mae_mean": b2_mae,
            "b3_mae_mean": b3_mae,
            "s1_minus_b2_mae": s1_mae - b2_mae,
            "s1_minus_b3_mae": s1_mae - b3_mae,
        })
    return sorted(rows, key=lambda row: row["s1_mae_mean"], reverse=True)


def finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def write_csv(path: Path, summaries: dict[str, dict[str, dict[str, object]]]) -> None:
    fields = ["split", "model"] + [f"{metric}_mean" for metric in METRICS] + [f"{metric}_std" for metric in METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, summary in summaries.items():
            for model, metrics in summary.items():
                row = {"split": split, "model": model}
                for metric in METRICS:
                    row[f"{metric}_mean"] = metrics[metric]["mean"]
                    row[f"{metric}_std"] = metrics[metric]["std"]
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = {split: summarize(args.project_root, split) for split in ("val", "test")}
    errors = {split: error_list(args.project_root, split) for split in ("val", "test")}
    report = {
        "protocol": {
            "unit": "track",
            "splits": {"val": 31, "test": 30},
            "models": list(MODELS),
            "seeds": {model: list(seeds_for(model)) for model in MODELS},
            "test_is_read_only_after_design_freeze": True,
        },
        "summary": summaries,
        "error_list_sorted_by_s1_mae": errors,
        "failure_and_exclusion_status": {"prediction_failures": [], "manifest_exclusions": []},
    }
    if not finite(report):
        raise ValueError("paper report contains non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.output.with_suffix(".csv"), summaries)
    print(f"PAPER_REPORT_OK output={args.output} csv={args.output.with_suffix('.csv')}")
    for split, summary in summaries.items():
        print(f"PAPER {split} " + " ".join(f"{model}={summary[model]['ccc_mean']['mean']:.6f}" for model in MODELS))
    print(f"ERROR_LISTS val={len(errors['val'])} test={len(errors['test'])}")


if __name__ == "__main__":
    main()