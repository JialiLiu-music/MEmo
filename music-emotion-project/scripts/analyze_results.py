#!/usr/bin/env python3
"""Analyze frozen test predictions with track-level uncertainty and boundary strata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODELS = ("b0", "b1", "b2", "b3", "s1")
SEEDS = (20260903, 20260904, 20260905)
METRICS = ("valence_ccc", "arousal_ccc", "ccc_mean", "valence_mae", "arousal_mae", "mae_mean", "valence_rmse", "arousal_rmse", "rmse_mean")
STRATA = ((0.0, 1.0, "0-1s"), (1.0, 3.0, "1-3s"), (3.0, 5.0, "3-5s"), (5.0, float("inf"), ">5s"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def track_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, name in enumerate(("valence", "arousal")):
        pred = prediction[:, index]
        truth = target[:, index]
        covariance = np.mean((pred - pred.mean()) * (truth - truth.mean()))
        denominator = np.var(pred) + np.var(truth) + (pred.mean() - truth.mean()) ** 2 + 1e-8
        result[f"{name}_ccc"] = float(2.0 * covariance / denominator)
        result[f"{name}_mae"] = float(np.mean(np.abs(pred - truth)))
        result[f"{name}_rmse"] = float(np.sqrt(np.mean((pred - truth) ** 2)))
    result["ccc_mean"] = (result["valence_ccc"] + result["arousal_ccc"]) / 2.0
    result["mae_mean"] = (result["valence_mae"] + result["arousal_mae"]) / 2.0
    result["rmse_mean"] = (result["valence_rmse"] + result["arousal_rmse"]) / 2.0
    return result


def load_structure_boundaries(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    starts = sorted({float(segment["start"]) for segment in segments if "start" in segment})
    return np.asarray(starts[1:], dtype=np.float64)


def boundary_stratum(distance: float) -> str:
    for lower, upper, label in STRATA:
        if lower <= distance < upper:
            return label
    raise ValueError(f"distance outside strata: {distance}")


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> dict[str, float]:
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "n_tracks": int(len(values)),
        "bootstrap_repetitions": repetitions,
    }


def paired_permutation(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> dict[str, float]:
    observed = float(values.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(repetitions, len(values)))
    null = (signs * values).mean(axis=1)
    p_value = float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (repetitions + 1))
    return {"mean_difference": observed, "p_two_sided": p_value, "n_tracks": int(len(values)), "permutations": repetitions}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    previous = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        corrected = min(1.0, (total - rank) * value)
        corrected = max(previous, corrected)
        adjusted[name] = corrected
        previous = corrected
    return adjusted


def load_exclusions(root: Path) -> list[dict[str, str]]:
    manifest_path = root / "data/manifest.csv"
    if not manifest_path.exists():
        return []
    exclusions = []
    for row in read_rows(manifest_path):
        quality = row.get("quality_status", "").strip().lower()
        reason = row.get("exclude_reason", "").strip()
        if quality == "excluded" or reason:
            exclusions.append({
                "track_id": row.get("track_id", ""),
                "quality_status": row.get("quality_status", ""),
                "exclude_reason": reason,
            })
    return exclusions


def track_failures(predictions: dict[tuple[str, int], dict[str, dict[str, object]]], track_ids: list[str]) -> list[dict[str, object]]:
    failures = []
    for track_id in track_ids:
        reasons = []
        for key, items in predictions.items():
            item = items[track_id]
            prediction = np.asarray(item["prediction"], dtype=np.float64)
            target = np.asarray(item["target"], dtype=np.float64)
            if not np.isfinite(prediction).all() or not np.isfinite(target).all():
                reasons.append(f"non_finite:{key[0]}_seed{key[1]}")
        if reasons:
            failures.append({"track_id": track_id, "reasons": reasons})
    return failures


def load_predictions(root: Path, model: str, seed: int) -> dict[str, dict[str, object]]:
    path = root / f"runs/formal-{model}-seed{seed}-valonly/test_predictions.json"
    return {item["track_id"]: item for item in json.loads(path.read_text(encoding="utf-8"))}


def load_all_predictions(root: Path) -> dict[tuple[str, int], dict[str, dict[str, object]]]:
    result = {}
    for model in MODELS:
        seeds = (20260903,) if model == "b0" else SEEDS
        for seed in seeds:
            result[(model, seed)] = load_predictions(root, model, seed)
    return result


def validate_alignment(predictions: dict[tuple[str, int], dict[str, dict[str, object]]]) -> list[str]:
    reference = predictions[("b0", 20260903)]
    track_ids = sorted(reference)
    for key, items in predictions.items():
        if set(items) != set(track_ids):
            raise ValueError(f"track set mismatch: {key}")
        for track_id in track_ids:
            item = items[track_id]
            if item["split"] != "test" or not (len(item["times"]) == len(item["target"]) == len(item["prediction"] ) == 30):
                raise ValueError(f"invalid prediction shape: {key} {track_id}")
    return track_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    predictions = load_all_predictions(args.project_root)
    track_ids = validate_alignment(predictions)
    structure_boundaries = {
        track_id: load_structure_boundaries(args.structure_dir / f"{track_id}.json")
        for track_id in track_ids
    }
    rng = np.random.default_rng(args.seed)
    per_track: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for key, items in predictions.items():
        per_track[key] = {}
        for track_id in track_ids:
            item = items[track_id]
            prediction = np.asarray(item["prediction"], dtype=np.float64)
            target = np.asarray(item["target"], dtype=np.float64)
            per_track[key][track_id] = track_metrics(prediction, target)

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODELS:
        seeds = (20260903,) if model == "b0" else SEEDS
        summary[model] = {}
        for metric in METRICS:
            values = np.asarray([np.mean([per_track[(model, seed)][track_id][metric] for seed in seeds]) for track_id in track_ids])
            summary[model][metric] = bootstrap_mean(values, rng, args.bootstrap)
            if len(seeds) > 1:
                summary[model][metric]["seed_values"] = [float(np.mean([per_track[(model, seed)][track_id][metric] for track_id in track_ids])) for seed in seeds]

    paired: dict[str, dict[str, dict[str, float]]] = {}
    for comparison in ("b2", "b3"):
        paired[f"s1_minus_{comparison}"] = {}
        for metric in METRICS:
            s1 = np.asarray([np.mean([per_track[("s1", seed)][track_id][metric] for seed in SEEDS]) for track_id in track_ids])
            baseline = np.asarray([np.mean([per_track[(comparison, seed)][track_id][metric] for seed in SEEDS]) for track_id in track_ids])
            difference = s1 - baseline
            paired[f"s1_minus_{comparison}"][metric] = {
                **paired_permutation(difference, rng, args.permutations),
                **bootstrap_mean(difference, rng, args.bootstrap),
            }

    primary_p_values = {
        comparison + ":" + metric: paired["s1_minus_" + comparison][metric]["p_two_sided"]
        for comparison in ("b2", "b3")
        for metric in ("valence_ccc", "arousal_ccc")
    }
    holm_values = holm_adjust(primary_p_values)
    for name, value in holm_values.items():
        comparison, metric = name.split(":", 1)
        paired["s1_minus_" + comparison][metric]["p_holm"] = value

    exclusions = load_exclusions(args.project_root)
    failures = track_failures(predictions, track_ids)

    boundary: dict[str, dict[str, dict[str, object]]] = {}
    for model in ("b2", "b3", "s1"):
        boundary[model] = {}
        seeds = SEEDS
        for _, _, label in STRATA:
            boundary[model][label] = {"n_track_seed_cells": 0, "n_timepoints": 0, "mae": {}, "rmse": {}}
        for track_id in track_ids:
            boundaries = structure_boundaries[track_id]
            times = np.asarray(predictions[(model, SEEDS[0])][track_id]["times"], dtype=np.float64)
            distances = np.min(np.abs(times[:, None] - boundaries[None, :]), axis=1) if len(boundaries) else np.full(len(times), np.nan)
            for _, _, label in STRATA:
                cell_errors = []
                for seed in seeds:
                    item = predictions[(model, seed)][track_id]
                    error = np.asarray(item["prediction"], dtype=np.float64) - np.asarray(item["target"], dtype=np.float64)
                    mask = np.asarray([boundary_stratum(float(distance)) == label for distance in distances])
                    if not np.any(mask):
                        continue
                    cell_errors.append(error[mask])
                    boundary[model][label]["n_track_seed_cells"] += 1
                    boundary[model][label]["n_timepoints"] += int(mask.sum())
                if cell_errors:
                    values = np.concatenate(cell_errors, axis=0)
                    boundary[model][label]["mae"].setdefault("values", []).append(float(np.mean(np.abs(values))))
                    boundary[model][label]["rmse"].setdefault("values", []).append(float(np.sqrt(np.mean(values**2))))
        for label in boundary[model]:
            for metric in ("mae", "rmse"):
                values = boundary[model][label][metric].pop("values", [])
                boundary[model][label][metric] = {
                    "track_mean": float(np.mean(values)) if values else None,
                    "n_tracks": len(values),
                }

    report = {
        "protocol": {
            "unit": "track",
            "test_tracks": len(track_ids),
            "seeds": {model: ([20260903] if model == "b0" else list(SEEDS)) for model in MODELS},
            "bootstrap_repetitions": args.bootstrap,
            "permutation_repetitions": args.permutations,
            "random_seed": args.seed,
            "boundary_definition": "distance from each test timestamp to the nearest internal All-In-One segment boundary",
            "boundary_strata": [label for _, _, label in STRATA],
            "structure_is_automatic_input_not_ground_truth": True,
        },
        "track_level_summary": summary,
        "paired_differences": paired,
        "multiple_comparison_correction": {
            "method": "Holm step-down",
            "family": "S1 versus B2/B3 x valence_ccc/arousal_ccc",
            "adjusted_p_values": holm_values,
        },
        "boundary_error": boundary,
        "failures": failures,
        "exclusions": exclusions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ANALYSIS_OK tracks={len(track_ids)} output={args.output}")
    for model in MODELS:
        print(f"CCC {model} {summary[model]['ccc_mean']['estimate']:.6f}")
    for comparison in ("b2", "b3"):
        print(f"PAIRED s1_minus_{comparison} ccc_mean={paired[f's1_minus_{comparison}']['ccc_mean']['mean_difference']:.6f} p={paired[f's1_minus_{comparison}']['ccc_mean']['p_two_sided']:.6f}")


if __name__ == "__main__":
    main()