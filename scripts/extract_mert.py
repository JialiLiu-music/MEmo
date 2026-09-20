#!/usr/bin/env python3
"""Extract frozen MERT-v1-95M representations on a reproducible 1 Hz timeline."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODEL_NAME = "m-a-p/MERT-v1-95M"
TARGET_SAMPLE_RATE = 24_000
DEFAULT_WINDOW_SEC = 10.0
DEFAULT_HOP_SEC = 5.0
DEFAULT_BIN_SEC = 1.0
FEATURE_DIM = 768


def load_audio(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = waveform.mean(axis=1)
    if sample_rate == TARGET_SAMPLE_RATE:
        return mono
    import librosa

    return librosa.resample(mono, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE).astype(np.float32)


def iter_windows(audio: np.ndarray, window_samples: int, hop_samples: int):
    if audio.size == 0:
        raise ValueError("audio is empty")
    if audio.size <= window_samples:
        yield 0, np.pad(audio, (0, window_samples - audio.size))
        return
    last_start = audio.size - window_samples
    starts = list(range(0, last_start + 1, hop_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        yield start, audio[start : start + window_samples]


def layer_average(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if len(hidden_states) < 4:
        raise ValueError(f"expected at least 4 hidden-state tensors, got {len(hidden_states)}")
    return torch.stack(hidden_states[-4:], dim=0).mean(dim=0)


def extract_track(
    model: torch.nn.Module,
    extractor: Wav2Vec2FeatureExtractor,
    audio: np.ndarray,
    device: torch.device,
    window_sec: float,
    hop_sec: float,
    bin_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    if window_sec <= 0 or hop_sec <= 0 or bin_sec <= 0:
        raise ValueError("window, hop and bin durations must be positive")
    duration_sec = audio.size / TARGET_SAMPLE_RATE
    bin_count = math.floor((duration_sec - 0.5 * bin_sec) / bin_sec) + 1
    if bin_count < 1:
        raise ValueError(f"audio is shorter than one complete {bin_sec}s bin")

    window_samples = max(1, round(window_sec * TARGET_SAMPLE_RATE))
    hop_samples = max(1, round(hop_sec * TARGET_SAMPLE_RATE))
    sums = np.zeros((bin_count, FEATURE_DIM), dtype=np.float64)
    counts = np.zeros(bin_count, dtype=np.int64)

    with torch.inference_mode():
        for start_sample, window in iter_windows(audio, window_samples, hop_samples):
            inputs = extractor(
                window,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=False,
            )
            input_values = inputs.input_values.to(device)
            output = model(input_values, output_hidden_states=True)
            frame_features = layer_average(output.hidden_states)[0].float().cpu().numpy()
            frame_count = frame_features.shape[0]
            if frame_count == 0:
                continue
            frame_duration = min(window_sec, (audio.size - start_sample) / TARGET_SAMPLE_RATE)
            frame_times = (start_sample / TARGET_SAMPLE_RATE) + (
                np.arange(frame_count, dtype=np.float64) + 0.5
            ) * frame_duration / frame_count
            bin_indices = np.floor(frame_times / bin_sec).astype(np.int64)
            keep = (bin_indices >= 0) & (bin_indices < bin_count)
            for bin_index in np.unique(bin_indices[keep]):
                selected = frame_features[keep & (bin_indices == bin_index)]
                sums[bin_index] += selected.sum(axis=0, dtype=np.float64)
                counts[bin_index] += selected.shape[0]

    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"no MERT frames for bins: {missing[:10]}")
    features = (sums / counts[:, None]).astype(np.float32)
    times = (np.arange(bin_count, dtype=np.float64) + 0.5) * bin_sec
    return features, times


def extract_one(
    model: torch.nn.Module,
    extractor: Wav2Vec2FeatureExtractor,
    project_root: Path,
    row: dict[str, str],
    output_dir: Path,
    device: torch.device,
    window_sec: float,
    hop_sec: float,
    bin_sec: float,
    overwrite: bool,
) -> dict[str, object]:
    track_id = row["track_id"].strip()
    audio_path = project_root / row["audio_path"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{track_id}.npz"
    metadata_path = output_dir / f"{track_id}.json"
    if output_path.exists() and metadata_path.exists() and not overwrite:
        return {"track_id": track_id, "status": "skipped", "frames": 0}
    audio = load_audio(audio_path)
    started = time.perf_counter()
    features, times = extract_track(model, extractor, audio, device, window_sec, hop_sec, bin_sec)
    np.savez_compressed(output_path, features=features, time_sec=times)
    metadata = {
        "track_id": track_id,
        "audio_path": row["audio_path"],
        "model_name": MODEL_NAME,
        "sample_rate_hz": TARGET_SAMPLE_RATE,
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "bin_sec": bin_sec,
        "timeline_rule": "bin i is [i*bin_sec, (i+1)*bin_sec), timestamp is bin center",
        "layer_rule": "mean of the final four hidden-state tensors",
        "feature_shape": list(features.shape),
        "duration_sec": audio.size / TARGET_SAMPLE_RATE,
        "device": str(device),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"track_id": track_id, "status": "written", "frames": int(features.shape[0])}


def read_manifest(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "track_id" not in rows[0] or "audio_path" not in rows[0]:
        raise ValueError("manifest must contain track_id and audio_path")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--hop-sec", type=float, default=DEFAULT_HOP_SEC)
    parser.add_argument("--bin-sec", type=float, default=DEFAULT_BIN_SEC)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.model_name != MODEL_NAME:
        raise ValueError(f"this frozen contract only permits {MODEL_NAME}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    written = skipped = failed = 0
    for row in read_manifest(args.manifest):
        try:
            result = extract_one(
                model,
                extractor,
                args.project_root,
                row,
                args.output_dir,
                device,
                args.window_sec,
                args.hop_sec,
                args.bin_sec,
                args.overwrite,
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            written += int(result["status"] == "written")
            skipped += int(result["status"] == "skipped")
        except Exception as exc:  # keep the batch auditable and continue other tracks
            failed += 1
            print(json.dumps({"track_id": row.get("track_id", ""), "status": "failed", "error": repr(exc)}, ensure_ascii=False), flush=True)
    print(json.dumps({"written": written, "skipped": skipped, "failed": failed}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()