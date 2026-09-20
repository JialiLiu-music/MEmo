#!/usr/bin/env python3
"""Train frozen-MERT baselines and the Transformer S1-v2 structure model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

FEATURE_DIM = 768
MODELS = {"b0", "b1", "b2", "b3", "s1", "b2_transformer", "s1_v2", "s1_v2_equal", "s1_v2_random", "b2_transformer_matched"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean()
    target_mean = target.mean()
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    variance_sum = pred.var(unbiased=False) + target.var(unbiased=False)
    denominator = variance_sum + (pred_mean - target_mean).square() + 1e-8
    return 1.0 - 2.0 * covariance / denominator


def average_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("no per-track metrics to aggregate")
    return {
        key: float(np.mean([item[key] for item in values]))
        for key in values[0]
    }


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, name in enumerate(("valence", "arousal")):
        p, y = pred[:, index], target[:, index]
        covariance = np.mean((p - p.mean()) * (y - y.mean()))
        denominator = np.var(p) + np.var(y) + (p.mean() - y.mean()) ** 2 + 1e-8
        result[f"{name}_ccc"] = float(2 * covariance / denominator)
        result[f"{name}_mae"] = float(np.mean(np.abs(p - y)))
        result[f"{name}_rmse"] = float(np.sqrt(np.mean((p - y) ** 2)))
    result["ccc_mean"] = (result["valence_ccc"] + result["arousal_ccc"]) / 2
    result["mae_mean"] = (result["valence_mae"] + result["arousal_mae"]) / 2
    result["rmse_mean"] = (result["valence_rmse"] + result["arousal_rmse"]) / 2
    return result


@dataclass
class Track:
    track_id: str
    split: str
    features: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor
    segment_ids: torch.Tensor
    segment_count: int
    times: np.ndarray


def load_tracks(project_root: Path, manifest_path: Path, aligned_path: Path, feature_dir: Path) -> list[Track]:
    manifest = {row["track_id"]: row for row in read_csv(manifest_path)}
    grouped: dict[str, list[dict[str, str]]] = {}
    # 保留所有对齐时间点作为上下文；valid 只在损失和评价阶段生效。
    for row in read_csv(aligned_path):
        grouped.setdefault(row["track_id"], []).append(row)
    tracks: list[Track] = []
    for track_id, manifest_row in manifest.items():
        if track_id not in grouped:
            continue
        rows = grouped[track_id]
        with np.load(feature_dir / f"{track_id}.npz") as payload:
            all_features = payload["features"].astype(np.float32)
            all_times = payload["time_sec"].astype(np.float64)
        if all_features.shape != (len(all_times), FEATURE_DIM) or not np.isfinite(all_features).all() or not np.all(np.diff(all_times) > 0):
            raise ValueError(f"invalid MERT array: {track_id}")
        targets = np.zeros((len(all_times), 2), dtype=np.float32)
        valid_mask = np.zeros(len(all_times), dtype=bool)
        segment_ids = np.full(len(all_times), -1, dtype=np.int64)
        seen = set()
        for row in rows:
            matches = np.flatnonzero(np.isclose(all_times, float(row["time_sec"]), atol=1e-6, rtol=0))
            if len(matches) == 0:
                # 标签表可能比 MERT 的 45 秒时间轴多一个尾部采样点。
                continue
            if len(matches) != 1 or int(matches[0]) in seen:
                raise ValueError(f"invalid/duplicate alignment: {track_id} {row['time_sec']}")
            index = int(matches[0])
            seen.add(index)
            if row.get("valid", "0").strip() == "1":
                targets[index] = [float(row["valence"]), float(row["arousal"])]
                valid_mask[index] = True
            if row.get("segment_index", "").strip():
                segment_ids[index] = int(row["segment_index"])
        if manifest_row.get("structure_path", "").strip():
            segments = json.loads((project_root / manifest_row["structure_path"]).read_text())["segments"]
            segment_ids[:] = -1
            for index, segment in enumerate(segments):
                selected = (all_times >= float(segment["start"])) & (all_times < float(segment["end"]))
                if np.any(segment_ids[selected] >= 0):
                    raise ValueError(f"overlapping structure: {track_id}")
                segment_ids[selected] = index
        # All-In-One 边界可能留有极短空隙；将空隙分配给最近的已知段，避免丢弃秒级上下文。
        if np.any(segment_ids < 0):
            known = np.flatnonzero(segment_ids >= 0)
            if len(known) == 0:
                raise ValueError(f"structure has no covered MERT frame: {track_id}")
            for index in np.flatnonzero(segment_ids < 0):
                segment_ids[index] = segment_ids[known[np.argmin(np.abs(known - index))]]
        if not valid_mask.any() or not np.isfinite(targets).all():
            raise ValueError(f"missing/nonfinite supervision: {track_id}")
        _, segment_ids = np.unique(segment_ids, return_inverse=True)
        tracks.append(Track(
            track_id=track_id, split=manifest_row["split"],
            features=torch.from_numpy(all_features), targets=torch.from_numpy(targets),
            valid_mask=torch.from_numpy(valid_mask), segment_ids=torch.from_numpy(segment_ids),
            segment_count=int(segment_ids.max()) + 1, times=all_times,
        ))
    if not tracks:
        raise ValueError("no valid tracks loaded")
    return tracks


def padded_local(features: torch.Tensor, radius: int = 2) -> torch.Tensor:
    padded = torch.cat([features[:1].repeat(radius, 1), features, features[-1:].repeat(radius, 1)])
    return torch.cat([padded[index : index + features.shape[0]] for index in range(2 * radius + 1)], dim=1)


class LocalModel(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM * 5, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, features: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.net(padded_local(features))


class FlatGRU(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(FEATURE_DIM, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 2)

    def forward(self, features: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        values, _ = self.gru(features.unsqueeze(0))
        return self.head(values[0])


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-np.log(10000.0) / dim))
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(positions * scale)
        encoding[:, 1::2] = torch.cos(positions * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.shape[1]]


class TransformerFrame(nn.Module):
    """逐秒 Transformer 基线，输入完整的有效时间序列。"""
    def __init__(self, hidden: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        self.projection = nn.Linear(FEATURE_DIM, hidden)
        self.position = PositionalEncoding(hidden)
        block = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Linear(hidden, 2)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        values = self.position(self.projection(features).unsqueeze(0))
        return self.encoder(values)[0]

    def forward(self, features: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(features))


class S1V2(nn.Module):
    """逐秒 Transformer + 段落池化 + 段落序列上下文的简单拼接版本。"""
    def __init__(self, hidden: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        self.frame = TransformerFrame(hidden, layers, heads)
        self.frame.head = nn.Identity()
        self.segment_position = PositionalEncoding(hidden)
        self.boundary_mode = "auto"
        segment_block = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.segment_encoder = nn.TransformerEncoder(segment_block, 1)
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 2))

    def forward(self, features: torch.Tensor, segment_ids: torch.Tensor, segment_count: int = 0, **_: torch.Tensor) -> torch.Tensor:
        frame_values = self.frame.encode(features)
        count = max(1, int(segment_count))
        ids = segment_ids.clamp(0, count - 1)
        if self.boundary_mode == "equal":
            ids = torch.arange(len(features), device=features.device) * count // len(features)
        elif self.boundary_mode == "none":
            return self.head(torch.cat([frame_values, frame_values, self.segment_encoder(self.segment_position(frame_values.unsqueeze(0)))[0]], dim=1))
        pooled = torch.stack([frame_values[ids == index].mean(0) if torch.any(ids == index) else frame_values.mean(0) for index in range(count)])
        context = self.segment_encoder(self.segment_position(pooled.unsqueeze(0)))[0]
        return self.head(torch.cat([frame_values, pooled[ids], context[ids]], dim=1))


class HierarchicalModel(nn.Module):
    def __init__(self, use_auto_segments: bool, hidden: int = 128):
        super().__init__()
        self.use_auto_segments = use_auto_segments
        self.segment_gru = nn.GRU(FEATURE_DIM, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(FEATURE_DIM + FEATURE_DIM + hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, features: torch.Tensor, segment_ids: torch.Tensor, segment_count: int = 0, **_: torch.Tensor) -> torch.Tensor:
        count = max(1, segment_count) if self.use_auto_segments else max(1, segment_count)
        ids = segment_ids.clone()
        if not self.use_auto_segments:
            ids = torch.div(torch.arange(features.shape[0], device=features.device) * count, features.shape[0], rounding_mode="floor")
        ids = ids.clamp(0, count - 1)
        pooled = []
        for index in range(count):
            selected = features[ids == index]
            pooled.append(selected.mean(dim=0) if len(selected) else features.mean(dim=0))
        segment_features = torch.stack(pooled).unsqueeze(0)
        context, _ = self.segment_gru(segment_features)
        local_segment = segment_features[0][ids]
        local_context = context[0][ids]
        return self.head(torch.cat([features, local_segment, local_context], dim=1))


def make_model(model_name: str) -> nn.Module:
    if model_name == "b1":
        return LocalModel()
    if model_name == "b2":
        return FlatGRU()
    if model_name == "b3":
        return HierarchicalModel(False)
    if model_name == "s1":
        return HierarchicalModel(True)
    if model_name == "b2_transformer":
        return TransformerFrame()
    if model_name in {"s1_v2", "s1_v2_equal", "s1_v2_random", "b2_transformer_matched"}:
        model = S1V2()
        model.boundary_mode = {"s1_v2_equal": "equal", "b2_transformer_matched": "none"}.get(model_name, "auto")
        return model
    raise ValueError(f"model {model_name} is not trainable")


def track_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ccc_loss(prediction[:, 0], target[:, 0]) + ccc_loss(prediction[:, 1], target[:, 1]) + 0.5 * torch.mean(torch.abs(prediction - target))


def predict(model: nn.Module, track: Track, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        output = model(track.features.to(device), segment_ids=track.segment_ids.to(device), segment_count=track.segment_count)
    return output.cpu().numpy()


def evaluate(model: nn.Module, tracks: list[Track], split: str, device: torch.device) -> tuple[dict[str, float], list[dict[str, object]]]:
    predictions = []
    targets = []
    per_track = []
    for track in tracks:
        if track.split != split:
            continue
        pred = predict(model, track, device)
        target = track.targets.numpy()
        mask = track.valid_mask.numpy()
        pred_eval, target_eval = pred[mask], target[mask]
        predictions.append(pred_eval)
        targets.append(target_eval)
        per_track.append({"track_id": track.track_id, "split": split, "times": track.times.tolist(), "target": target.tolist(), "prediction": pred.tolist(), "valid_mask": mask.tolist(), "metrics": metrics(pred_eval, target_eval)})
    if not predictions:
        raise ValueError(f"no tracks for split {split}")
    return average_metrics([item["metrics"] for item in per_track]), per_track


def evaluate_existing(args: argparse.Namespace) -> None:
    tracks = load_tracks(args.project_root, args.manifest, args.aligned_samples, args.feature_dir)
    metrics_path = args.run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing run metrics: {metrics_path}")
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_name = run_metrics["model"]
    device = torch.device(args.device)
    if model_name == "b0":
        mean = np.asarray(run_metrics["train_mean"], dtype=np.float32)
        test_predictions = []
        for track in tracks:
            if track.split != "test":
                continue
            prediction = np.tile(mean, (len(track.targets), 1))
            mask = track.valid_mask.numpy()
            test_predictions.append({
                "track_id": track.track_id,
                "split": "test",
                "times": track.times.tolist(),
                "target": track.targets.numpy().tolist(),
                "prediction": prediction.tolist(),
                "valid_mask": mask.tolist(),
                "metrics": metrics(prediction[mask], track.targets.numpy()[mask]),
            })
    else:
        checkpoint_path = args.run_dir / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        model = make_model(model_name).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        _, test_predictions = evaluate(model, tracks, "test", device)
    if not test_predictions:
        raise ValueError("no test tracks available")
    test_metrics = average_metrics([item["metrics"] for item in test_predictions])
    result = {
        "model": model_name,
        "seed": run_metrics["seed"],
        "source_run_dir": str(args.run_dir),
        "split": "test",
        "test_metrics": test_metrics,
        "config": {
            "evaluate_only": True,
            "source_best_epoch": run_metrics.get("best_epoch"),
            "source_evaluate_test": run_metrics.get("config", {}).get("evaluate_test", False),
        },
    }
    (args.run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.run_dir / "test_predictions.json").write_text(json.dumps(test_predictions), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    if args.evaluate_only:
        evaluate_existing(args)
        return
    seed_everything(args.seed)
    device = torch.device(args.device)
    tracks = load_tracks(args.project_root, args.manifest, args.aligned_samples, args.feature_dir)
    if args.max_tracks:
        train_tracks = [track for track in tracks if track.split == "train"]
        val_tracks = [track for track in tracks if track.split == "val"]
        train_limit = max(1, int(round(args.max_tracks * 0.8)))
        tracks = train_tracks[:train_limit] + val_tracks[: max(1, args.max_tracks - train_limit)]
    model_name = args.model.lower()
    if model_name == "s1_v2_random":
        for track in tracks:
            seed = int.from_bytes(hashlib.sha256(track.track_id.encode()).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            cuts = np.sort(rng.choice(np.arange(1, len(track.features)), track.segment_count - 1, replace=False))
            track.segment_ids = torch.from_numpy(np.searchsorted(cuts, np.arange(len(track.features)), side="right"))
    if model_name == "b0":
        train_targets = torch.cat([track.targets[track.valid_mask] for track in tracks if track.split == "train"])
        mean = train_targets.mean(dim=0).numpy()
        val_predictions = []
        for track in tracks:
            if track.split != "val":
                continue
            prediction = np.tile(mean, (len(track.targets), 1))
            mask = track.valid_mask.numpy()
            val_predictions.append({
                "track_id": track.track_id,
                "split": "val",
                "times": track.times.tolist(),
                "target": track.targets.numpy().tolist(),
                "prediction": prediction.tolist(),
                "valid_mask": mask.tolist(),
                "metrics": metrics(prediction[mask], track.targets.numpy()[mask]),
            })
        args.run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        result = {
            "model": "b0",
            "seed": args.seed,
            "best_epoch": None,
            "train_mean": mean.tolist(),
            "val_metrics": average_metrics([item["metrics"] for item in val_predictions]),
            "test_metrics": None,
            "history": [],
            "config": config,
        }
        (args.run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (args.run_dir / "val_predictions.json").write_text(json.dumps(val_predictions), encoding="utf-8")
        (args.run_dir / "baseline.json").write_text(json.dumps({"model": "b0", "seed": args.seed, "train_mean": mean.tolist()}, indent=2), encoding="utf-8")
        return

    model = make_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best = -float("inf")
    stale = 0
    args.run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for track in tracks:
            if track.split != "train":
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = model(track.features.to(device), segment_ids=track.segment_ids.to(device), segment_count=track.segment_count)
            mask = track.valid_mask.to(device)
            if not torch.any(mask):
                continue
            loss = track_loss(prediction[mask], track.targets.to(device)[mask])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, track {track.track_id}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics, _ = evaluate(model, tracks, "val", device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_metrics": val_metrics})
        score = val_metrics["ccc_mean"]
        if score > best:
            best = score
            stale = 0
            torch.save({"model": model.state_dict(), "model_name": model_name, "seed": args.seed, "epoch": epoch}, args.run_dir / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    checkpoint = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_metrics, val_predictions = evaluate(model, tracks, "val", device)
    test_metrics = None
    test_predictions: list[dict[str, object]] = []
    if args.evaluate_test and any(track.split == "test" for track in tracks):
        test_metrics, test_predictions = evaluate(model, tracks, "test", device)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["trainable_params"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = {"model": model_name, "seed": args.seed, "best_epoch": checkpoint["epoch"], "val_metrics": val_metrics, "test_metrics": test_metrics, "history": history, "config": config}
    (args.run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.run_dir / "val_predictions.json").write_text(json.dumps(val_predictions), encoding="utf-8")
    if test_predictions:
        (args.run_dir / "test_predictions.json").write_text(json.dumps(test_predictions), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--aligned-samples", type=Path, default=Path("data/aligned_samples.csv"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features_mert95m"))
    parser.add_argument("--model", choices=sorted(MODELS), default="b2")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--evaluate-test", action="store_true", help="evaluate the frozen test split after design freeze")
    parser.add_argument("--evaluate-only", action="store_true", help="evaluate an existing run on test without retraining")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
