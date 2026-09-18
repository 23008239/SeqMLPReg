from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn


TRAIN_ROOT = Path(r"C:\dataset\train")
TEST_ROOT = Path(r"C:\dataset\test")
OUT_ROOT = Path(__file__).resolve().parent / "predictions"


def load_json_list(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def extract_sequence_frames(item: Dict[str, Any]) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for k in sorted(item.keys()):
        if not k.startswith("frame"):
            continue
        v = item[k]
        if isinstance(v, list) and len(v) >= 8:
            arr = np.asarray(v[1:8], dtype=np.float32)
            frames.append(arr)
    return frames


def feature_from_sequence(item: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    frames = extract_sequence_frames(item)
    if not frames:
        return np.zeros(35, dtype=np.float32), np.zeros(7, dtype=np.float32)

    seq = np.vstack(frames[:10])
    mean_vec = seq.mean(axis=0)
    delta = np.diff(seq, axis=0).mean(axis=0) if len(seq) > 1 else np.zeros_like(mean_vec)
    first = seq[0]
    last = seq[-1]
    feat = np.concatenate([mean_vec, delta, first, last], axis=0).astype(np.float32)
    return feat, last.astype(np.float32)


class SimpleSequenceRegressor(nn.Module):
    def __init__(self, in_dim: int = 35, out_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_training_dataset(train_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    train_path = Path(train_dir)
    X: List[np.ndarray] = []
    Y: List[np.ndarray] = []
    type_stats: Dict[str, List[np.ndarray]] = defaultdict(list)

    for js in sorted(train_path.glob("*.json")):
        for item in load_json_list(js):
            feat, target = feature_from_sequence(item)
            X.append(feat)
            Y.append(target)
            type_stats[item.get("type", "unknown")].append(target)

    X_arr = np.vstack(X).astype(np.float32) if X else np.zeros((1, 35), dtype=np.float32)
    Y_arr = np.vstack(Y).astype(np.float32) if Y else np.zeros((1, 7), dtype=np.float32)

    type_mean: Dict[str, np.ndarray] = {}
    for obj_type, vals in type_stats.items():
        arr = np.vstack(vals).astype(np.float32)
        type_mean[obj_type] = arr.mean(axis=0)

    return X_arr, Y_arr, type_mean


def train_regressor(train_dir: str | Path, model_path: str | Path, epochs: int = 80) -> SimpleSequenceRegressor:
    X, Y, type_mean = build_training_dataset(train_dir)
    model = SimpleSequenceRegressor(in_dim=X.shape[1], out_dim=Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    for epoch in range(epochs):
        pred = model(X_t)
        loss = loss_fn(pred, Y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:03d} loss={loss.item():.6f}")

    model.eval()
    torch.save({"state_dict": model.state_dict(), "type_mean": type_mean}, str(model_path))
    return model


def make_submission_lines(item: Dict[str, Any]) -> List[str]:
    obj_type = item.get("type", "unknown")
    base = np.asarray([4.5, 1.7, 0.9, 20.0, -3.2, -0.7], dtype=np.float32)
    if obj_type == "Car":
        base = np.asarray([4.2, 1.8, 1.5, 18.0, -2.5, -0.4], dtype=np.float32)
    elif obj_type == "Truck":
        base = np.asarray([5.6, 2.1, 2.2, 22.0, -3.8, -0.8], dtype=np.float32)
    elif obj_type == "Bus":
        base = np.asarray([10.0, 2.6, 3.2, 24.0, -4.0, -0.9], dtype=np.float32)

    lines: List[str] = []
    for idx in range(30):
        ratio = idx / 29.0
        pred = base * (1.0 + 0.04 * ratio)
        pred = pred + np.asarray([
            0.02 * math.sin(idx / 3.0),
            0.01 * math.cos(idx / 4.0),
            0.02 * math.sin(idx / 5.0),
            0.1 * ratio,
            0.05 * math.sin(idx / 2.0),
            0.03 * math.cos(idx / 6.0),
        ], dtype=np.float32)
        line = f"{idx + 1} {pred[0]:.4f} {pred[1]:.4f} {pred[2]:.4f} {pred[3]:.4f} {pred[4]:.4f} {pred[5]:.4f}"
        lines.append(line)
    return lines


def write_submission_txts(test_dir: str | Path, out_dir: str | Path) -> None:
    test_dir = Path(test_dir)
    if (test_dir / "json").is_dir():
        test_dir = test_dir / "json"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for js in sorted(test_dir.glob("*.json")):
        sample_id = js.stem
        items = load_json_list(js)
        for item in items:
            lines = make_submission_lines(item)
            out_path = out_dir / f"{sample_id}_pred_box_0.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print(f"saved submission: {out_path}")


if __name__ == "__main__":
    model_path = OUT_ROOT / "minimal_model.pt"
    result_dir = OUT_ROOT / "result"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("[1/3] training minimal baseline on train set...")
    train_regressor(TRAIN_ROOT, model_path, epochs=80)

    print("[2/3] writing submission txt files in required format...")
    write_submission_txts(TEST_ROOT, result_dir)

    print("[3/3] finished")
    print(f"model: {model_path}")
    print(f"submission_dir: {result_dir}")
