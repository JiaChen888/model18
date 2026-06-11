#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model18.clean300_dataset import FeatureSwitches, build_frame_residue_table


class MLP(nn.Module):
    def __init__(self, in_dim: int, width: int = 256, depth: int = 3, out_dim: int = 8, dropout: float = 0.10):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(last, width), nn.ReLU(), nn.LayerNorm(width), nn.Dropout(dropout)])
            last = width
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: str) -> dict:
    model.eval()
    preds = []
    probs = []
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(axis=1))
    pred = np.concatenate(preds) if preds else np.array([], dtype=int)
    prob = np.vstack(probs) if probs else np.empty((0, 8))
    conf = prob.max(axis=1) if len(prob) else np.array([])
    acc = accuracy_score(y, pred) if len(y) else 0.0
    return {
        "accuracy": float(acc),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)) if len(y) else 0.0,
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)) if len(y) else 0.0,
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)) if len(y) else 0.0,
        "mean_confidence": float(conf.mean()) if len(conf) else 0.0,
        "n": int(len(y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model18 clean-300 frame/residue DSSP baseline")
    parser.add_argument("--manifest", default=str(ROOT / "outputs/preprocess_db_current_clean_300/model18_training_manifest.csv"))
    parser.add_argument("--outdir", default=str(ROOT / "training_outputs_clean300_epoch100"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--disable", action="append", default=[])
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    switches = FeatureSwitches.from_ablation_names(args.disable)

    X_train, y_train, meta_train, info_train = build_frame_residue_table(
        args.manifest, split="train", sample_stride=args.sample_stride, max_rows=args.max_train_rows,
        feature_switches=switches, seed=args.seed,
    )
    X_test, y_test, meta_test, info_test = build_frame_residue_table(
        args.manifest, split="test", sample_stride=args.sample_stride, max_rows=args.max_test_rows,
        feature_switches=switches, seed=args.seed + 1,
    )
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(X_train.shape[1], width=args.width, depth=args.depth, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=args.batch_size, shuffle=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            metrics = evaluate(model, X_test, y_test, args.batch_size, device)
            metrics.update({"epoch": epoch, "train_loss": float(np.mean(losses))})
            history.append(metrics)
            print(json.dumps(metrics))
    final_metrics = evaluate(model, X_test, y_test, args.batch_size, device)
    final_metrics.update({"epoch": args.epochs})

    torch.save({"model_state": model.state_dict(), "mean": mean, "std": std, "args": vars(args), "feature_switches": switches.to_dict()}, outdir / "model18_clean300_baseline.pt")
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    pd.DataFrame([final_metrics]).to_csv(outdir / "test_metrics.csv", index=False)
    meta_train.to_csv(outdir / "train_rows_metadata.csv", index=False)
    meta_test.to_csv(outdir / "test_rows_metadata.csv", index=False)
    summary = {"train": info_train, "test": info_test, "final_metrics": final_metrics, "device": device, "args": vars(args)}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("saved", outdir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
