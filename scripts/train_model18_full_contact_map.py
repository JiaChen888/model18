#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model18.full_contact_dataset import Clean300FullContactFrameDataset, FullContactSwitches
from model18.full_contact_map_model import FullContactMapDSSPModel


def move_batch(batch: dict, device: str) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def masked_focal_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, gamma: float = 1.2, label_smoothing: float = 0.05) -> torch.Tensor:
    z = logits.reshape(-1, 8)
    y = target.reshape(-1).long()
    m = mask.reshape(-1).bool()
    if not m.any():
        return z.sum() * 0.0
    z = z[m]
    y = y[m]
    counts = torch.bincount(y, minlength=8).float().to(z.device)
    weights = counts.sum().clamp_min(1.0) / (8.0 * counts.clamp_min(1.0))
    weights = (weights / weights.mean().clamp_min(1e-6)).clamp(0.35, 3.5)
    log_prob = F.log_softmax(z, dim=-1)
    prob = log_prob.exp()
    if label_smoothing > 0:
        true_dist = torch.zeros_like(log_prob).fill_(label_smoothing / 7.0)
        true_dist.scatter_(1, y.unsqueeze(1), 1.0 - label_smoothing)
        ce = -(true_dist * log_prob).sum(dim=-1)
        pt = (true_dist * prob).sum(dim=-1).clamp_min(1e-6)
    else:
        ce = F.nll_loss(log_prob, y, reduction="none")
        pt = prob.gather(1, y.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
    return (weights[y] * (1.0 - pt).pow(gamma) * ce).mean()


def contact_aux_loss(model: FullContactMapDSSPModel, batch: dict, weight: float) -> torch.Tensor:
    if weight <= 0 or not getattr(model, "last_aux", None):
        return batch["target"].float().sum() * 0.0
    cmap = batch["contact_map"].float().clamp_min(0.0)
    mask = batch["mask"].float()
    valid = (cmap > 0).float()
    max_val = cmap.detach().amax() if cmap.numel() else torch.tensor(0.0, device=cmap.device)
    if max_val > 1.5:
        strength = torch.exp(-cmap / 0.6) * valid
    else:
        strength = cmap * valid
    n = strength.shape[-1]
    eye = torch.eye(n, device=strength.device, dtype=torch.bool).unsqueeze(0)
    strength = strength.masked_fill(eye, 0.0)
    target_number = strength.sum(dim=-1) / max(n - 1, 1)
    target_density = strength.mean(dim=-1)
    aux = model.last_aux
    v = mask.bool()
    if not v.any():
        return strength.sum() * 0.0
    loss = F.smooth_l1_loss(torch.sigmoid(aux["contact_number"])[v], target_number[v].clamp(0, 1))
    loss = loss + F.smooth_l1_loss(torch.sigmoid(aux["contact_density"])[v], target_density[v].clamp(0, 1))
    return weight * loss


@torch.no_grad()
def evaluate(model: FullContactMapDSSPModel, loader: DataLoader, device: str) -> dict:
    model.eval()
    pred_all = []
    true_all = []
    conf_all = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch)
        prob = torch.softmax(logits, dim=-1)
        mask = batch["mask"].bool()
        pred_all.append(prob.argmax(dim=-1)[mask].detach().cpu().numpy())
        true_all.append(batch["target"][mask].detach().cpu().numpy())
        conf_all.append(prob.max(dim=-1).values[mask].detach().cpu().numpy())
    y = np.concatenate(true_all) if true_all else np.array([], dtype=int)
    pred = np.concatenate(pred_all) if pred_all else np.array([], dtype=int)
    conf = np.concatenate(conf_all) if conf_all else np.array([], dtype=float)
    return {
        "accuracy": float(accuracy_score(y, pred)) if len(y) else 0.0,
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)) if len(y) else 0.0,
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)) if len(y) else 0.0,
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)) if len(y) else 0.0,
        "mean_confidence": float(conf.mean()) if len(conf) else 0.0,
        "n": int(len(y)),
    }


def infer_dims(dataset: Clean300FullContactFrameDataset) -> tuple[int, int, int]:
    item = dataset[0]
    return int(item["residue_features"].shape[-1]), int(item["dynamic"].shape[-1]), int(item["contact_features"].shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model18 full contact-map CNN/GNN branch on clean-300")
    parser.add_argument("--manifest", default=str(ROOT / "outputs/preprocess_db_current_clean_300/model18_training_manifest.csv"))
    parser.add_argument("--outdir", default=str(ROOT / "training_outputs_clean300_full_contact_epoch100"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sample-stride", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--graph-dim", type=int, default=64)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--max-train-frames", type=int, default=None)
    parser.add_argument("--max-test-frames", type=int, default=None)
    parser.add_argument("--disable", action="append", default=[])
    parser.add_argument("--contact-aux-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    switches = FullContactSwitches.from_ablation_names(args.disable)
    train_ds = Clean300FullContactFrameDataset(args.manifest, "train", args.sample_stride, args.max_train_frames, switches, args.seed)
    test_ds = Clean300FullContactFrameDataset(args.manifest, "test", args.sample_stride, args.max_test_frames, switches, args.seed + 1)
    n_residue, n_dynamic, n_contact = infer_dims(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FullContactMapDSSPModel(
        n_residue_features=n_residue,
        n_dynamic=n_dynamic,
        n_contact_features=n_contact,
        hidden=args.hidden,
        graph_dim=args.graph_dim,
        top_n=args.top_n,
        dropout=args.dropout,
        **switches.to_dict(),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    history = []
    best = {"accuracy": -1.0, "epoch": 0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            loss = masked_focal_ce(logits, batch["target"], batch["mask"])
            loss = loss + contact_aux_loss(model, batch, args.contact_aux_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            metrics = evaluate(model, test_loader, device)
            metrics.update({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else 0.0})
            history.append(metrics)
            scheduler.step(metrics["f1_macro"])
            print(json.dumps(metrics), flush=True)
            if metrics["accuracy"] > best.get("accuracy", -1.0):
                best = metrics.copy()
                torch.save({"model_state": model.state_dict(), "args": vars(args), "feature_switches": switches.to_dict(), "best_metrics": best}, outdir / "model18_full_contact_map_best.pt")
    final_metrics = evaluate(model, test_loader, device)
    final_metrics.update({"epoch": args.epochs})
    torch.save({"model_state": model.state_dict(), "args": vars(args), "feature_switches": switches.to_dict(), "final_metrics": final_metrics}, outdir / "model18_full_contact_map_final.pt")
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    pd.DataFrame([final_metrics]).to_csv(outdir / "test_metrics.csv", index=False)
    summary = {"train": train_ds.info(), "test": test_ds.info(), "final_metrics": final_metrics, "best_metrics": best, "device": device, "args": vars(args), "model": "FullContactMapDSSPModel"}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("saved", outdir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
