#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model18.full_contact_dataset import FullContactSwitches
from model18.model19_dataset import Clean300Model19WindowDataset
from model18.model19_model import Model19DSSPModel, window_auxiliary_loss
from scripts.train_model18_full_contact_map import contact_aux_loss, evaluate, masked_focal_ce, move_batch


def infer_dims(dataset: Clean300Model19WindowDataset) -> tuple[int, int, int, int]:
    item = dataset[0]
    return (
        int(item["residue_features"].shape[-1]),
        int(item["dynamic"].shape[-1]),
        int(item["contact_features"].shape[-1]),
        int(item["force_window_proxy"].shape[-1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model19/model12-plus-contact-map on clean300")
    parser.add_argument("--manifest", default=str(ROOT / "outputs/preprocess_db_current_clean_300/model18_training_manifest.csv"))
    parser.add_argument("--outdir", default=str(ROOT / "training_outputs_clean300_model19_epoch100_stride500"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sample-stride", type=int, default=500)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--graph-dim", type=int, default=64)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--window-encoder", choices=["transformer", "gru"], default="transformer")
    parser.add_argument("--contact-aux-weight", type=float, default=0.02)
    parser.add_argument("--window-aux-weight", type=float, default=0.0)
    parser.add_argument("--force-window-manifest", default=None, help="Optional manifest from preprocess_model12_force_windows.py")
    parser.add_argument("--max-train-frames", type=int, default=None)
    parser.add_argument("--max-test-frames", type=int, default=None)
    parser.add_argument("--disable", action="append", default=[])
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    switches = FullContactSwitches.from_ablation_names(args.disable)
    train_ds = Clean300Model19WindowDataset(
        args.manifest,
        "train",
        args.sample_stride,
        args.window_size,
        args.max_train_frames,
        switches,
        args.seed,
        args.force_window_manifest,
    )
    test_ds = Clean300Model19WindowDataset(
        args.manifest,
        "test",
        args.sample_stride,
        args.window_size,
        args.max_test_frames,
        switches,
        args.seed + 1,
        args.force_window_manifest,
    )
    n_residue, n_dynamic, n_contact, n_window = infer_dims(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model19DSSPModel(
        n_residue_features=n_residue,
        n_dynamic=n_dynamic,
        n_contact_features=n_contact,
        n_window_features=n_window,
        hidden=args.hidden,
        graph_dim=args.graph_dim,
        top_n=args.top_n,
        dropout=args.dropout,
        window_encoder=args.window_encoder,
        window_aux_weight=args.window_aux_weight,
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
            loss = loss + window_auxiliary_loss(model, batch)
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
                torch.save({"model_state": model.state_dict(), "args": vars(args), "feature_switches": switches.to_dict(), "best_metrics": best}, outdir / "model19_model12_plus_contact_best.pt")
    final_metrics = evaluate(model, test_loader, device)
    final_metrics.update({"epoch": args.epochs})
    torch.save({"model_state": model.state_dict(), "args": vars(args), "feature_switches": switches.to_dict(), "final_metrics": final_metrics}, outdir / "model19_model12_plus_contact_final.pt")
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    pd.DataFrame([final_metrics]).to_csv(outdir / "test_metrics.csv", index=False)
    summary = {"train": train_ds.info(), "test": test_ds.info(), "final_metrics": final_metrics, "best_metrics": best, "device": device, "args": vars(args), "model": "Model19DSSPModel"}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("saved", outdir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
