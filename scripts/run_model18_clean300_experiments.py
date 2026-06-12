#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_model18_clean300_baseline.py"


EXPERIMENTS = [
    {
        "name": "epoch100_stride500",
        "outdir": "training_outputs_clean300_epoch100_stride500",
        "epochs": 100,
        "disable": [],
    },
    {
        "name": "epoch300_stride500",
        "outdir": "training_outputs_clean300_epoch300_stride500",
        "epochs": 300,
        "disable": [],
    },
    {
        "name": "epoch500_stride500",
        "outdir": "training_outputs_clean300_epoch500_stride500",
        "epochs": 500,
        "disable": [],
    },
    {
        "name": "ablation_no_dynamic_epoch100",
        "outdir": "training_outputs_clean300_ablation_no_dynamic_epoch100",
        "epochs": 100,
        "disable": ["dynamic"],
    },
    {
        "name": "ablation_no_contact_features_epoch100",
        "outdir": "training_outputs_clean300_ablation_no_contact_features_epoch100",
        "epochs": 100,
        "disable": ["contact_features"],
    },
    {
        "name": "ablation_no_contact_map_stats_epoch100",
        "outdir": "training_outputs_clean300_ablation_no_contact_map_stats_epoch100",
        "epochs": 100,
        "disable": ["contact_map_stats"],
    },
    {
        "name": "ablation_no_contact_all_epoch100",
        "outdir": "training_outputs_clean300_ablation_no_contact_all_epoch100",
        "epochs": 100,
        "disable": ["contact_features", "contact_map_stats"],
    },
    {
        "name": "ablation_no_residue_features_epoch100",
        "outdir": "training_outputs_clean300_ablation_no_residue_features_epoch100",
        "epochs": 100,
        "disable": ["residue_features"],
    },
    {
        "name": "ablation_no_position_epoch100",
        "outdir": "training_outputs_clean300_ablation_no_position_epoch100",
        "epochs": 100,
        "disable": ["position"],
    },
]


def has_metrics(outdir: Path) -> bool:
    return (outdir / "test_metrics.csv").exists() and (outdir / "summary.json").exists()


def run_one(exp: dict, sample_stride: int, force: bool) -> None:
    outdir = ROOT / exp["outdir"]
    if has_metrics(outdir) and not force:
        print(f"[skip] {exp['name']} already has metrics: {outdir}")
        return
    cmd = [
        sys.executable,
        str(TRAIN),
        "--epochs",
        str(exp["epochs"]),
        "--sample-stride",
        str(sample_stride),
        "--outdir",
        str(outdir),
    ]
    for name in exp["disable"]:
        cmd.extend(["--disable", name])
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean-300 model18 epoch and ablation experiments")
    parser.add_argument("--sample-stride", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", action="append", default=[], help="Run only experiments whose name contains this text")
    args = parser.parse_args()

    selected = EXPERIMENTS
    if args.only:
        selected = [exp for exp in EXPERIMENTS if any(token in exp["name"] for token in args.only)]
    if not selected:
        raise SystemExit("No experiments selected")
    for exp in selected:
        run_one(exp, sample_stride=args.sample_stride, force=args.force)


if __name__ == "__main__":
    main()
