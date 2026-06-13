#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_metric(path: Path) -> dict:
    if path.exists():
        df = pd.read_csv(path)
        if len(df):
            return df.iloc[0].to_dict()
    summary = path.with_name("summary.json")
    if summary.exists():
        data = json.loads(summary.read_text())
        return data.get("final_metrics", {})
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model19 raw force-window with previous clean300 models")
    parser.add_argument("--outdir", default="outputs/model19_raw_force_comparison")
    parser.add_argument("--raw-force", default="training_outputs_clean300_model19_raw_force_epoch100_stride500/test_metrics.csv")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"model": "model12 historical epoch100", "accuracy": 0.8805, "f1_macro": None, "source": "historical clean300 summary"},
        {"model": "model12 historical epoch500", "accuracy": 0.8875, "f1_macro": None, "source": "historical clean300 summary"},
        {"model": "model18 full-contact stride500", "accuracy": 0.8430, "f1_macro": 0.5871, "source": "previous run"},
        {"model": "model18 full-contact stride100", "accuracy": 0.8736, "f1_macro": 0.6441, "source": "previous run"},
        {"model": "model19 proxy stride500", "accuracy": 0.8315, "f1_macro": 0.5472, "source": "previous run"},
    ]
    metric = read_metric(Path(args.raw_force))
    if metric:
        rows.append({
            "model": "model19 raw-force stride500",
            "accuracy": float(metric.get("accuracy", float("nan"))),
            "f1_macro": float(metric.get("f1_macro", float("nan"))),
            "source": str(args.raw_force),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "model19_raw_force_comparison.csv", index=False)

    plot_df = df.dropna(subset=["accuracy"]).copy()
    plt.figure(figsize=(9, 4.8))
    colors = ["#777777" if "historical" in m else "#4C78A8" if "model18" in m else "#F58518" for m in plot_df["model"]]
    bars = plt.bar(range(len(plot_df)), plot_df["accuracy"], color=colors, edgecolor="black", linewidth=0.5)
    plt.xticks(range(len(plot_df)), plot_df["model"], rotation=30, ha="right")
    plt.ylabel("Top-1 DSSP accuracy")
    plt.ylim(0.70, 0.92)
    plt.title("Clean300 DSSP prediction: model12 vs contact-map vs raw force-window")
    for bar, val in zip(bars, plot_df["accuracy"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig_path = outdir / "model19_raw_force_vs_previous_models.png"
    plt.savefig(fig_path, dpi=300)
    print(df.to_string(index=False))
    print(fig_path)


if __name__ == "__main__":
    main()
