#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGS = ROOT / "outputs" / "sci_figures"
FIGS.mkdir(parents=True, exist_ok=True)


DSSP_COLORS = {
    "accuracy": "#2F6DB3",
    "f1_macro": "#4C9F70",
    "precision_macro": "#8E5EA2",
    "recall_macro": "#D6862A",
}


def load_result(outdir: Path) -> dict | None:
    metrics_csv = outdir / "test_metrics.csv"
    summary_json = outdir / "summary.json"
    if not metrics_csv.exists() or not summary_json.exists():
        return None
    metrics = pd.read_csv(metrics_csv).iloc[0].to_dict()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    args = summary.get("args", {})
    switches = summary.get("test", {}).get("feature_switches", {})
    disable = args.get("disable", [])
    exp = {
        "run": outdir.name,
        "epochs": int(args.get("epochs", metrics.get("epoch", 0))),
        "sample_stride": int(args.get("sample_stride", 0)),
        "disabled": "+".join(disable) if disable else "none",
        "train_rows": int(summary.get("train", {}).get("rows", 0)),
        "test_rows": int(summary.get("test", {}).get("rows", metrics.get("n", 0))),
        "feature_dim": int(summary.get("test", {}).get("feature_dim", 0)),
        "device": summary.get("device", ""),
    }
    exp.update({k: metrics.get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "mean_confidence", "n", "epoch"]})
    exp.update({f"switch_{k}": v for k, v in switches.items()})
    return exp


def collect() -> pd.DataFrame:
    rows = []
    for outdir in sorted(ROOT.glob("training_outputs_clean300*")):
        row = load_result(outdir)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["disabled", "epochs", "sample_stride", "run"]).reset_index(drop=True)
    return df


def annotate_bars(ax) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if pd.notna(height):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + 0.006,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )


def plot_epoch(df: pd.DataFrame) -> Path | None:
    base = df[(df["disabled"] == "none") & (df["sample_stride"] == 500)].copy()
    if base.empty:
        return None
    base = base.sort_values("epochs")
    x = [str(v) for v in base["epochs"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    width = 0.36
    xs = range(len(base))
    ax.bar([i - width / 2 for i in xs], base["accuracy"], width=width, color=DSSP_COLORS["accuracy"], label="Top-1 accuracy")
    ax.bar([i + width / 2 for i in xs], base["f1_macro"], width=width, color=DSSP_COLORS["f1_macro"], label="Macro F1")
    ax.set_xticks(list(xs), x)
    ax.set_xlabel("Training epochs")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(1.0, float(base[["accuracy", "f1_macro"]].max().max()) + 0.08))
    ax.set_title("model18 clean-300 epoch comparison")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    annotate_bars(ax)
    fig.tight_layout()
    out = FIGS / "model18_clean300_epoch_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_ablation(df: pd.DataFrame) -> Path | None:
    abl = df[(df["epochs"] == 100) & (df["sample_stride"] == 500)].copy()
    if abl.empty:
        return None
    order = [
        "none",
        "dynamic",
        "contact_features",
        "contact_map_stats",
        "contact_features+contact_map_stats",
        "residue_features",
        "position",
    ]
    abl["disabled"] = pd.Categorical(abl["disabled"], categories=order, ordered=True)
    abl = abl.sort_values("disabled")
    labels = [str(v).replace("contact_features+contact_map_stats", "all contact").replace("contact_map_stats", "map stats").replace("contact_features", "contact feat").replace("residue_features", "residue feat") for v in abl["disabled"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = range(len(abl))
    ax.bar(xs, abl["accuracy"], color="#2F6DB3", label="Top-1 accuracy")
    ax.plot(xs, abl["f1_macro"], color="#4C9F70", marker="o", linewidth=2.0, label="Macro F1")
    ax.set_xticks(list(xs), labels, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(1.0, float(abl[["accuracy", "f1_macro"]].max().max()) + 0.08))
    ax.set_title("model18 feature ablation on clean-300")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    annotate_bars(ax)
    fig.tight_layout()
    out = FIGS / "model18_clean300_ablation_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_length_distribution() -> Path | None:
    manifest = ROOT / "outputs" / "preprocess_db_current_clean_300" / "model18_training_manifest.csv"
    if not manifest.exists():
        return None
    df = pd.read_csv(manifest)
    ready = df[df["ready"] == True]
    counts = ready.groupby(["length", "split"]).size().unstack(fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    counts.plot(kind="bar", stacked=True, color=["#6B6B6B", "#D6862A"], ax=ax)
    ax.set_xlabel("PolyQ length")
    ax.set_ylabel("Number of samples")
    ax.set_title("Clean-300 sample composition")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, title="Split")
    fig.tight_layout()
    out = FIGS / "model18_clean300_sample_distribution.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def write_docs(df: pd.DataFrame, figures: list[Path]) -> None:
    csv_path = DOCS / "MODEL18_CLEAN300_EPOCH_AND_ABLATION_RESULTS.csv"
    md_path = DOCS / "MODEL18_CLEAN300_FINAL_RESULTS.md"
    df.to_csv(csv_path, index=False)
    base = df[(df["disabled"] == "none") & (df["sample_stride"] == 500)].sort_values("epochs")
    ablation = df[(df["epochs"] == 100) & (df["sample_stride"] == 500)].copy()
    best = df.sort_values(["accuracy", "f1_macro"], ascending=False).iloc[0]
    lines = [
        "# MODEL18 Clean-300 Final Results",
        "",
        "## Data and Split",
        "",
        "All reported training runs read the fixed clean-300 preprocessed database through `outputs/preprocess_db_current_clean_300/model18_training_manifest.csv`. The database contains 300 ready SMD samples: Q22 = 111, Q36 = 103, Q46 = 86. The split follows the previous model8from6a clean-300 split and is sample-level, not frame-random.",
        "",
        "## Best Current Run",
        "",
        f"- Run: `{best['run']}`",
        f"- Epochs: {int(best['epochs'])}",
        f"- Sample stride: {int(best['sample_stride'])}",
        f"- Disabled features: `{best['disabled']}`",
        f"- Top-1 accuracy: {best['accuracy']:.4f}",
        f"- Macro F1: {best['f1_macro']:.4f}",
        f"- Macro precision: {best['precision_macro']:.4f}",
        f"- Macro recall: {best['recall_macro']:.4f}",
        f"- Test residues: {int(best['n'])}",
        "",
        "## Epoch Comparison",
        "",
        base[["run", "epochs", "accuracy", "f1_macro", "precision_macro", "recall_macro", "n"]].to_markdown(index=False) if not base.empty else "Epoch comparison is not complete yet.",
        "",
        "## Ablation Comparison",
        "",
        ablation[["run", "disabled", "accuracy", "f1_macro", "precision_macro", "recall_macro", "feature_dim"]].sort_values("disabled").to_markdown(index=False) if not ablation.empty else "Ablation comparison is not complete yet.",
        "",
        "## SCI Figure Outputs",
        "",
    ]
    for fig in figures:
        lines.append(f"- `{fig.relative_to(ROOT)}`")
    lines.extend([
        "",
        "## Figure Captions",
        "",
        "**Figure 8. Clean-300 epoch comparison of model18 dynamic DSSP prediction.** Top-1 accuracy and macro F1 are shown for 100, 300, and 500 epochs using the same clean-300 manifest and the same sample stride. This panel evaluates whether longer optimization improves residue-level SS8 classification under a fixed data split.",
        "",
        "**Figure 9. Feature ablation of model18 on the clean-300 database.** The full model is compared with variants removing dynamic geometry, contact features, contact-map row statistics, all contact information, residue static descriptors, or residue position. Accuracy and macro F1 quantify the contribution of each module to dynamic DSSP prediction.",
        "",
        "**Figure 10. Clean-300 dataset composition.** The fixed preprocessing database contains Q22, Q36, and Q46 SMD samples and preserves the model8from6a sample-level split. This figure documents the training source used for all model18 epoch and ablation experiments.",
        "",
        "## Main Text Draft",
        "",
        "We next evaluated model18 on the fixed clean-300 PolyQ SMD database using a modular frame-residue representation. Each residue-frame sample was represented by dynamic trajectory descriptors, contact features, contact-map row statistics, residue-level descriptors, and normalized residue position. Training and testing were performed from the saved preprocessing database rather than recomputing raw trajectories, which ensured that all epoch and ablation comparisons used identical input samples.",
        "",
        "The epoch comparison showed the optimization behavior of the clean-300 model18 baseline under a fixed split. Feature ablation further quantified the relative contribution of dynamic geometry and contact-derived information. Because the labels are model-derived dynamic DSSP states and PolyQ is intrinsically disordered, the model output should be interpreted as a probabilistic ensemble descriptor rather than a unique native fold assignment.",
        "",
        "## Reliability Statement",
        "",
        "These results are suitable for an SCI methods/results section as computational model outputs if the manuscript clearly states that DSSP labels are model-derived, no mkdssp or experimental DSSP labels were used, and the 3D structures are trajectory-retrieved ensemble conformers rather than experimentally determined native structures.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    df = collect()
    if df.empty:
        raise SystemExit("No training result directories found")
    figures = []
    for fig in [plot_epoch(df), plot_ablation(df), plot_length_distribution()]:
        if fig:
            figures.append(fig)
            print(f"wrote {fig}")
    write_docs(df, figures)


if __name__ == "__main__":
    main()
