#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Step3: build train/inference manifest from preprocessed DB index")
    parser.add_argument("--index", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--split-source", default=None, help="Optional model run directory containing train_indices.npy/test_indices.npy")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.index)
    df = df[df["ready"] == True].copy()
    df["sample_id_int"] = pd.to_numeric(df["sample_id"], errors="coerce").astype("Int64")
    df["split"] = "inference"
    if args.split_source:
        import numpy as np
        split = Path(args.split_source)
        train_path = split / "train_indices.npy"
        test_path = split / "test_indices.npy"
        if train_path.exists():
            train = set(map(int, np.load(train_path)))
            df.loc[df["sample_id_int"].isin(train), "split"] = "train"
        if test_path.exists():
            test = set(map(int, np.load(test_path)))
            df.loc[df["sample_id_int"].isin(test), "split"] = "test"
    df.to_csv(outdir / "model18_training_manifest.csv", index=False)
    df.groupby(["length", "split"]).size().reset_index(name="samples").to_csv(outdir / "model18_training_manifest_summary.csv", index=False)
    print(df.groupby(["length", "split"]).size().reset_index(name="samples").to_string(index=False))


if __name__ == "__main__":
    main()
