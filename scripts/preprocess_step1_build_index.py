#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model18.preprocessed_db import build_preprocessed_index, summarize_preprocessed_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Step1: build index for existing per-sample preprocessed database")
    parser.add_argument("--process-dir", required=True)
    parser.add_argument("--outdir", default=str(ROOT / "outputs/preprocess_db"))
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    index = build_preprocessed_index(args.process_dir, outdir / "preprocessed_sample_index.csv")
    summary = summarize_preprocessed_index(index)
    summary.to_csv(outdir / "preprocessed_sample_summary_by_length.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
