#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model18.pipeline import run_ensemble_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="model18 modular PolyQ DSSP-to-3D ensemble pipeline")
    parser.add_argument("--config", default=str(ROOT / "configs/default_paths.json"))
    parser.add_argument("--length", type=int, default=46, choices=[22, 36, 46])
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()
    sequence = args.sequence or ("Q" * args.length)
    summary = run_ensemble_pipeline(args.config, sequence, n_samples=args.samples, top_k=args.top_k, seed=args.seed)
    print("model18 complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
