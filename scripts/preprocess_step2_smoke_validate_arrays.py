#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model18.preprocessed_db import load_sample_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Step2: smoke-validate per-frame DSSP/contact/dynamic arrays")
    parser.add_argument("--index", required=True)
    parser.add_argument("--outdir", default=str(ROOT / "outputs/preprocess_db"))
    parser.add_argument("--max-samples", type=int, default=12)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    ready = index[index["ready"] == True].head(args.max_samples)
    rows = []
    for _, row in ready.iterrows():
        arrays = load_sample_arrays(row["sample_dir"], mmap=True)
        dssp = arrays["dssp_labels"]
        contact = arrays["contact_maps"]
        dyn = arrays["dynamic_features"]
        rows.append({
            "sample_id": row["sample_id"],
            "length": int(row["length"]),
            "dssp_frames": int(dssp.shape[0]),
            "dssp_length": int(dssp.shape[1]),
            "contact_shape": "x".join(map(str, contact.shape)),
            "dynamic_shape": "x".join(map(str, dyn.shape)),
            "dssp_min": int(np.nanmin(dssp)),
            "dssp_max": int(np.nanmax(dssp)),
            "contact_min": float(np.nanmin(contact)),
            "contact_max": float(np.nanmax(contact)),
        })
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "preprocessed_smoke_validation.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
