from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_ARRAYS = [
    "sequences.npy",
    "dssp_labels.npy",
    "dynamic_features.npy",
    "contact_maps.npy",
    "contact_features.npy",
    "residue_features.npy",
    "seq_masks.npy",
    "y_masks.npy",
]

OPTIONAL_ARRAYS = [
    "sequence_times_ps.npy",
]


def infer_length_from_sample(sample_dir: str | Path) -> int | None:
    sample_dir = Path(sample_dir)
    mask_path = sample_dir / "y_masks.npy"
    if mask_path.exists():
        mask = np.load(mask_path, mmap_mode="r")
        if mask.ndim == 2 and mask.size:
            per_frame = np.asarray(mask.sum(axis=1)).astype(float)
            valid = per_frame[per_frame > 0]
            if valid.size:
                return int(round(float(np.median(valid))))
    label_path = sample_dir / "dssp_labels.npy"
    if label_path.exists():
        arr = np.load(label_path, mmap_mode="r")
        return int(arr.shape[1]) if arr.ndim == 2 else None
    return None


def audit_preprocessed_sample(sample_dir: str | Path) -> dict:
    sample_dir = Path(sample_dir)
    row = {"sample_id": sample_dir.name, "sample_dir": str(sample_dir), "exists": sample_dir.exists()}
    ok = True
    for name in REQUIRED_ARRAYS:
        path = sample_dir / name
        row[f"has_{name[:-4]}"] = path.exists()
        if not path.exists():
            ok = False
            continue
        arr = np.load(path, mmap_mode="r", allow_pickle=True)
        row[f"shape_{name[:-4]}"] = "x".join(map(str, arr.shape))
        row[f"dtype_{name[:-4]}"] = str(arr.dtype)
    for name in OPTIONAL_ARRAYS:
        path = sample_dir / name
        row[f"has_{name[:-4]}"] = path.exists()
        if path.exists():
            arr = np.load(path, mmap_mode="r", allow_pickle=True)
            row[f"shape_{name[:-4]}"] = "x".join(map(str, arr.shape))
            row[f"dtype_{name[:-4]}"] = str(arr.dtype)
    row["length"] = infer_length_from_sample(sample_dir)
    row["ready"] = ok and row["length"] in {22, 36, 46}
    return row


def build_preprocessed_index(process_dir: str | Path, output_csv: str | Path | None = None) -> pd.DataFrame:
    process_dir = Path(process_dir)
    rows = []
    for sample_dir in sorted([p for p in process_dir.iterdir() if p.is_dir()], key=lambda x: (0, int(x.name)) if x.name.isdigit() else (1, x.name)):
        rows.append(audit_preprocessed_sample(sample_dir))
    df = pd.DataFrame(rows)
    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def summarize_preprocessed_index(index_df: pd.DataFrame) -> pd.DataFrame:
    return index_df.groupby("length", dropna=False).agg(
        samples=("sample_id", "count"),
        ready=("ready", "sum"),
    ).reset_index()


def load_sample_arrays(sample_dir: str | Path, mmap: bool = True) -> dict[str, np.ndarray]:
    sample_dir = Path(sample_dir)
    mode = "r" if mmap else None
    arrays = {name[:-4]: np.load(sample_dir / name, mmap_mode=mode, allow_pickle=True) for name in REQUIRED_ARRAYS}
    for name in OPTIONAL_ARRAYS:
        path = sample_dir / name
        if path.exists():
            arrays[name[:-4]] = np.load(path, mmap_mode=mode, allow_pickle=True)
    return arrays
