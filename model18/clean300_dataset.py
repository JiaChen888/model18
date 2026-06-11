from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FeatureSwitches:
    use_dynamic: bool = True
    use_contact_features: bool = True
    use_contact_map_stats: bool = True
    use_residue_features: bool = True
    use_residue_position: bool = True

    @classmethod
    def from_ablation_names(cls, disabled: list[str] | None = None) -> "FeatureSwitches":
        cfg = cls()
        for name in disabled or []:
            if name in {"dynamic", "geometry"}:
                cfg.use_dynamic = False
            elif name == "contact_features":
                cfg.use_contact_features = False
            elif name in {"contact", "contact_map", "contact_map_stats"}:
                cfg.use_contact_map_stats = False
            elif name in {"residue", "residue_features", "static"}:
                cfg.use_residue_features = False
            elif name in {"position", "residue_position"}:
                cfg.use_residue_position = False
        return cfg

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _sample_frames(n_frames: int, stride: int) -> np.ndarray:
    stride = max(int(stride), 1)
    idx = np.arange(0, n_frames, stride, dtype=int)
    if len(idx) == 0 or idx[-1] != n_frames - 1:
        idx = np.unique(np.r_[idx, n_frames - 1]).astype(int)
    return idx


def build_frame_residue_table(
    manifest_csv: str | Path,
    split: str = "train",
    sample_stride: int = 100,
    max_samples: int | None = None,
    max_rows: int | None = None,
    feature_switches: FeatureSwitches | None = None,
    seed: int = 18,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Load clean-300 preprocessed samples into a frame/residue matrix.

    The clean-300 database stores arrays per sample. This function builds an in-memory
    supervised table suitable for fast baseline training and ablation tests.
    """
    switches = feature_switches or FeatureSwitches()
    manifest = pd.read_csv(manifest_csv)
    df = manifest[(manifest["ready"] == True) & (manifest["split"] == split)].copy()
    df = df.sort_values("sample_id_int")
    if max_samples is not None:
        df = df.head(max_samples)
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    meta_rows: list[dict] = []
    for _, row in df.iterrows():
        sample_dir = Path(row["sample_dir"])
        dssp = np.load(sample_dir / "dssp_labels.npy", mmap_mode="r")
        ymask = np.load(sample_dir / "y_masks.npy", mmap_mode="r")
        dyn = np.load(sample_dir / "dynamic_features.npy", mmap_mode="r")
        cfeat = np.load(sample_dir / "contact_features.npy", mmap_mode="r")
        rfeat = np.load(sample_dir / "residue_features.npy", mmap_mode="r")
        cmap = np.load(sample_dir / "contact_maps.npy", mmap_mode="r") if switches.use_contact_map_stats else None
        length = int(row["length"])
        frames = _sample_frames(dssp.shape[0], sample_stride)
        for frame in frames:
            valid_res = np.where(np.asarray(ymask[frame]) > 0)[0]
            if len(valid_res) == 0:
                valid_res = np.arange(length)
            for res in valid_res:
                parts = []
                if switches.use_dynamic:
                    parts.append(np.asarray(dyn[frame], dtype=np.float32))
                if switches.use_contact_features:
                    parts.append(np.asarray(cfeat[frame], dtype=np.float32))
                if switches.use_contact_map_stats and cmap is not None:
                    row_map = np.asarray(cmap[frame, res], dtype=np.float32)
                    parts.append(np.asarray([row_map.mean(), row_map.std(), row_map.max(), row_map.sum()], dtype=np.float32))
                if switches.use_residue_features:
                    parts.append(np.asarray(rfeat[res], dtype=np.float32))
                if switches.use_residue_position:
                    denom = max(length - 1, 1)
                    parts.append(np.asarray([res / denom, length / 46.0], dtype=np.float32))
                x = np.concatenate(parts).astype(np.float32)
                rows_x.append(x)
                rows_y.append(np.asarray(dssp[frame, res], dtype=np.int64))
                meta_rows.append({"sample_id": int(row["sample_id_int"]), "length": length, "frame": int(frame), "residue": int(res)})
    X = np.vstack(rows_x).astype(np.float32) if rows_x else np.empty((0, 0), dtype=np.float32)
    y = np.asarray(rows_y, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)
    if max_rows is not None and len(y) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_rows, replace=False)
        idx.sort()
        X = X[idx]
        y = y[idx]
        meta = meta.iloc[idx].reset_index(drop=True)
    info = {
        "split": split,
        "samples": int(df["sample_id_int"].nunique()),
        "rows": int(len(y)),
        "feature_dim": int(X.shape[1]) if X.ndim == 2 and X.size else 0,
        "sample_stride": int(sample_stride),
        "feature_switches": switches.to_dict(),
    }
    return X, y, meta, info
