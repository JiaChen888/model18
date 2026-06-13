from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from model18.full_contact_dataset import FullContactSwitches, sample_frame_indices


class Clean300Model19WindowDataset(Dataset):
    """Frame-level dataset for model19/model12-plus-contact-map.

    clean300 currently does not export raw force windows. This dataset therefore
    builds a model12-compatible temporal window from available per-frame dynamic
    and contact features. The tensor is intentionally named `force_window_proxy`
    so raw pull-force windows can replace it later without changing model code.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        sample_stride: int = 500,
        window_size: int = 5,
        max_frames: int | None = None,
        switches: FullContactSwitches | None = None,
        seed: int = 19,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.split = split
        self.sample_stride = int(sample_stride)
        self.window_size = int(window_size)
        self.switches = switches or FullContactSwitches()
        manifest = pd.read_csv(self.manifest_csv)
        df = manifest[(manifest["ready"] == True) & (manifest["split"] == split)].copy()
        df = df.sort_values("sample_id_int").reset_index(drop=True)
        rows: list[dict] = []
        for _, row in df.iterrows():
            sample_dir = Path(row["sample_dir"])
            dssp = np.load(sample_dir / "dssp_labels.npy", mmap_mode="r")
            for frame in sample_frame_indices(dssp.shape[0], self.sample_stride):
                rows.append({
                    "sample_id": int(row["sample_id_int"]),
                    "sample_dir": str(sample_dir),
                    "frame": int(frame),
                    "length": int(row["length"]),
                })
        if max_frames is not None and len(rows) > max_frames:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(rows), size=max_frames, replace=False))
            rows = [rows[i] for i in keep]
        self.rows = rows
        self._cache_key: str | None = None
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_sample(self, sample_dir: str) -> dict[str, np.ndarray]:
        if self._cache_key == sample_dir:
            return self._cache
        p = Path(sample_dir)
        self._cache_key = sample_dir
        self._cache = {
            "dssp": np.load(p / "dssp_labels.npy", mmap_mode="r"),
            "ymask": np.load(p / "y_masks.npy", mmap_mode="r"),
            "dynamic": np.load(p / "dynamic_features.npy", mmap_mode="r"),
            "contact_features": np.load(p / "contact_features.npy", mmap_mode="r"),
            "residue_features": np.load(p / "residue_features.npy", mmap_mode="r"),
            "contact_maps": np.load(p / "contact_maps.npy", mmap_mode="r"),
        }
        return self._cache

    def _window(self, dyn: np.ndarray, contact_feat: np.ndarray, frame: int) -> np.ndarray:
        half = self.window_size // 2
        idx = np.arange(frame - half, frame - half + self.window_size)
        idx = np.clip(idx, 0, dyn.shape[0] - 1)
        return np.concatenate([dyn[idx], contact_feat[idx]], axis=-1).astype(np.float32).copy()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        data = self._load_sample(row["sample_dir"])
        f = row["frame"]
        y = np.asarray(data["dssp"][f], dtype=np.int64).copy()
        mask = np.asarray(data["ymask"][f], dtype=np.float32).copy()
        cmap = np.asarray(data["contact_maps"][f], dtype=np.float32).copy()
        dyn = np.asarray(data["dynamic"][f], dtype=np.float32).copy()
        cfeat = np.asarray(data["contact_features"][f], dtype=np.float32).copy()
        rfeat = np.asarray(data["residue_features"], dtype=np.float32).copy()
        force_proxy = self._window(data["dynamic"], data["contact_features"], f)
        n = int(mask.shape[0])
        length = int(row["length"])
        pos = np.arange(n, dtype=np.float32) / max(length - 1, 1)
        length_feat = np.full(n, length / 46.0, dtype=np.float32)
        return {
            "contact_map": torch.from_numpy(cmap),
            "dynamic": torch.from_numpy(dyn),
            "contact_features": torch.from_numpy(cfeat),
            "force_window_proxy": torch.from_numpy(force_proxy),
            "residue_features": torch.from_numpy(rfeat),
            "position": torch.from_numpy(np.stack([pos, length_feat], axis=-1)),
            "target": torch.from_numpy(y),
            "mask": torch.from_numpy(mask),
            "length": torch.tensor(length, dtype=torch.long),
            "sample_id": torch.tensor(int(row["sample_id"]), dtype=torch.long),
            "frame": torch.tensor(int(f), dtype=torch.long),
        }

    def info(self) -> dict:
        return {
            "split": self.split,
            "frames": len(self.rows),
            "sample_stride": self.sample_stride,
            "window_size": self.window_size,
            "feature_switches": self.switches.to_dict(),
        }
