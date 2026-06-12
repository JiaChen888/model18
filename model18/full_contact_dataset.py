from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class FullContactSwitches:
    use_dynamic: bool = True
    use_contact_features: bool = True
    use_residue_features: bool = True
    use_residue_position: bool = True
    use_full_contact_map: bool = True

    @classmethod
    def from_ablation_names(cls, disabled: list[str] | None = None) -> "FullContactSwitches":
        cfg = cls()
        for name in disabled or []:
            if name in {"dynamic", "geometry"}:
                cfg.use_dynamic = False
            elif name == "contact_features":
                cfg.use_contact_features = False
            elif name in {"residue", "residue_features", "static"}:
                cfg.use_residue_features = False
            elif name in {"position", "residue_position"}:
                cfg.use_residue_position = False
            elif name in {"full_contact", "full_contact_map", "contact_map", "contact"}:
                cfg.use_full_contact_map = False
        return cfg

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def sample_frame_indices(n_frames: int, stride: int) -> np.ndarray:
    stride = max(int(stride), 1)
    idx = np.arange(0, n_frames, stride, dtype=int)
    if len(idx) == 0 or idx[-1] != n_frames - 1:
        idx = np.unique(np.r_[idx, n_frames - 1]).astype(int)
    return idx


class Clean300FullContactFrameDataset(Dataset):
    """Frame-level clean-300 dataset preserving full LxL contact maps.

    Each item is one trajectory frame. The target is the residue-level SS8 label
    vector for that frame. This is the model18 bridge back to the model6/model11/
    model12 graph-contact design, where the complete contact/distance matrix is
    encoded before residue-level DSSP prediction.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        sample_stride: int = 500,
        max_frames: int | None = None,
        switches: FullContactSwitches | None = None,
        seed: int = 18,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.split = split
        self.sample_stride = int(sample_stride)
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
        n = int(mask.shape[0])
        length = int(row["length"])
        pos = np.arange(n, dtype=np.float32) / max(length - 1, 1)
        length_feat = np.full(n, length / 46.0, dtype=np.float32)
        return {
            "contact_map": torch.from_numpy(cmap),
            "dynamic": torch.from_numpy(dyn),
            "contact_features": torch.from_numpy(cfeat),
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
            "feature_switches": self.switches.to_dict(),
        }
