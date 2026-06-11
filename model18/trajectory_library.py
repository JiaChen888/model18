from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import SS8_ORDER


GEOM_COLS = [
    "rg_A", "end_to_end_A", "mean_pair_dist_A", "contact_density_8A", "contact_density_12A",
    "ca_dist_i_i5_mean_A", "ca_dist_i_i10_mean_A", "pseudo_dihedral_mean", "pseudo_dihedral_std",
]


class TrajectoryLibrary:
    def __init__(self, temporal_table: str | Path, geometry_table: str | Path):
        self.temporal_table = Path(temporal_table)
        self.geometry_table = Path(geometry_table)

    def load_frame_profiles(self, length: int) -> pd.DataFrame:
        usecols = ["group", "run_id", "frame_index", "time_ps", "residue", "length", "hard_label_ss8",
                   "contact_row_sum_8A", "dist_row_mean_A", *[f"frame_{c}" for c in GEOM_COLS]]
        df = pd.read_csv(self.temporal_table, usecols=usecols)
        df = df[df["length"] == length].copy()
        if df.empty:
            raise ValueError(f"No temporal/contact records for length {length}")
        rows = []
        for (group, run_id, frame_index, time_ps), g in df.groupby(["group", "run_id", "frame_index", "time_ps"], sort=False):
            g = g.sort_values("residue")
            dssp = "".join(g["hard_label_ss8"].astype(str).tolist())
            row = {
                "group": group,
                "run_id": run_id,
                "frame_index": int(frame_index),
                "time_ps": float(time_ps),
                "length": length,
                "frame_dssp": dssp,
                "contact_row_sum_mean": float(g["contact_row_sum_8A"].mean()),
                "dist_row_mean_A": float(g["dist_row_mean_A"].mean()),
            }
            first = g.iloc[0]
            for c in GEOM_COLS:
                row[c] = float(first[f"frame_{c}"])
            rows.append(row)
        return pd.DataFrame(rows)

    def geometry_sources(self) -> pd.DataFrame:
        return pd.read_csv(self.geometry_table)

    def source_for_frame(self, run_id: str, frame_index: int) -> dict:
        geom = self.geometry_sources()
        hit = geom[(geom["run_id"] == run_id) & (geom["frame_index"] == frame_index)]
        if hit.empty:
            raise ValueError(f"No source trajectory for {run_id} frame {frame_index}")
        row = hit.iloc[0].to_dict()
        return row


def dssp_composition(sequence: str) -> dict[str, float]:
    n = max(len(sequence), 1)
    return {s: sequence.count(s) / n for s in SS8_ORDER}


def hamming_similarity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n
