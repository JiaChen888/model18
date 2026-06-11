from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import SS8_ORDER
from .trajectory_library import GEOM_COLS, dssp_composition, hamming_similarity


def rank_frames(frame_profiles: pd.DataFrame, target_dssp: str, top_k: int = 5) -> pd.DataFrame:
    frames = frame_profiles.copy()
    target_comp = dssp_composition(target_dssp)
    geom_means = frames[GEOM_COLS].mean()
    geom_stds = frames[GEOM_COLS].std().replace(0, 1.0).fillna(1.0)
    rows = []
    for _, row in frames.iterrows():
        frame_dssp = row["frame_dssp"]
        pos_score = hamming_similarity(target_dssp, frame_dssp)
        comp_score = 1.0 - 0.5 * sum(abs(target_comp[s] - frame_dssp.count(s) / len(frame_dssp)) for s in SS8_ORDER)
        geom_z = np.sqrt(np.mean(((row[GEOM_COLS] - geom_means) / geom_stds) ** 2))
        geom_score = float(np.exp(-0.5 * geom_z))
        contact_score = float(np.exp(-abs(row["contact_density_8A"] - geom_means["contact_density_8A"]) / (geom_stds["contact_density_8A"] + 1e-6)))
        total = 0.45 * pos_score + 0.25 * comp_score + 0.20 * geom_score + 0.10 * contact_score
        out = row.to_dict()
        out.update({
            "dssp_position_score": pos_score,
            "dssp_composition_score": comp_score,
            "geometry_score": geom_score,
            "contact_score": contact_score,
            "total_score": total,
        })
        rows.append(out)
    ranked = pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked.head(top_k)
