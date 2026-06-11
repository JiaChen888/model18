from __future__ import annotations

import numpy as np
import pandas as pd

from .ablation import AblationConfig, load_ablation_config
from .constants import SS8_ORDER
from .trajectory_library import GEOM_COLS, dssp_composition, hamming_similarity


def _normalized_weights(cfg: AblationConfig) -> dict[str, float]:
    weights = {
        "dssp_position": cfg.dssp_position_weight if cfg.use_dssp_position else 0.0,
        "dssp_composition": cfg.dssp_composition_weight if cfg.use_dssp_composition else 0.0,
        "geometry": cfg.geometry_weight if cfg.use_geometry else 0.0,
        "contact": cfg.contact_weight if cfg.use_contact else 0.0,
    }
    total = sum(weights.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}


def rank_frames(frame_profiles: pd.DataFrame, target_dssp: str, top_k: int = 5,
                ablation: AblationConfig | dict | None = None) -> pd.DataFrame:
    cfg = ablation if isinstance(ablation, AblationConfig) else load_ablation_config(ablation)
    frames = frame_profiles.copy()
    target_comp = dssp_composition(target_dssp)
    geom_means = frames[GEOM_COLS].mean()
    geom_stds = frames[GEOM_COLS].std().replace(0, 1.0).fillna(1.0)
    weights = _normalized_weights(cfg)
    rows = []
    for _, row in frames.iterrows():
        frame_dssp = row.get("stress_adjusted_dssp", row["frame_dssp"]) if cfg.use_stress_adjusted_dssp else row["frame_dssp"]
        pos_score = hamming_similarity(target_dssp, frame_dssp)
        comp_score = 1.0 - 0.5 * sum(abs(target_comp[s] - frame_dssp.count(s) / len(frame_dssp)) for s in SS8_ORDER)
        geom_z = np.sqrt(np.mean(((row[GEOM_COLS] - geom_means) / geom_stds) ** 2))
        geom_score = float(np.exp(-0.5 * geom_z))
        contact_score = float(np.exp(-abs(row["contact_density_8A"] - geom_means["contact_density_8A"]) / (geom_stds["contact_density_8A"] + 1e-6)))
        total = (
            weights["dssp_position"] * pos_score
            + weights["dssp_composition"] * comp_score
            + weights["geometry"] * geom_score
            + weights["contact"] * contact_score
        ) if sum(weights.values()) > 0 else 0.0
        out = row.to_dict()
        out.update({
            "ranking_dssp_sequence_used": frame_dssp,
            "dssp_position_score": pos_score,
            "dssp_composition_score": comp_score,
            "geometry_score": geom_score,
            "contact_score": contact_score,
            "weight_dssp_position": weights["dssp_position"],
            "weight_dssp_composition": weights["dssp_composition"],
            "weight_geometry": weights["geometry"],
            "weight_contact": weights["contact"],
            "total_score": total,
        })
        rows.append(out)
    ranked = pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked.head(top_k)
