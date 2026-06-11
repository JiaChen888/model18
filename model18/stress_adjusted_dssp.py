from __future__ import annotations

import pandas as pd


def stress_adjust_sequence(dssp: str, force: float | None = None, extension: float | None = None,
                           force_threshold: float | None = None, extension_threshold: float | None = None) -> str:
    """Convert likely force-induced S states to C under high-stress conditions."""
    high_force = force is not None and force_threshold is not None and force >= force_threshold
    high_extension = extension is not None and extension_threshold is not None and extension >= extension_threshold
    if not (high_force or high_extension):
        return dssp
    return "".join("C" if x == "S" else x for x in dssp)


def add_stress_adjusted_column(df: pd.DataFrame, dssp_col: str = "frame_dssp", force_col: str = "force",
                               extension_col: str = "end_to_end_A", force_quantile: float = 0.75,
                               extension_quantile: float = 0.75) -> pd.DataFrame:
    out = df.copy()
    force_threshold = out[force_col].quantile(force_quantile) if force_col in out.columns else None
    extension_threshold = out[extension_col].quantile(extension_quantile) if extension_col in out.columns else None
    out["stress_adjusted_dssp"] = [
        stress_adjust_sequence(row[dssp_col], row.get(force_col), row.get(extension_col), force_threshold, extension_threshold)
        for _, row in out.iterrows()
    ]
    return out
