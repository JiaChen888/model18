from __future__ import annotations

import pandas as pd


def filter_pre_pore_entry(features: pd.DataFrame, force_col: str = "force", z_col: str | None = None,
                          max_time_ps: float | None = None, force_quantile: float = 0.25) -> pd.DataFrame:
    """Lightweight early-SMD filter.

    If explicit pore-entry labels are unavailable, retain low-force early frames as a
    conservative pre-entry proxy. A z-coordinate column can be added later when pore
    center coordinates are standardized.
    """
    df = features.copy()
    mask = pd.Series(True, index=df.index)
    if max_time_ps is not None and "time_ps" in df.columns:
        mask &= df["time_ps"] <= max_time_ps
    if force_col in df.columns:
        threshold = df[force_col].quantile(force_quantile)
        mask &= df[force_col] <= threshold
    if z_col and z_col in df.columns:
        mask &= df[z_col].notna()
    return df[mask].copy()
