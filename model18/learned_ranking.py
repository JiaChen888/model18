from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANK_FEATURES = [
    "dssp_position_score", "dssp_composition_score", "geometry_score", "contact_score",
    "rg_A", "end_to_end_A", "mean_pair_dist_A", "contact_density_8A", "contact_density_12A",
]


@dataclass
class LearnedRanker:
    model: Pipeline
    features: list[str]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features].fillna(0.0))


def train_learned_ranker(training_df: pd.DataFrame, target_col: str = "target_score") -> LearnedRanker:
    """Train a lightweight ranking regressor.

    target_col can later be force-derived, FES-derived, or event-label-derived.
    """
    features = [c for c in RANK_FEATURES if c in training_df.columns]
    if target_col not in training_df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=18, min_samples_leaf=2)),
    ])
    model.fit(training_df[features].fillna(0.0), training_df[target_col])
    return LearnedRanker(model=model, features=features)
