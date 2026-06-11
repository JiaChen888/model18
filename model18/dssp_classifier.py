from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import SS8_ORDER


@dataclass
class DSSPProfile:
    length: int
    probabilities: pd.DataFrame
    consensus: str
    source: str


class DSSPClassifierBase:
    """Interface for DSSP modules. Replace this with direct model12 inference later."""

    def predict_probabilities(self, sequence: str) -> DSSPProfile:
        raise NotImplementedError


class Model12ProbabilityTableClassifier(DSSPClassifierBase):
    """Default model18 adapter.

    It derives residue-wise SS8 probabilities from the existing model12/model16/model17
    probability table. This preserves the model12-centric pipeline while keeping the
    interface replaceable by a direct model12 checkpoint loader.
    """

    def __init__(self, temporal_table: str | Path):
        self.temporal_table = Path(temporal_table)
        self._prob_cols = [f"soft_prob_{s}" for s in SS8_ORDER]

    def predict_probabilities(self, sequence: str) -> DSSPProfile:
        length = len(sequence)
        usecols = ["length", "residue", *self._prob_cols]
        df = pd.read_csv(self.temporal_table, usecols=usecols)
        df = df[df["length"] == length]
        if df.empty:
            raise ValueError(f"No probability records for sequence length {length}")
        grouped = df.groupby("residue", as_index=False)[self._prob_cols].mean()
        grouped = grouped.sort_values("residue")
        if len(grouped) != length:
            raise ValueError(f"Expected {length} residue probabilities, found {len(grouped)}")
        probs = grouped[self._prob_cols].to_numpy(dtype=float)
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        out = pd.DataFrame(probs, columns=SS8_ORDER)
        out.insert(0, "residue", np.arange(1, length + 1))
        consensus = "".join(SS8_ORDER[i] for i in probs.argmax(axis=1))
        return DSSPProfile(length=length, probabilities=out, consensus=consensus, source=str(self.temporal_table))
