from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import SS8_ORDER
from .dssp_classifier import DSSPProfile


@dataclass
class SampledDSSPSet:
    sequences: list[str]
    frequency_table: pd.DataFrame
    consensus: str


def sample_dssp_sequences(profile: DSSPProfile, n: int = 200, seed: int = 18) -> SampledDSSPSet:
    rng = np.random.default_rng(seed)
    probs = profile.probabilities[SS8_ORDER].to_numpy(dtype=float)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    samples = []
    for _ in range(n):
        labels = [rng.choice(SS8_ORDER, p=row) for row in probs]
        samples.append("".join(labels))
    counts = {s: np.zeros(profile.length, dtype=float) for s in SS8_ORDER}
    for seq in samples:
        for i, label in enumerate(seq):
            counts[label][i] += 1
    freq = pd.DataFrame({"residue": np.arange(1, profile.length + 1)})
    for label in SS8_ORDER:
        freq[label] = counts[label] / float(n)
    consensus = "".join(freq.loc[i, SS8_ORDER].idxmax() for i in range(profile.length))
    return SampledDSSPSet(sequences=samples, frequency_table=freq, consensus=consensus)
