"""Similarity helpers for Phase 1 probes (not part of the codec itself)."""

import numpy as np


def centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean-centered cosine similarity, as used in the paper's probes.
    May not be really required as the codec vector is already mean-centered
    """
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / denom)
