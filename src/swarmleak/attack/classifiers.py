"""Attackers (Section 4.4): random forests (adaptive adversary) and the heuristic baseline."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..constants import RF_MIN_LEAF, RF_TREES
from .features import TX_BYTES_COL


def make_rf(seed: int, n_estimators: int = RF_TREES, min_samples_leaf: int = RF_MIN_LEAF,
            n_jobs: int = 1) -> RandomForestClassifier:
    """Random forest with balanced class weights (Section 4.4, Table 2)."""
    return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs=n_jobs,
                                  random_state=seed, class_weight="balanced")


def argmax_random_ties(scores: np.ndarray, rng: np.random.Generator) -> int:
    """argmax with uniform random tie-breaking (audit M2)."""
    s = np.asarray(scores, float)
    if s.size == 0:
        raise ValueError("empty score vector")
    best = np.flatnonzero(s == s.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def positive_proba(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """P(class 1); zeros if the model never saw class 1 during training."""
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return model.predict_proba(X)[:, classes.index(1)]


def heuristic_role_estimate(window_rows: np.ndarray, rng: np.random.Generator) -> int:
    """Heuristic baseline: the UAV transmitting the most unicast bytes in the window."""
    return argmax_random_ties(window_rows[:, TX_BYTES_COL], rng)
