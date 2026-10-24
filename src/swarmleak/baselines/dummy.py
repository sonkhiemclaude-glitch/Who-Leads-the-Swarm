"""DUMMY baselines for unit tests only. Never report their numbers."""
from __future__ import annotations

import numpy as np


class DummyRandomAttacker:
    """Names a uniformly random UAV per window. Test fixture, not a baseline from the paper."""

    is_dummy = True

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def estimate(self, n_windows: int, n_uavs: int) -> np.ndarray:
        return self.rng.integers(0, n_uavs, size=n_windows)
