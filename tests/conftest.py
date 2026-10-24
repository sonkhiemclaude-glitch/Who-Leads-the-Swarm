from __future__ import annotations

import numpy as np
import pytest

from swarmleak.data.schemas import FRAME_DTYPE
from swarmleak.data.synthetic import surrogate_trace
from swarmleak.mobility import gauss_markov_3d


@pytest.fixture
def small_trace():
    return surrogate_trace(6, 180.0, np.random.default_rng(1))


@pytest.fixture
def small_traj():
    return gauss_markov_3d(6, 180.0, np.random.default_rng(2), area=(600.0, 600.0))


def make_frames(rows):
    """rows: (t, size, tx, rx)"""
    F = np.zeros(len(rows), dtype=FRAME_DTYPE)
    for k, (t, size, tx, rx) in enumerate(rows):
        F[k] = (t, size, tx, rx, k % 4096, 0, 0, -60.0)
    return F
