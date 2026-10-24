import numpy as np
import pytest

from swarmleak.attack.classifiers import argmax_random_ties, heuristic_role_estimate
from swarmleak.attack.datasets import orch_at
from swarmleak.attack.evaluate import mission_folds
from swarmleak.attack.features import ROLE_FEATURE_NAMES, WINDOW_FEATURE_NAMES, role_features, window_features

from .conftest import make_frames


def test_role_features_controlled_example():
    # UAV 0 sends 3 large frames to 1,2,3 within one second; UAV 1 sends one small frame to 0; one broadcast by 2
    F = make_frames([(1.1, 1500, 0, 1), (1.2, 1500, 0, 2), (1.3, 1500, 0, 3), (2.0, 300, 1, 0), (2.5, 60, 2, -1)])
    X = role_features(F, 4, 0.0, 10.0)
    assert X.shape == (4, 2 * len(ROLE_FEATURE_NAMES))
    assert X[0, 0] == 3 and X[0, 1] == 4500 and X[0, 4] == 3  # tx frames, bytes, fan-out
    assert X[0, 10] == 3  # >= 1400 bin
    assert X[0, 12] == 1.0  # all in one second
    assert X[0, 14] == 3  # fan-out of large frames
    assert X[0, 5] == 1 and X[0, 15] == 1  # fan-in, fan-in of small frames
    assert X[2, 13] == 1  # broadcast
    assert abs(X[:, 16:].mean(0)).max() < 1e-6  # standardised columns


def test_window_features():
    F = make_frames([(0.5, 1500, 0, 1), (1.5, 1500, 0, 2), (2.5, 300, 1, 0), (3.0, 60, 2, -1)])
    f = window_features(F, 3, 0.0, 4.0)
    assert len(f) == len(WINDOW_FEATURE_NAMES)
    assert f[0] == 3 and f[2] == 1 and f[10] == 2 and f[12] == 2 and f[13] == 2


def test_empty_window_is_zero():
    F = make_frames([])
    assert not role_features(F, 3, 0, 10)[:, :16].any()
    assert not window_features(F, 3, 0, 10).any()


def test_orch_at_piecewise():
    orch = np.array([[0, 4], [50, 7]], float)
    assert orch_at(orch, 10) == 4 and orch_at(orch, 50) == 7 and orch_at(orch, 1e9) == 7


def test_argmax_random_ties_unbiased():
    rng = np.random.default_rng(0)
    picks = [argmax_random_ties(np.array([1.0, 1.0, 0.0]), rng) for _ in range(400)]
    assert set(picks) == {0, 1} and 0.4 < np.mean(picks) < 0.6
    assert argmax_random_ties(np.array([0.0, 2.0, 1.0]), rng) == 1


def test_heuristic_uses_tx_bytes():
    rows = np.zeros((3, 32))
    rows[2, 1] = 99
    assert heuristic_role_estimate(rows, np.random.default_rng(0)) == 2


def test_mission_folds_disjoint():
    folds = mission_folds(12, 4)
    flat = [i for f in folds for i in f]
    assert sorted(flat) == list(range(12)) and len(folds) == 4
    assert len(mission_folds(3, 4)) == 3
    assert len(mission_folds(6, 4)) == 4
    with pytest.raises(ValueError):
        mission_folds(1, 4)
