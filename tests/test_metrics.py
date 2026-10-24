import math

import numpy as np
import pytest

from swarmleak.metrics.confidence_intervals import paired_bootstrap, t_interval
from swarmleak.metrics.leakage import (
    detect_events,
    majority_rate,
    match_events,
    phase_accuracy,
    plan_change_f1,
    role_accuracy_at_horizon,
)
from swarmleak.metrics.rotation_stats import rotation_stats


def test_role_accuracy():
    assert role_accuracy_at_horizon(np.array([1, 2, 3, 4]), np.array([1, 2, 0, 0])) == 0.5
    assert math.isnan(role_accuracy_at_horizon(np.array([]), np.array([])))
    with pytest.raises(ValueError):
        role_accuracy_at_horizon(np.array([1]), np.array([1, 2]))


def test_phase_accuracy_and_majority():
    assert phase_accuracy(np.array(["a", "b"]), np.array(["a", "a"])) == 0.5
    assert majority_rate(np.array(["a", "a", "b", "c"])) == 0.5


def test_event_matching_temporal_order_nearest_unmatched():
    truth = np.array([10.0, 20.0])
    # first detection takes 10, second (11) has 10 used and 20 too far -> unmatched
    assert match_events(np.array([9.0, 11.0]), truth, tol=5) == 1
    assert match_events(np.array([9.0, 18.0]), truth, tol=5) == 2
    s = plan_change_f1(np.array([9.0, 11.0, 30.0]), truth, tol=5)
    assert s.tp == 1 and s.precision == pytest.approx(1 / 3) and s.recall == 0.5
    assert s.f1 == pytest.approx(2 * (1 / 3) * 0.5 / (1 / 3 + 0.5))


def test_f1_edge_cases():
    assert math.isnan(plan_change_f1(np.array([]), np.array([])).f1)
    s = plan_change_f1(np.array([]), np.array([5.0]))
    assert s.f1 == 0.0 and math.isnan(s.precision) and s.recall == 0.0
    s = plan_change_f1(np.array([5.0]), np.array([]))
    assert s.f1 == 0.0 and math.isnan(s.recall)


def test_detect_events_peaks_threshold_merge():
    ts = np.arange(10.0)
    p = np.array([0.1, 0.6, 0.9, 0.6, 0.2, 0.7, 0.7, 0.3, 0.55, 0.1])
    ev = detect_events(ts, p, W=4.0, threshold=0.5, merge_s=3.0)
    # peaks at 2 (0.9), plateau 5-6 keeps 5, 8 (0.55); 2+2=4, 5+2=7 (gap 3 not > 3 -> merged), 8+2=10
    assert ev.tolist() == [4.0, 10.0]


def test_t_interval():
    m, h = t_interval([1.0, 2.0, 3.0])
    assert m == 2.0 and h == pytest.approx(4.302652729911275 * 1.0 / math.sqrt(3))
    m, h = t_interval([5.0])
    assert m == 5.0 and math.isnan(h)
    assert math.isnan(t_interval([float("nan")])[0])


def test_paired_bootstrap():
    a = [0.5, 0.6, 0.7, 0.8]
    b = [0.4, 0.5, 0.6, 0.7]
    r = paired_bootstrap(a, b, n_boot=500, seed=1)
    assert r["mean_diff"] == pytest.approx(0.1) and r["lo"] == pytest.approx(0.1) and r["hi"] == pytest.approx(0.1)
    assert paired_bootstrap([np.nan], [1.0])["n"] == 0


def test_rotation_stats():
    orch = np.array([[0, 1], [50, 2], [100, 3], [105, 4]], float)
    st = rotation_stats(orch, np.array([40.0, 60.0, 95.0]), horizon=15)
    assert st["p_E"] == pytest.approx(2 / 3) and st["p_two_plus"] == pytest.approx(1 / 3)
    assert st["mean_interval_s"] == pytest.approx(105 / 3)
