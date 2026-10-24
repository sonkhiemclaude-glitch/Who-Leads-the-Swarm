import numpy as np
import pytest

from swarmleak.theory.horizon_bound import (
    guaranteed_reduction,
    prop1_bound,
    prop1_exact,
    rotation_probability_upper_bound,
    top_k_pmax,
)


def test_bound_formula_and_boundaries():
    assert prop1_bound(0.9, 0.0, 1 / 3) == pytest.approx(0.9)  # no rotation -> alpha0
    assert prop1_bound(1.0, 1.0, 1 / 3) == pytest.approx(0.0)
    assert prop1_bound(0.9, 0.5, 1 / 3) == pytest.approx(0.9 * 0.5 + 0.5 * 0.1)
    assert prop1_bound(0.5, 0.5, 0.2) == pytest.approx(0.25 + 0.5 * 0.2)
    assert guaranteed_reduction(0.3, 0.5, 1 / 3) <= 0  # no guarantee when alpha0 <= min{1-alpha0, p_max}
    with pytest.raises(ValueError):
        prop1_bound(1.2, 0.1, 0.1)


def test_exact_value_below_bound():
    for a in (0.2, 0.6, 0.95):
        for p in (0.1, 0.5):
            assert prop1_exact(a, p, 20) <= prop1_bound(a, p, 1.0) + 1e-12


def test_monte_carlo_exact_value_under_assumptions():
    """Simulate C, E independent and a uniform successor (conditionally independent of o_hat)."""
    rng = np.random.default_rng(0)
    N, a, p, K = 10, 0.7, 0.4, 400_000
    o_t = rng.integers(0, N, K)
    correct = rng.random(K) < a
    wrong = (o_t + rng.integers(1, N, K)) % N
    o_hat = np.where(correct, o_t, wrong)
    E = rng.random(K) < p
    succ = (o_t + rng.integers(1, N, K)) % N
    sigma = np.where(E, succ, o_t)
    assert np.mean(o_hat == sigma) == pytest.approx(prop1_exact(a, p, N), abs=3e-3)


def test_monte_carlo_bound_holds_with_positive_correlation_and_top3():
    rng = np.random.default_rng(1)
    N, K = 20, 400_000
    E = rng.random(K) < 0.5
    # identification is easier when a rotation is imminent: P(C|E) = 0.9 > P(C|not E) = 0.7 (A1 holds)
    correct = rng.random(K) < np.where(E, 0.9, 0.7)
    a, p = correct.mean(), E.mean()
    o_t = rng.integers(0, N, K)
    top3 = np.stack([(o_t + k) % N for k in (1, 2, 3)], axis=1)
    succ = top3[np.arange(K), rng.integers(0, 3, K)]
    # adversary that, when wrong, guesses the first top-3 candidate (knows the topology)
    o_hat = np.where(correct, o_t, top3[:, 0])
    sigma = np.where(E, succ, o_t)
    acc = np.mean(o_hat == sigma)
    assert acc <= prop1_bound(a, p, top_k_pmax(3, N - 1)) + 3e-3


def test_consequences():
    assert rotation_probability_upper_bound(10, 50) == pytest.approx(0.2)
    assert rotation_probability_upper_bound(100, 50) == 1.0
    assert top_k_pmax(3, 19) == pytest.approx(1 / 3)
    assert top_k_pmax(3, 2) == pytest.approx(1 / 2)
