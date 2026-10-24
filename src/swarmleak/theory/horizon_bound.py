"""Proposition 1 (role accuracy at a horizon, Section 5.5) and its consequences.

Assumptions (paper): (A1) Pr[C and E] >= alpha0 * p_E (C = {o_hat = o(t)}, E = {>= 1 rotation in
(t, t+D]}); (A2) at most one rotation in (t, t+D]; p_max bounds the probability of any single UAV
being the successor given the adversary's observations and E.
"""
from __future__ import annotations

from ..utils.validation import check_positive, check_probability


def prop1_bound(alpha0: float, p_e: float, p_max: float) -> float:
    """Pr[o_hat = sigma] <= alpha0 (1 - p_E) + p_E * min{1 - alpha0, p_max}   (Proposition 1)."""
    a, p, q = check_probability("alpha0", alpha0), check_probability("p_E", p_e), check_probability("p_max", p_max)
    return a * (1 - p) + p * min(1 - a, q)


def prop1_exact(alpha0: float, p_e: float, n_uavs: int) -> float:
    """Exact value alpha0 (1 - p_E) + p_E (1 - alpha0) / (N - 1) when C and E are independent and the
    successor is uniform on the other N - 1 UAVs, conditionally independent of o_hat (Proposition 1)."""
    a, p = check_probability("alpha0", alpha0), check_probability("p_E", p_e)
    if n_uavs < 2:
        raise ValueError("N must be >= 2")
    return a * (1 - p) + p * (1 - a) / (n_uavs - 1)


def guaranteed_reduction(alpha0: float, p_e: float, p_max: float) -> float:
    """alpha0 - bound = p_E (alpha0 - min{1 - alpha0, p_max}); positive only if alpha0 exceeds the min."""
    return alpha0 - prop1_bound(alpha0, p_e, p_max)


def rotation_probability_upper_bound(horizon_s: float, mean_interval_s: float) -> float:
    """p_E <= D / L for a stationary rotation process with mean interval L (capped at 1)."""
    check_positive("horizon", horizon_s, allow_zero=True)
    check_positive("mean interval", mean_interval_s)
    return min(1.0, horizon_s / mean_interval_s)


def top_k_pmax(k: int, n_candidates: int) -> float:
    """Uniform choice among the top-k candidates gives p_max <= 1/min(k, N-1) for any adversary."""
    if k < 1 or n_candidates < 1:
        raise ValueError("k and n_candidates must be >= 1")
    return 1.0 / min(k, n_candidates)
