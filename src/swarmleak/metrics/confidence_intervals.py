"""Uncertainty over missions (Sections 4.4, 6): Student-t CI and paired bootstrap."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def t_interval(values: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    """(mean, half-width) of a Student-t interval over missions; NaNs dropped.
    Half-width is NaN with fewer than two values."""
    x = np.asarray([v for v in values if v == v], float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) < 2:
        return float(x.mean()), float("nan")
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(h)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, level: float = 0.95,
                     seed: int = 0) -> dict[str, float]:
    """Mean paired difference a - b over missions with a percentile bootstrap CI (resampling
    missions). Pairs with a NaN on either side are dropped."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    keep = ~(np.isnan(x) | np.isnan(y))
    d = x[keep] - y[keep]
    if len(d) == 0:
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * (0.5 - level / 2), 100 * (0.5 + level / 2)])
    return {"mean_diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d))}
