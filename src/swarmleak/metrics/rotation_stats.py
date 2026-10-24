"""Rotation statistics used with Proposition 1 (Section 5.5): mean interval L, p_E and the
frequency of two or more rotations within a horizon (violations of assumption A2)."""
from __future__ import annotations

import numpy as np


def rotation_times(orch: np.ndarray) -> np.ndarray:
    o = np.asarray(orch, float)
    return o[1:, 0] if len(o) > 1 else np.zeros(0)


def rotation_stats(orch: np.ndarray, window_ends: np.ndarray, horizon: float) -> dict[str, float]:
    """For each window end t: E = {>= 1 rotation in (t, t + horizon]}. Returns p_E, Pr[>= 2] and the
    mean interval between rotations (NaN with fewer than one rotation)."""
    ts = rotation_times(orch)
    ends = np.asarray(window_ends, float)
    if len(ends) == 0:
        return {"p_E": float("nan"), "p_two_plus": float("nan"), "mean_interval_s": float("nan")}
    counts = np.array([np.sum((ts > t) & (ts <= t + horizon)) for t in ends])
    mean_int = float(np.mean(np.diff(np.r_[0.0, ts]))) if len(ts) else float("nan")
    return {"p_E": float(np.mean(counts >= 1)), "p_two_plus": float(np.mean(counts >= 2)),
            "mean_interval_s": mean_int}
