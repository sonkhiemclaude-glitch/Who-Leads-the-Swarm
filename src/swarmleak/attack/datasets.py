"""Labelled window datasets for the three secrets (Sections 3.3, 4.3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import HORIZONS_S, PHASE_WINDOW, PLAN_WINDOW, ROLE_WINDOW
from ..data.schemas import RunRecord, phase_at
from .features import role_features, window_features


def orch_at(orch: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """o(t): UAV hosting the orchestrator at time t (piecewise constant, right-continuous)."""
    orch = np.asarray(orch, float)
    if orch.ndim != 2 or orch.shape[1] != 2 or len(orch) == 0:
        raise ValueError("orch must be a non-empty [[t_start, uav], ...] array")
    i = np.searchsorted(orch[:, 0], t, side="right") - 1
    return orch[np.clip(i, 0, len(orch) - 1), 1].astype(int)


@dataclass
class RoleData:
    X: np.ndarray  # [K*n, 32]
    y: dict[float, np.ndarray]  # horizon -> [K*n] one-hot per window
    window_id: np.ndarray  # [K*n]
    window_start: np.ndarray  # [K]
    n: int


def role_dataset(r: RunRecord, W: float = ROLE_WINDOW[0], step: float = ROLE_WINDOW[1],
                 horizons: tuple[float, ...] = HORIZONS_S) -> RoleData:
    """Windows [t_k, t_k+W) with t_k + W + max(horizon) <= T, so every horizon uses the same
    windows (audit M1). Label for horizon D: o(t_k + W + D)."""
    n, T = r.n, r.duration
    starts = np.arange(0.0, T - W - max(horizons) + 1e-9, step)
    rows, wid = [], []
    ys: dict[float, list[np.ndarray]] = {h: [] for h in horizons}
    for k, t0 in enumerate(starts):
        rows.append(role_features(r.frames, n, t0, t0 + W))
        for h in horizons:
            y = np.zeros(n, int)
            y[int(orch_at(r.orch, t0 + W + h))] = 1
            ys[h].append(y)
        wid.append(np.full(n, k))
    if not rows:
        raise ValueError("mission too short for the role windows")
    return RoleData(np.vstack(rows), {h: np.concatenate(v) for h, v in ys.items()}, np.concatenate(wid), starts, n)


def phase_dataset(r: RunRecord, W: float = PHASE_WINDOW[0], step: float = PHASE_WINDOW[1]) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate window features with label phi(t_k + W/2) (Definition 2)."""
    starts = np.arange(0.0, r.duration - W + 1e-9, step)
    X = np.array([window_features(r.frames, r.n, t0, t0 + W) for t0 in starts])
    y = np.array([phase_at(r.phases, t0 + W / 2) for t0 in starts])
    return X, y


def plan_dataset(r: RunRecord, W: float = PLAN_WINDOW[0], step: float = PLAN_WINDOW[1]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short windows; label 1 if a plan change falls in [t_k, t_k+W); features + first differences."""
    pcs = np.asarray(r.plan_changes, float)
    starts = np.arange(0.0, r.duration - W + 1e-9, step)
    X, y = [], []
    prev = np.zeros(14)
    for t0 in starts:
        f = window_features(r.frames, r.n, t0, t0 + W)
        X.append(np.concatenate([f, f - prev]))
        prev = f
        y.append(int(np.any((pcs >= t0) & (pcs < t0 + W))))
    return np.array(X), np.array(y), starts
