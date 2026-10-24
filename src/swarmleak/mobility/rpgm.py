"""Reference-point group mobility variant (Section 4.2, Table 2).

The group centre follows 3D Gauss-Markov motion; each UAV keeps an offset inside a sphere of
`radius` metres that drifts as a bounded random walk (`member_speed` m/s, vertical drift x0.3).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import ALTITUDE_M, AREA_M, RPGM_DRIFT_MPS, RPGM_RADIUS_M
from .gauss_markov import gauss_markov_3d


def rpgm_3d(n_nodes: int, duration: float, rng: np.random.Generator, dt: float = 1.0,
            area: tuple[float, float] = (AREA_M, AREA_M), alt: tuple[float, float] = ALTITUDE_M,
            mean_speed: float = 15.0, radius: float = RPGM_RADIUS_M, member_speed: float = RPGM_DRIFT_MPS,
            **gm_kw: Any) -> np.ndarray:
    if radius <= 0:
        raise ValueError("radius must be positive")
    centre = gauss_markov_3d(1, duration, rng, dt=dt, area=area, alt=alt, mean_speed=mean_speed, **gm_kw)[:, 0, :]
    steps = centre.shape[0]
    off = rng.normal(0, radius / 2, (n_nodes, 3))
    off[:, 2] *= 0.3
    scale = np.array([1.0, 1.0, 0.3])
    pos = np.zeros((steps, n_nodes, 3))
    for k in range(steps):
        off = off + rng.normal(0, member_speed * dt, off.shape) * scale
        r = np.linalg.norm(off, axis=1, keepdims=True)
        off = np.where(r > radius, off * radius / np.maximum(r, 1e-12), off)
        q = centre[k] + off
        q[:, 0] = np.clip(q[:, 0], 0, area[0])
        q[:, 1] = np.clip(q[:, 1], 0, area[1])
        q[:, 2] = np.clip(q[:, 2], alt[0], alt[1])
        pos[k] = q
    return pos
