"""3D Gauss-Markov mobility (Section 4.2, Table 2).

Speed, heading and pitch each follow  x_n = a x_{n-1} + (1-a) mean + sqrt(1-a^2) N(0, sigma),
as in ns-3's GaussMarkovMobilityModel. The mean heading is steered towards the area centre when a
UAV is within `edge` metres of a boundary; the mean pitch is steered towards mid-altitude.
Initial positions are uniform in the central 40% of each horizontal axis. Output: [T+1, N, 3]
positions sampled every dt seconds (replayed in ns-3 as 1 s waypoints).
"""
from __future__ import annotations

import numpy as np

from ..constants import ALTITUDE_M, AREA_M, GM_EDGE_M, GM_HEADING_SIGMA, GM_MEMORY, GM_PITCH_SIGMA, GM_SPEED_CLIP, GM_SPEED_SIGMA


def gauss_markov_3d(n_nodes: int, duration: float, rng: np.random.Generator, dt: float = 1.0,
                    area: tuple[float, float] = (AREA_M, AREA_M), alt: tuple[float, float] = ALTITUDE_M,
                    mean_speed: float = 15.0, speed_sigma: float = GM_SPEED_SIGMA, alpha: float = GM_MEMORY,
                    heading_sigma: float = GM_HEADING_SIGMA, pitch_sigma: float = GM_PITCH_SIGMA,
                    edge: float = GM_EDGE_M) -> np.ndarray:
    if n_nodes < 1 or duration <= 0 or dt <= 0:
        raise ValueError("n_nodes >= 1, duration > 0 and dt > 0 are required")
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must be in [0, 1)")
    steps = int(round(duration / dt)) + 1
    pos = np.zeros((steps, n_nodes, 3))
    pos[0, :, 0] = rng.uniform(0.3 * area[0], 0.7 * area[0], n_nodes)
    pos[0, :, 1] = rng.uniform(0.3 * area[1], 0.7 * area[1], n_nodes)
    pos[0, :, 2] = rng.uniform(alt[0], alt[1], n_nodes)
    s = np.full(n_nodes, float(mean_speed))
    d = rng.uniform(0, 2 * np.pi, n_nodes)
    p = np.zeros(n_nodes)
    d_mean = d.copy()
    sq = np.sqrt(1 - alpha ** 2)
    centre = np.array([area[0] / 2, area[1] / 2, (alt[0] + alt[1]) / 2])
    lo, hi = GM_SPEED_CLIP
    for k in range(1, steps):
        x = pos[k - 1]
        near = (x[:, 0] < edge) | (x[:, 0] > area[0] - edge) | (x[:, 1] < edge) | (x[:, 1] > area[1] - edge)
        to_c = np.arctan2(centre[1] - x[:, 1], centre[0] - x[:, 0])
        d_mean = np.where(near, to_c, d_mean)
        p_mean = np.clip((centre[2] - x[:, 2]) / max(alt[1] - alt[0], 1e-9), -0.3, 0.3)
        s = alpha * s + (1 - alpha) * mean_speed + sq * rng.normal(0, speed_sigma, n_nodes)
        s = np.clip(s, lo * mean_speed, hi * mean_speed)
        d = alpha * d + (1 - alpha) * d_mean + sq * rng.normal(0, heading_sigma, n_nodes)
        p = alpha * p + (1 - alpha) * p_mean + sq * rng.normal(0, pitch_sigma, n_nodes)
        v = np.stack([s * np.cos(d) * np.cos(p), s * np.sin(d) * np.cos(p), s * np.sin(p)], axis=1)
        nx = x + v * dt
        nx[:, 0] = np.clip(nx[:, 0], 0, area[0])
        nx[:, 1] = np.clip(nx[:, 1], 0, area[1])
        nx[:, 2] = np.clip(nx[:, 2], alt[0], alt[1])
        pos[k] = nx
    return pos
