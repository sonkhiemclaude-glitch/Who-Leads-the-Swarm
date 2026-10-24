"""One-hop topology estimates from positions (Section 5.2-5.4: range estimate R-hat)."""
from __future__ import annotations

import numpy as np


def neighbours(pos_t: np.ndarray, range_m: float) -> np.ndarray:
    """Boolean adjacency (N x N) at one instant: distance < range_m, no self-loops."""
    if pos_t.ndim != 2 or pos_t.shape[1] != 3:
        raise ValueError("pos_t must have shape [N, 3]")
    diff = pos_t[:, None, :] - pos_t[None, :, :]
    adj = np.sqrt((diff ** 2).sum(-1)) < range_m
    np.fill_diagonal(adj, False)
    return adj


def adjacency_at(traj: np.ndarray, t: float, range_m: float) -> np.ndarray:
    """Adjacency at time t using the last 1 s sample (positions are known per second)."""
    k = min(max(int(t), 0), traj.shape[0] - 1)
    return neighbours(traj[k], range_m)


def two_hop(adj: np.ndarray, i: int) -> list[int]:
    """Nodes within two hops of i, excluding i."""
    reach = adj[i] | (adj[i].astype(int) @ adj.astype(int) > 0)
    reach[i] = False
    return [int(j) for j in np.flatnonzero(reach)]


def mean_degree(traj: np.ndarray, range_m: float, t: float = 0.0) -> float:
    return float(adjacency_at(traj, t, range_m).sum(1).mean())
