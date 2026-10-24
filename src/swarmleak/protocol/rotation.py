"""Orchestrator role rotation (Section 5.2).

At every plan change except the mission-start plan, the orchestrator picks a successor and a
handover delay u ~ Unif[4, 20] s and commits both inside the encrypted plan messages. Candidates
are scored by

    score(j) = d_j / max_j' d_j'  -  b_j / max_j' b_j'                      (Section 5.2)

(d_j: estimated one-hop degree; b_j: coordination bytes j has originated). The successor is drawn
uniformly from the top-k (k = 3) candidates, which gives p_max <= 1/3 in Proposition 1.
At tau + u the two hosts exchange states (plan-sized and report-sized messages) and swap agents.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def successor_scores(cand: list[int], degree: np.ndarray, bytes_sent: np.ndarray) -> np.ndarray:
    """Score of Section 5.2 for each candidate (denominators floored at 1)."""
    d = np.array([degree[j] for j in cand], float)
    b = np.array([bytes_sent[j] for j in cand], float)
    return d / max(d.max(initial=0.0), 1.0) - b / max(b.max(initial=0.0), 1.0)


@dataclass(frozen=True)
class RotationPolicy:
    delay_s: tuple[float, float] = (4.0, 20.0)
    top_k: int = 3

    def choose_successor(self, current: int, adj: np.ndarray, bytes_sent: np.ndarray,
                         rng: np.random.Generator) -> int:
        n = adj.shape[0]
        cand = [j for j in range(n) if j != current]
        if not cand:
            raise ValueError("rotation needs at least two UAVs")
        score = successor_scores(cand, adj.sum(1), bytes_sent)
        # tiny random jitter breaks exact score ties without favouring low indices
        order = np.argsort(-(score + 1e-9 * rng.random(len(cand))))
        top = order[: min(self.top_k, len(cand))]
        return int(cand[int(rng.choice(top))])

    def draw_delay(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(*self.delay_s))
