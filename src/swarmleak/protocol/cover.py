"""Budgeted, rate-limited cover traffic (Section 5.3, Algorithm 1) and cover baselines.

Message kinds
  decoy burst   plan-sized messages from j to all its current one-hop neighbours in random order,
                cumulative gaps Unif[0.02, 0.15] s; each receiver replies ack-sized after Unif[0.2, 1.0] s
  mirror        one report-sized message to a random one-hop neighbour
  const         1024-byte message, round-robin over one-hop neighbours (constant-rate baseline)

Schedulers
  targeted  shares 0.4 / 0.3 / 0.3 of B_i for co-timed decoys / spread decoys / mirrors
  uniform   all of B_i on spread decoys
  const     rate rho = B / (T [c_tx(1024) + c_rx(1024)]) per UAV

Admission (Algorithm 1, function Admit): both endpoints within the share cap and within B_i, the
sender holds a token (bucket capacity 25, refill 3/s), and the directed link has been idle for
0.5 s. The ack reply is itself a cover message, so the same checks apply to its sender (audit M10).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from typing import Any

import numpy as np

from ..data.schemas import PhysicalMessage, Trace, phase_at
from ..mobility.topology import adjacency_at, mean_degree
from ..utils.random import rng_for
from .config import DefenseSpec, ProtocolConfig
from .energy import RadioModel
from .size_memory import SizeMemory


class TokenBucket:
    """Token bucket with monotone refill time (audit M11)."""

    def __init__(self, capacity: float, rate_hz: float) -> None:
        self.capacity = float(capacity)
        self.rate = float(rate_hz)
        self.tokens = float(capacity)
        self.t_last = 0.0

    def available(self, t: float) -> float:
        if t > self.t_last:
            self.tokens = min(self.capacity, self.tokens + (t - self.t_last) * self.rate)
            self.t_last = t
        return self.tokens

    def take(self) -> None:
        if self.tokens < 1.0:
            raise RuntimeError("token bucket empty")
        self.tokens -= 1.0


@dataclass
class CoverLedger:
    """Estimated incremental energy per node and per stage (for budget feasibility checks)."""

    n: int
    budget_j: float
    caps: dict[str, float]
    spent: dict[str, np.ndarray] = field(default_factory=dict)
    total: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        self.spent = {s: np.zeros(self.n) for s in self.caps}
        self.total = np.zeros(self.n)

    def fits(self, stage: str, i: int, need_i: float, j: int, need_j: float) -> bool:
        cap = self.caps[stage]
        return (self.spent[stage][i] + need_i <= cap + 1e-15 and self.spent[stage][j] + need_j <= cap + 1e-15
                and self.total[i] + need_i <= self.budget_j + 1e-15 and self.total[j] + need_j <= self.budget_j + 1e-15)

    def charge(self, stage: str, i: int, need_i: float, j: int, need_j: float) -> None:
        self.spent[stage][i] += need_i
        self.spent[stage][j] += need_j
        self.total[i] += need_i
        self.total[j] += need_j


@dataclass
class CoverResult:
    messages: list[PhysicalMessage]
    ledger: CoverLedger
    n_candidates: int
    n_rejected: int


def schedule_cover(trace: Trace, decoy_starts: list[tuple[float, int]], traj: np.ndarray, defense: DefenseSpec,
                   budget_j: float, cfg: ProtocolConfig, radio: RadioModel, sizes: SizeMemory,
                   seed: int, mission: int) -> CoverResult:
    """Generate cover candidates causally and admit them in time order (Algorithm 1)."""
    if defense.cover is None:
        raise ValueError("defense has no cover traffic")
    if budget_j < 0:
        raise ValueError("budget must be non-negative")
    n, T = trace.n_agents, trace.duration
    mode = defense.cover
    any_hop = defense.cover_hops == "any"
    per = (trace.profile or {}).get("report_period") or {}
    rng = rng_for(seed, "cover-admit", mission)

    def dest_pool(i: int, t: float) -> list[int]:
        if any_hop:
            return [j for j in range(n) if j != i]
        return [int(j) for j in np.flatnonzero(adjacency_at(traj, t, cfg.range_est_m)[i])]

    # mean estimated degree at mission start (known to every UAV from shared positions)
    d0 = max(mean_degree(traj, cfg.range_est_m, 0.0), 1.0)
    burst_cost = d0 * (radio.tx_cost(cfg.size_profile_b["plan"]) + radio.rx_cost(cfg.size_profile_b["ack"]))

    cand: list[tuple[float, str, str, int]] = []  # (time, stage, kind, node)
    if mode in ("uniform", "targeted"):
        share = 1.0 if mode == "uniform" else cfg.targeted_shares["spread"]
        mu = share * budget_j / (T * burst_cost)  # spread-decoy rate (Section 5.3)
        for i in range(n):
            r = rng_for(seed, "spread", mission, i)  # keyed per node (audit M13)
            t = float(r.exponential(1 / mu)) if mu > 0 else T
            while t < T:
                cand.append((t, "spread", "burst", i))
                t += float(r.exponential(1 / mu))
    if mode == "targeted":
        cand += [(t, "cotimed", "burst", j) for t, j in decoy_starts]
        base = 8.0
        for i in range(n):
            r = rng_for(seed, "mirror", mission, i)
            t = float(r.exponential(base))
            while t < T:
                cand.append((t, "mirror", "mirror", i))
                ph = phase_at(trace.phases, t)
                t += float(r.exponential(per.get(ph, _fallback_period(ph))))
    if mode == "const":
        rho = budget_j / (T * (radio.tx_cost(cfg.const_size_b) + radio.rx_cost(cfg.const_size_b)))
        for i in range(n):
            if rho <= 0:
                continue
            r = rng_for(seed, "const", mission, i)
            t = float(r.uniform(0, 1 / rho))
            while t < T:
                cand.append((t, "const", "const", i))
                t += 1 / rho
    cand.sort(key=lambda c: (c[0], c[3]))

    if mode == "targeted":
        caps = {s: cfg.targeted_shares[s] * budget_j for s in ("cotimed", "spread", "mirror")}
    else:
        caps = {"spread": budget_j, "const": budget_j}
    ledger = CoverLedger(n, budget_j, caps)
    buckets = [TokenBucket(cfg.bucket_capacity, cfg.bucket_rate_hz) for _ in range(n)]
    last_link: dict[tuple[int, int], float] = {}
    rr: dict[int, int] = defaultdict(int)
    out: list[PhysicalMessage] = []
    rejected = 0

    def link_free(i: int, j: int, t: float) -> bool:
        return t - last_link.get((i, j), -1e9) >= cfg.link_gap_s

    def admit(t: float, stage: str, i: int, j: int, size: int, kind: str) -> bool:
        # Each transmission, including an application ACK, is admitted at its own
        # timestamp. No token or energy is borrowed from a future timestamp.
        if not (0 <= t < T) or (not any_hop and j not in dest_pool(i, t)):
            return False
        need_i, need_j = radio.tx_cost(size), radio.rx_cost(size)
        if not ledger.fits(stage, i, need_i, j, need_j):
            return False
        if buckets[i].available(t) < 1 or not link_free(i, j, t):
            return False
        buckets[i].take()
        last_link[(i, j)] = t
        ledger.charge(stage, i, need_i, j, need_j)
        out.append(PhysicalMessage(t, i, j, size, "cover_" + kind, False, -1))
        return True

    serial = count()
    events: list[tuple[float, int, str, Any]] = []

    def push(t: float, kind: str, value: Any) -> None:
        if 0 <= t < T:
            heappush(events, (t, next(serial), kind, value))

    for t, stage, kind, i in cand:
        push(t, "candidate", (stage, kind, i))
    while events:
        t, _, event_kind, value = heappop(events)
        if event_kind == "transmission":
            stage, i, j, kind, reply = value
            size = cfg.const_size_b if kind == "const" else sizes.draw(i, kind, rng, t)
            if admit(t, stage, i, j, size, kind):
                if reply:
                    push(t + float(rng.uniform(*cfg.ack_delay_s)), "transmission", (stage, j, i, "ack", False))
            else:
                rejected += 1
            continue
        stage, kind, i = value
        pool = dest_pool(i, t)
        if not pool:
            rejected += 1
            continue
        if kind == "burst":
            tt = t
            for j in rng.permutation(pool):
                tt += float(rng.uniform(*cfg.burst_gap_s))
                push(tt, "transmission", (stage, i, int(j), "plan", True))
        elif kind == "mirror":
            push(t, "transmission", (stage, i, int(rng.choice(pool)), "report", False))
        else:
            j = pool[rr[i] % len(pool)]
            rr[i] += 1
            push(t, "transmission", (stage, i, j, "const", False))
    return CoverResult(out, ledger, len(cand), rejected)


def _fallback_period(phase: str) -> float:
    from ..constants import BASE_REPORT_PERIOD_S

    return float(BASE_REPORT_PERIOD_S.get(phase, 8.0))
