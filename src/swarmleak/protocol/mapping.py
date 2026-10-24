"""Agent-to-UAV mapping, rotation, padding and cover: from a logical trace to physical messages.

Every decision uses information a UAV has at that moment (Section 5.1):
- rotation successor, handover time and decoy assignments are committed in the plan messages;
- decoys fire after their own plan message is sent + 0.05 s assumed delivery (audit M14);
- spread decoys, mirrors and constant-rate cover follow keyed per-node schedules;
- cover sizes come from each UAV's past observations (SizeMemory).
Positions are taken from the trajectory, standing in for position sharing that is not simulated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from typing import Any

import numpy as np

from ..data.schemas import PhysicalMessage, Trace
from ..mobility.topology import adjacency_at, two_hop
from ..utils.random import rng_for
from .config import DefenseSpec, ProtocolConfig
from .cover import CoverResult, schedule_cover
from .energy import RadioModel
from .rotation import RotationPolicy
from .size_memory import SizeMemory


@dataclass
class MappingResult:
    messages: list[PhysicalMessage]
    orch: list[tuple[float, int]]  # (t_start, UAV hosting the orchestrator)
    decoy_starts: list[tuple[float, int]] = field(default_factory=list)
    cover: CoverResult | None = None


def pad_messages(msgs: list[PhysicalMessage], chunk_b: int) -> list[PhysicalMessage]:
    """Split and pad every message into ceil(size/chunk) chunks of `chunk` bytes, 1 ms apart."""
    if chunk_b <= 0:
        raise ValueError("chunk size must be positive")
    out: list[PhysicalMessage] = []
    for m in msgs:
        k = max(1, int(np.ceil(m.size / chunk_b)))
        out += [PhysicalMessage(m.t + 0.001 * c, m.src, m.dst, chunk_b, m.kind, m.real, m.gid) for c in range(k)]
    return out


def build_physical(trace: Trace, traj: np.ndarray, defense: DefenseSpec, budget_j: float,
                   cfg: ProtocolConfig, radio: RadioModel, seed: int, mission: int) -> MappingResult:
    n, T = trace.n_agents, trace.duration
    if traj.shape[1] != n:
        raise ValueError(f"trajectory has {traj.shape[1]} UAVs but trace has {n} agents")
    rng = rng_for(seed, "mapping", mission)
    policy = RotationPolicy(cfg.rotation_delay_s, cfg.rotation_top_k)
    perm = rng.permutation(n)  # perm[agent] = UAV
    orch: list[tuple[float, int]] = [(0.0, int(perm[0]))]
    bytes_sent = np.zeros(n)
    sizes = SizeMemory(n, cfg.size_profile_b, cfg.size_memory_min)
    out: list[PhysicalMessage] = []
    gid = 0
    pcs = list(trace.plan_changes)
    decoy_assign: dict[int, set[int]] = {}
    decoy_starts: list[tuple[float, int]] = []
    targeted = defense.cover == "targeted"
    pc_idx = -1
    pending_id = -1
    serial = count()
    events: list[tuple[float, int, int, str, Any]] = []

    def push(t: float, priority: int, kind: str, value: Any) -> None:
        if 0 <= t < T:
            heappush(events, (t, priority, next(serial), kind, value))

    for k, tau in enumerate(pcs):
        push(tau, 1, "plan_change", k)
    for msg in trace.messages:
        push(msg.t, 2, "message", msg)

    while events:
        t, _, _, kind, value = heappop(events)
        if kind == "handover":
            event_id, succ = value
            if event_id != pending_id:
                continue  # a newer plan replaced this handover before it started
            pending_id = -1
            cur = int(perm[0])
            if succ == cur:
                continue
            out.append(PhysicalMessage(t, cur, succ, sizes.draw(cur, "plan", rng, t), "handover_orch", True, gid))
            gid += 1
            out.append(PhysicalMessage(t + float(rng.uniform(0.05, 0.3)), succ, cur,
                                       sizes.draw(succ, "report", rng, t), "handover_worker", True, gid))
            gid += 1
            # Replay abstraction: switch after the assumed 50 ms state-delivery delay;
            # both mapping and labels switch at exactly the same instant.
            push(t + 0.05, 0, "switch", succ)
        elif kind == "switch":
            succ = int(value)
            cur = int(perm[0])
            a_succ = int(np.flatnonzero(perm == succ)[0])
            perm[0], perm[a_succ] = succ, cur
            orch.append((t, succ))
        elif kind == "plan_change":
            pc_idx = int(value)
            cur = int(perm[0])
            adj = adjacency_at(traj, t, cfg.range_est_m)
            if defense.rotate and pc_idx >= 1:
                pending_id = pc_idx
                succ = policy.choose_successor(cur, adj, bytes_sent, rng)
                push(t + policy.draw_delay(rng), 0, "handover", (pc_idx, succ))
            if targeted:
                ring = two_hop(adj, cur)
                decoy_assign[pc_idx] = {int(j) for j in rng.permutation(ring)[: cfg.n_cotimed_decoys]}
        else:
            m = value
            s, d = int(perm[m.src]), int(perm[m.dst])
            bytes_sent[s] += m.size
            sizes.see(s, m.kind, m.size, m.t)
            sizes.see(d, m.kind, m.size, m.t)
            if targeted and m.kind == "plan" and d in decoy_assign.get(pc_idx, set()):
                decoy_starts.append((m.t + cfg.decoy_delivery_s + float(rng.uniform(*cfg.decoy_start_s)), d))
                decoy_assign[pc_idx].discard(d)
            out.append(PhysicalMessage(m.t, s, d, m.size, m.kind, True, gid))
            gid += 1

    if defense.pad:
        out = pad_messages(out, cfg.pad_chunk_b)
    cover = None
    if defense.cover is not None:
        cover = schedule_cover(trace, decoy_starts, traj, defense, budget_j, cfg, radio, sizes, seed, mission)
        out += cover.messages
    # security overhead on every datagram (Section 3.1)
    out = [PhysicalMessage(x.t, x.src, x.dst, x.size + radio.security_overhead_b, x.kind, x.real, x.gid)
           for x in out if 0 <= x.t < T]
    out.sort(key=lambda x: (x.t, x.gid))
    return MappingResult(out, orch, decoy_starts, cover)
