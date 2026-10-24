"""Structured representations of the objects in the paper (Sections 3-4).

- Message: one logical agent-to-agent message (Section 3.1 tuple (t, a_s, a_d, l, kappa)).
- Trace: a mission's logical messages plus ground truth (phases, plan-change instants).
- PhysicalMessage: a message after agent->UAV mapping and defenses (real or cover).
- FRAME_DTYPE: the adversary's view, one row per decoded data MPDU (Section 3.2).
- RunRecord: everything one simulation produces, saved as a compressed .npz.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import PHASES

MESSAGE_KINDS = ("plan", "ack", "report", "feedback", "tool_call", "tool_result")
PHYSICAL_KINDS = MESSAGE_KINDS + (
    "handover_orch", "handover_worker", "cover_plan", "cover_ack", "cover_report", "cover_const")

FRAME_DTYPE = np.dtype([("t", "f8"), ("size", "u4"), ("tx", "i4"), ("rx", "i4"), ("seq", "u2"),
                        ("retry", "u1"), ("mon", "u2"), ("rssi", "f4")])
"""Frame log row. tx/rx are UAV indices; rx = -1 for broadcast frames or unknown addresses."""


@dataclass(frozen=True)
class Message:
    t: float
    src: int
    dst: int
    size: int
    kind: str

    def validate(self, n_agents: int) -> None:
        if not np.isfinite(self.t) or self.t < 0:
            raise ValueError(f"message time must be finite and >= 0, got {self.t}")
        if not (0 <= self.src < n_agents and 0 <= self.dst < n_agents):
            raise ValueError(f"agent index out of range [0, {n_agents}): {self.src}->{self.dst}")
        if self.src == self.dst:
            raise ValueError("a message must have different sender and receiver agents")
        if self.size <= 0:
            raise ValueError(f"message size must be positive, got {self.size}")
        if self.kind not in MESSAGE_KINDS:
            raise ValueError(f"unknown message kind {self.kind!r}; expected one of {MESSAGE_KINDS}")


@dataclass(frozen=True)
class Phase:
    start: float
    end: float
    name: str


@dataclass
class Trace:
    """Logical coordination trace of one mission. Agent 0 is the orchestrator agent."""

    messages: list[Message]
    phases: list[Phase]
    plan_changes: list[float]
    duration: float
    n_agents: int
    source: str
    profile: dict[str, dict[str, float]] | None = None
    generator: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Trace:
        if self.n_agents < 2:
            raise ValueError("a trace needs at least 2 agents (orchestrator + 1 worker)")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        last = -np.inf
        for m in self.messages:
            m.validate(self.n_agents)
            if m.t < last:
                raise ValueError("messages must be sorted by time")
            last = m.t
        if not self.phases:
            raise ValueError("a trace needs at least one phase")
        prev_end = self.phases[0].start
        for p in self.phases:
            if p.name not in PHASES:
                raise ValueError(f"unknown phase {p.name!r}")
            if p.end < p.start or abs(p.start - prev_end) > 1e-6:
                raise ValueError("phases must be contiguous and non-decreasing")
            prev_end = p.end
        if any(b < a for a, b in zip(self.plan_changes, self.plan_changes[1:])):
            raise ValueError("plan_changes must be sorted")
        return self

    def phase_at(self, t: float) -> str:
        return phase_at(self.phases, t)

    def to_dict(self) -> dict[str, Any]:
        return {"messages": [asdict(m) for m in self.messages],
                "phases": [[p.start, p.end, p.name] for p in self.phases],
                "plan_changes": list(self.plan_changes), "duration": self.duration,
                "n_agents": self.n_agents, "source": self.source, "profile": self.profile,
                "generator": self.generator}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        required = ("messages", "phases", "plan_changes", "duration", "n_agents")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"trace is missing fields {missing}")
        msgs = [Message(float(m["t"]), int(m["src"]), int(m["dst"]), int(m["size"]), str(m["kind"]))
                for m in d["messages"]]
        phases = [Phase(float(a), float(b), str(n)) for a, b, n in d["phases"]]
        return cls(msgs, phases, [float(x) for x in d["plan_changes"]], float(d["duration"]),
                   int(d["n_agents"]), str(d.get("source", "unknown")), d.get("profile"),
                   dict(d.get("generator") or {})).validate()

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load_json(cls, path: str | Path) -> Trace:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def phase_at(phases: list[Phase] | list[tuple[float, float, str]], t: float) -> str:
    """Phase active at time t; the last phase extends to the end of the mission."""
    items: list[tuple[float, float, str]] = []
    for p in phases:
        if isinstance(p, Phase):
            items.append((p.start, p.end, p.name))
        else:
            items.append((float(p[0]), float(p[1]), str(p[2])))
    if not items:
        raise ValueError("no phases")
    for a, b, name in items:
        if a <= t < b:
            return name
    return items[-1][2]


@dataclass
class PhysicalMessage:
    t: float
    src: int
    dst: int
    size: int
    kind: str
    real: bool
    gid: int  # groups padding chunks of one logical message; -1 for cover


def empty_frames(n: int = 0) -> np.ndarray:
    return np.zeros(n, dtype=FRAME_DTYPE)


def dedupe_frames(frames: np.ndarray) -> np.ndarray:
    """Merge frames decoded by several monitors (audit M20).

    Key: (tx, seq, size, retry, time rounded to 0.1 ms). Retransmissions keep the retry flag and a
    different time, so they are not merged with the original transmission.
    """
    if len(frames) == 0:
        return frames
    key = np.stack([frames["tx"].astype(np.int64), frames["seq"].astype(np.int64),
                    frames["size"].astype(np.int64), frames["retry"].astype(np.int64),
                    np.round(frames["t"] * 1e4).astype(np.int64)], axis=1)
    _, idx = np.unique(key, axis=0, return_index=True)
    kept = frames[np.sort(idx)]
    return kept[np.argsort(kept["t"], kind="stable")]


@dataclass
class RunRecord:
    """Output of one simulation (one mission, one defense configuration, one backend)."""

    frames: np.ndarray  # FRAME_DTYPE, deduplicated adversary view
    rx_time: np.ndarray  # delivery time per physical message (NaN if undelivered)
    energy_j: np.ndarray  # radio energy consumed per UAV
    msg_t: np.ndarray
    msg_src: np.ndarray
    msg_dst: np.ndarray
    msg_size: np.ndarray
    msg_real: np.ndarray
    msg_gid: np.ndarray
    msg_kind: np.ndarray
    orch: np.ndarray  # [[t_start, uav], ...]
    plan_changes: np.ndarray
    phases: list[tuple[float, float, str]]
    meta: dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.meta["n"])

    @property
    def duration(self) -> float:
        return float(self.meta["duration"])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, frames=self.frames, rx_time=self.rx_time, energy_j=self.energy_j,
                            msg_t=self.msg_t, msg_src=self.msg_src, msg_dst=self.msg_dst,
                            msg_size=self.msg_size, msg_real=self.msg_real, msg_gid=self.msg_gid,
                            msg_kind=self.msg_kind, orch=self.orch, plan_changes=self.plan_changes,
                            phases=json.dumps(self.phases), meta=json.dumps(self.meta, default=str))

    @classmethod
    def load(cls, path: str | Path) -> RunRecord:
        with np.load(path, allow_pickle=False) as d:
            return cls(frames=dedupe_frames(d["frames"]), rx_time=d["rx_time"], energy_j=d["energy_j"],
                       msg_t=d["msg_t"], msg_src=d["msg_src"], msg_dst=d["msg_dst"],
                       msg_size=d["msg_size"], msg_real=d["msg_real"].astype(bool), msg_gid=d["msg_gid"],
                       msg_kind=d["msg_kind"], orch=d["orch"], plan_changes=d["plan_changes"],
                       phases=[tuple(p) for p in json.loads(str(d["phases"]))],
                       meta=json.loads(str(d["meta"])))


def physical_arrays(msgs: list[PhysicalMessage]) -> dict[str, np.ndarray]:
    return {"msg_t": np.array([m.t for m in msgs], float), "msg_src": np.array([m.src for m in msgs], int),
            "msg_dst": np.array([m.dst for m in msgs], int), "msg_size": np.array([m.size for m in msgs], int),
            "msg_real": np.array([m.real for m in msgs], bool), "msg_gid": np.array([m.gid for m in msgs], int),
            "msg_kind": np.array([m.kind for m in msgs], dtype="U16")}
