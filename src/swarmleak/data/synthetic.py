"""Scripted mission timing and the parametric SURROGATE workload (Section 4.1).

Mission timing (phase sequence and durations, report periods, replanning rates) is scripted for
both the surrogate and the real-LLM generator; each mission draws its own timing profile, every
phase's report period and replanning rate being the base value times Unif[0.6, 1.6].

`surrogate_trace` is a parametric stand-in for LLM-agent traffic used ONLY to test the pipeline
and for sensitivity checks. It is not the measurement the paper's RQ1 requires; any output made
from it is a synthetic smoke-test output, not a paper result.
"""
from __future__ import annotations

import numpy as np

from ..constants import BASE_REPLAN_RATE_HZ, BASE_REPORT_PERIOD_S, PHASE_DURATION_S, PHASES, PROFILE_RANGE, RETURN_PHASE_S
from .schemas import Message, Phase, Trace

# Surrogate message sizes in bytes (assumptions; A2A-style JSON envelopes are a few hundred bytes).
SURROGATE_REPORT_MEDIAN_B = {"search": 520, "track": 760, "regroup": 460, "return": 380}
SURROGATE_PLAN_MEDIAN_B = 900
SURROGATE_ACK_MEDIAN_B = 260
SURROGATE_SIGMA = 0.35


def mission_profile(rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """Per-mission timing profile (Section 4.1, Table 2)."""
    lo, hi = PROFILE_RANGE
    return {"report_period": {p: BASE_REPORT_PERIOD_S[p] * float(rng.uniform(lo, hi)) for p in PHASES},
            "replan_rate": {p: BASE_REPLAN_RATE_HZ[p] * float(rng.uniform(lo, hi)) for p in PHASES}}


def mission_phases(rng: np.random.Generator, duration: float) -> list[Phase]:
    """Phase sequence: search -> track -> (search | regroup) ..., durations Unif[60,140] s,
    final 40 s return phase (Table 2)."""
    if duration <= RETURN_PHASE_S + 1:
        raise ValueError(f"mission duration must exceed {RETURN_PHASE_S + 1} s")
    seq = ["search"]
    while len(seq) < 12:
        last = seq[-1]
        if last == "search":
            seq.append("track")
        elif last == "track":
            seq.append(str(rng.choice(["search", "regroup"], p=[0.6, 0.4])))
        else:
            seq.append(str(rng.choice(["search", "track"])))
    out: list[Phase] = []
    t = 0.0
    end_work = duration - RETURN_PHASE_S
    for ph in seq:
        if t >= end_work:
            break
        dur = float(rng.uniform(*PHASE_DURATION_S))
        out.append(Phase(t, min(t + dur, end_work), ph))
        t += dur
    out.append(Phase(out[-1].end, duration, "return"))
    return out


def _lognormal(rng: np.random.Generator, median: float, sigma: float = SURROGATE_SIGMA,
               lo: int = 80, hi: int = 12000) -> int:
    return int(np.clip(rng.lognormal(np.log(median), sigma), lo, hi))


def surrogate_trace(n_agents: int, duration: float, rng: np.random.Generator) -> Trace:
    """Parametric orchestrator-worker workload (pipeline validation only)."""
    if n_agents < 2:
        raise ValueError("n_agents must be >= 2")
    phases = mission_phases(rng, duration)
    prof = mission_profile(rng)
    raw: list[tuple[float, int, int, int, str]] = []
    plan_changes: list[float] = []

    def plan_burst(t0: float) -> None:
        plan_changes.append(t0)
        order = rng.permutation(np.arange(1, n_agents))
        t = t0
        for w in order:
            t += float(rng.uniform(0.02, 0.15))
            raw.append((t, 0, int(w), _lognormal(rng, SURROGATE_PLAN_MEDIAN_B), "plan"))
            raw.append((t + float(rng.uniform(0.2, 1.0)), int(w), 0,
                        _lognormal(rng, SURROGATE_ACK_MEDIAN_B, 0.2), "ack"))
        if n_agents > 2:
            for w in order:
                if rng.random() < 0.5:
                    peer = int(rng.choice([a for a in range(1, n_agents) if a != w]))
                    tq = t0 + float(rng.uniform(1.0, 6.0))
                    raw.append((tq, int(w), peer, _lognormal(rng, 400), "tool_call"))
                    raw.append((tq + float(rng.uniform(0.3, 1.5)), peer, int(w), _lognormal(rng, 700), "tool_result"))

    for ph in phases:
        plan_burst(ph.start + float(rng.uniform(0.0, 0.5)))
        rate = prof["replan_rate"][ph.name]
        t = ph.start
        while rate > 0:
            t += float(rng.exponential(1 / rate))
            if t >= ph.end - 5:
                break
            plan_burst(t)
        per = prof["report_period"][ph.name]
        for w in range(1, n_agents):
            t = ph.start + float(rng.uniform(0, per))
            while t < ph.end:
                raw.append((t, w, 0, _lognormal(rng, SURROGATE_REPORT_MEDIAN_B[ph.name]), "report"))
                if rng.random() < 0.3:
                    raw.append((t + float(rng.uniform(0.3, 1.2)), 0, w, _lognormal(rng, 300), "feedback"))
                t += per * float(rng.uniform(0.8, 1.2))
    raw = sorted((r for r in raw if r[0] < duration), key=lambda r: r[0])
    msgs = [Message(*r) for r in raw]
    return Trace(msgs, phases, sorted(plan_changes), float(duration), n_agents, "surrogate", prof).validate()
