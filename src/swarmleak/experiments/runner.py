"""Simulation runner: one run = (grid entry, mission) -> runs/<tag>.npz (Sections 4, 6).

Pairing across configurations (Section 6, 'Statistics'): every configuration of the same mission
uses the same trace, the same mobility seed and the same backend run number, so differences are
paired by mission.
"""
from __future__ import annotations

import csv
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..data.adapters import TraceDirectoryAdapter
from ..data.schemas import RunRecord, Trace, dedupe_frames, physical_arrays
from ..data.synthetic import surrogate_trace
from ..llm.agents import MockChatClient, generate_llm_trace
from ..mobility import gauss_markov_3d, rpgm_3d
from ..protocol.config import get_defense
from ..protocol.energy import RadioModel
from ..protocol.mapping import build_physical
from ..sim.base import SimSettings, get_backend
from ..utils.io import stable_hash
from ..utils.random import int_seed, rng_for
from .config import ExperimentConfig, GridEntry, Scenario
from .integrity import expected_runs, validate_run
from .provenance import run_metadata, source_digest

log = logging.getLogger(__name__)


def load_trace(cfg: ExperimentConfig, scen: Scenario, mission: int) -> Trace:
    if cfg.trace_source == "surrogate":
        return surrogate_trace(scen.n, scen.duration, rng_for(cfg.seed, "trace", scen.n, mission))
    if cfg.trace_source == "llm_mock":
        client = MockChatClient(seed=int_seed(cfg.seed, "mock-llm", scen.n, mission))
        return generate_llm_trace(client, scen.n, scen.duration, rng_for(cfg.seed, "trace", scen.n, mission),
                                  float(cfg.llm.get("latency_scale", 0.2)), float(cfg.llm.get("report_llm_prob", 1.0)))
    if cfg.trace_root is None:
        raise ValueError("data.root is required for real LLM traces")
    tr = TraceDirectoryAdapter(cfg.trace_root, cfg.trace_pattern).load(scen.n, mission)
    if abs(tr.duration - scen.duration) > 1e-6:
        raise ValueError(f"trace duration {tr.duration} != scenario duration {scen.duration} (n={scen.n}, m={mission})")
    return tr


def trace_is_synthetic(trace: Trace) -> bool:
    """A trace counts as synthetic if it came from the surrogate or the mock client, whatever the config says."""
    gen = trace.generator or {}
    return (trace.source in ("surrogate", "llm:mock") or bool(gen.get("mock"))
            or "SYNTHETIC" in str(gen.get("note", "")).upper())


def trajectories(cfg: ExperimentConfig, scen: Scenario, mission: int) -> np.ndarray:
    fn = gauss_markov_3d if scen.mobility == "gm" else rpgm_3d
    return fn(scen.n, scen.duration, rng_for(cfg.seed, "mobility", mission), mean_speed=scen.speed,
              area=(scen.area, scen.area), alt=scen.altitude)


def run_one(cfg: ExperimentConfig, entry: GridEntry, mission: int, overwrite: bool = False) -> Path:
    out = cfg.output_dir / "runs" / f"{entry.tag(mission)}.npz"
    scen = entry.scenario
    trace = load_trace(cfg, scen, mission)
    trace_hash = stable_hash(trace.to_dict())
    if out.exists() and not overwrite:
        validate_run(RunRecord.load(out), cfg, entry, mission, trace_hash)
        log.info("skip existing %s", out.name)
        return out
    t0 = time.time()
    traj = trajectories(cfg, scen, mission)
    defense = get_defense(entry.defense)
    radio = RadioModel(mtu_b=scen.mtu_b)
    mapped = build_physical(trace, traj, defense, entry.budget, cfg.protocol, radio, cfg.seed, mission)
    settings = SimSettings(routing=scen.routing, monitors=scen.monitor_positions(), warmup_s=scen.warmup_s,
                           tx_power_dbm=scen.tx_power_dbm, loss_exponent=scen.loss_exponent, mtu_b=scen.mtu_b,
                           seed=int_seed(cfg.seed, "backend"), run=int(mission), extra=dict(cfg.toy))
    backend = get_backend(cfg.backend)
    sim = backend.run(mapped.messages, traj, settings)
    synthetic = cfg.synthetic or trace_is_synthetic(trace) or sim.synthetic_backend
    meta = run_metadata(cfg.config_hash, synthetic, cfg.backend, cfg.trace_source,
                        experiment=cfg.name, tag=entry.tag(mission), config_key=entry.config_key(),
                        defense=entry.defense, budget=entry.budget, mission=mission, n=scen.n, duration=scen.duration,
                        speed=scen.speed, mobility=scen.mobility, monitors=scen.monitors, routing=scen.routing,
                        trace_model=trace.generator.get("model", trace.source), n_rotations=len(mapped.orch) - 1,
                        source_sha256=source_digest(), trace_sha256=trace_hash,
                        simulated_backend=sim.backend,
                        n_decoy_starts=len(mapped.decoy_starts),
                        cover_rejected=(mapped.cover.n_rejected if mapped.cover else 0),
                        cover_est_energy_max_j=(float(mapped.cover.ledger.total.max()) if mapped.cover else 0.0),
                        wall_s=time.time() - t0)
    rec = RunRecord(frames=dedupe_frames(sim.frames), rx_time=sim.rx_time, energy_j=sim.energy_j,
                    **physical_arrays(mapped.messages), orch=np.array(mapped.orch, float),
                    plan_changes=np.asarray(trace.plan_changes, float),
                    phases=[(p.start, p.end, p.name) for p in trace.phases], meta=meta)
    rec.save(out)
    log.info("%s: %d messages, %d frames, %.1fs", out.name, len(mapped.messages), len(rec.frames), meta["wall_s"])
    return out


def _task(args: tuple[ExperimentConfig, GridEntry, int, bool]) -> str:
    cfg, entry, mission, overwrite = args
    return str(run_one(cfg, entry, mission, overwrite))


def run_grid(cfg: ExperimentConfig, overwrite: bool = False, only: list[str] | None = None) -> list[Path]:
    """Run every (grid entry, mission). ns-3 runs each go to a fresh process (one simulator per process)."""
    expected_runs(cfg)  # reject duplicate tags before starting any task
    if only is not None and (set(only) - {e.defense for e in cfg.grid}):
        raise ValueError(f"unknown or unconfigured defenses in --only: {only}")
    tasks = [(cfg, e, m, overwrite) for e in cfg.grid for m in e.missions
             if only is None or e.defense in only]
    (cfg.output_dir / "runs").mkdir(parents=True, exist_ok=True)
    done: list[Path] = []
    status: list[tuple[str, str]] = []
    if cfg.backend == "toy" and cfg.jobs <= 1:
        for t in tasks:
            try:
                done.append(Path(_task(t)))
                status.append((t[1].tag(t[2]), "ok"))
            except Exception as err:  # noqa: BLE001 - record failures before raising below
                log.error("run %s failed: %s", t[1].tag(t[2]), err)
                status.append((t[1].tag(t[2]), f"failed: {err}"))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, cfg.jobs), mp_context=ctx, max_tasks_per_child=1) as ex:
            futs = {ex.submit(_task, t): t for t in tasks}
            for f in as_completed(futs):
                e, m = futs[f][1], futs[f][2]
                try:
                    done.append(Path(f.result()))
                    status.append((e.tag(m), "ok"))
                except Exception as err:  # noqa: BLE001 - record and continue the sweep
                    log.error("run %s failed: %s", e.tag(m), err)
                    status.append((e.tag(m), f"failed: {err}"))
    with open(cfg.output_dir / "runs" / "manifest.csv", "a", newline="") as fh:
        csv.writer(fh).writerows(status)
    failed = [tag for tag, state in status if state != "ok"]
    if failed:
        raise RuntimeError(f"{len(failed)}/{len(tasks)} runs failed; see runs/manifest.csv. "
                           "Evaluation must wait until every requested run succeeds.")
    return done
