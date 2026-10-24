"""Read-only checks of the environment, required traces and cached-run integrity."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..data.schemas import RunRecord
from ..sim.ns3_backend import ns3_available
from ..utils.io import stable_hash
from .config import ExperimentConfig
from .integrity import expected_runs, validate_run, validate_run_files
from .provenance import environment, source_digest
from .runner import load_trace, trace_is_synthetic


def preflight(cfg: ExperimentConfig, stage: str = "simulate") -> dict[str, Any]:
    if stage not in ("gen-traces", "simulate", "evaluate"):
        raise ValueError(f"unknown preflight stage: {stage}")
    issues: list[str] = []
    warnings = ["Technical checks only. Research correctness must also pass the source review "
                "in audits/experiment_readiness_review.md before paper experiments."]
    expected = expected_runs(cfg)
    if cfg.jobs < 1:
        issues.append("experiment.jobs must be >= 1")
    if stage != "gen-traces" and cfg.backend == "ns3" and not ns3_available():
        issues.append("ns-3 Python bindings not detected (ns and cppyy). An actual ns-3 replay test is also required.")
    if stage == "gen-traces":
        if cfg.trace_root is None:
            issues.append("data.root must be set for trace generation")
        for setting in ("endpoint", "model"):
            if not cfg.llm.get(setting):
                issues.append(f"llm.{setting} must be set for real trace generation")
        warnings.append("LLM endpoint was not contacted; model availability is not checked.")
    required = sorted({(e.scenario.n, e.scenario.duration, m) for e, m in expected.values()})
    # One filename cannot safely represent multiple mission durations.
    identities: dict[tuple[int, int], set[float]] = {}
    for n, duration, m in required:
        identities.setdefault((n, m), set()).add(duration)
    if any(len(ds) > 1 for ds in identities.values()):
        issues.append("grid reuses (n, mission) trace filenames with different durations")
    hashes: dict[tuple[int, float, int], str] = {}
    if stage != "gen-traces" and cfg.trace_source == "llm":
        for entry, mission in expected.values():
            key = (entry.scenario.n, entry.scenario.duration, mission)
            if key in hashes:
                continue
            try:
                trace = load_trace(cfg, entry.scenario, mission)
                if trace_is_synthetic(trace):
                    raise ValueError("mock/surrogate trace cannot be used as a real LLM trace")
                if not trace.source.startswith("llm:") or not trace.generator.get("model"):
                    raise ValueError("real LLM trace needs source='llm:<model>' and generator.model")
                if not trace.messages or any(m.t >= trace.duration for m in trace.messages):
                    raise ValueError("trace must contain messages within the mission duration")
                if any(not (0 <= t < trace.duration) for t in trace.plan_changes):
                    raise ValueError("plan-change ground truth is outside mission duration")
                if abs(trace.phases[0].start) > 1e-6 or abs(trace.phases[-1].end - trace.duration) > 1e-6:
                    raise ValueError("phases must cover the complete mission")
                hashes[key] = stable_hash(trace.to_dict())
            except (OSError, ValueError, KeyError, TypeError) as err:
                hashes[key] = "invalid"
                issues.append(f"trace n={key[0]}, duration={key[1]:g}, mission={key[2]}: {err}")
    if stage == "evaluate":
        try:
            files = validate_run_files(cfg)
            for path in files:
                entry, mission = expected[path.stem]
                key = (entry.scenario.n, entry.scenario.duration, mission)
                if key not in hashes:
                    hashes[key] = stable_hash(load_trace(cfg, entry.scenario, mission).to_dict())
                validate_run(RunRecord.load(path), cfg, entry, mission, hashes[key])
        except (OSError, ValueError, KeyError, TypeError) as err:
            issues.append(str(err))
    elif stage == "simulate":
        found = {p.stem: p for p in (cfg.output_dir / "runs").glob("*.npz")}
        if set(found) - set(expected):
            issues.append("output directory has runs outside the requested grid; use a separate directory")
        for tag in sorted(set(found) & set(expected)):
            entry, mission = expected[tag]
            try:
                key = (entry.scenario.n, entry.scenario.duration, mission)
                trace_hash = hashes.get(key) or stable_hash(load_trace(cfg, entry.scenario, mission).to_dict())
                validate_run(RunRecord.load(found[tag]), cfg, entry, mission, trace_hash)
            except (OSError, ValueError, KeyError, TypeError) as err:
                issues.append(str(err))
    if cfg.synthetic:
        warnings.append("Synthetic configuration: results are pipeline checks, not paper results.")
    for entry in cfg.grid:
        if len(entry.missions) < 2:
            issues.append(f"{entry.config_key()}: evaluation requires at least two missions")
    root = cfg.output_dir.resolve()
    while not root.exists():
        root = root.parent
    return {"stage": stage, "technical_checks_passed": not issues,
            "scope": "Environment, input data and cached-run integrity; not scientific validation",
            "experiment": cfg.name, "cwd": str(Path.cwd()), "output_dir": str(cfg.output_dir.resolve()),
            "config_hash": cfg.config_hash, "source_sha256": source_digest(),
            "planned_runs": len(expected), "planned_configurations": len(cfg.grid),
            "required_trace_files": len(identities), "trace_requirements": [list(k) for k in required],
            "jobs": cfg.jobs, "free_disk_gib": round(shutil.disk_usage(root).free / 2**30, 2),
            "environment": environment(), "issues": list(dict.fromkeys(issues)),
            "warnings": list(dict.fromkeys(warnings))}
