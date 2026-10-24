"""Evaluation stage: runs/*.npz -> per_mission.csv, metrics.csv, comparisons.csv, provenance.json.

All numbers are computed from the saved frame logs; nothing is hard-coded.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..attack.evaluate import evaluate_config
from ..data.schemas import RunRecord
from ..metrics.confidence_intervals import paired_bootstrap, t_interval
from ..metrics.network import incremental_energy, network_metrics
from ..utils.io import dump_json, stable_hash
from .config import ExperimentConfig
from .integrity import expected_runs, validate_run, validate_run_files
from .provenance import environment, provenance_label, source_digest
from .runner import load_trace

log = logging.getLogger(__name__)

KEY_COLS = ["defense", "budget", "routing", "n", "speed", "mobility", "monitors"]


def _scenario_key(m: dict[str, Any]) -> str:
    return f"{m['routing']}_n{m['n']}_v{float(m['speed']):g}_{m['mobility']}_{m['monitors']}"


def evaluate_runs(cfg: ExperimentConfig) -> dict[str, Path]:
    files = validate_run_files(cfg)
    expected = expected_runs(cfg)
    runs = []
    trace_hashes: dict[tuple[int, float, int], str] = {}
    for path in files:
        entry, mission = expected[path.stem]
        trace_key = (entry.scenario.n, entry.scenario.duration, mission)
        if trace_key not in trace_hashes:
            trace_hashes[trace_key] = stable_hash(load_trace(cfg, entry.scenario, mission).to_dict())
        rec = RunRecord.load(path)
        validate_run(rec, cfg, entry, mission, trace_hashes[trace_key])
        runs.append(rec)
    synth = {bool(r.meta.get("synthetic", True)) for r in runs}
    synthetic = cfg.synthetic or True in synth
    groups: dict[str, list[RunRecord]] = defaultdict(list)
    for r in runs:
        groups[r.meta["config_key"]].append(r)
    undefended = {(_scenario_key(r.meta), int(r.meta["mission"])): r for r in runs if r.meta["defense"] == "none"}
    rows: list[dict[str, Any]] = []
    for key, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: int(r.meta["mission"]))
        if len(rs) < 2:
            log.warning("config %s has %d mission(s); at least 2 are needed for cross-validation, skipped", key, len(rs))
            continue
        leak = evaluate_config(rs, cfg.attack, cfg.seed)
        by_mission = {int(x["mission"]): x for x in leak}
        for r in rs:
            m = int(r.meta["mission"])
            row: dict[str, Any] = {"config_key": key, **{c: r.meta[c] for c in KEY_COLS}, "mission": m}
            row.update(by_mission.get(m, {}))
            row.update(network_metrics(r))
            base = undefended.get((_scenario_key(r.meta), m))
            row["dE_j_mean"] = 0.0 if r.meta["defense"] == "none" else (
                incremental_energy(r, base) if base is not None else float("nan"))
            row["provenance_label"] = r.meta.get("provenance_label", "")
            rows.append(row)
    if not rows:
        raise ValueError("no configuration had enough missions to evaluate")
    per = pd.DataFrame(rows)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"per_mission": out_dir / "per_mission.csv", "metrics": out_dir / "metrics.csv",
             "comparisons": out_dir / "comparisons.csv", "provenance": out_dir / "provenance.json"}
    per.to_csv(paths["per_mission"], index=False)

    metric_cols = [c for c in per.columns if c not in KEY_COLS + ["config_key", "mission", "fold", "provenance_label"]
                   and pd.api.types.is_numeric_dtype(per[c])]
    agg = []
    for ckey, g in per.groupby("config_key", sort=True):
        arow: dict[str, Any] = {"config_key": ckey, **{c: g.iloc[0][c] for c in KEY_COLS}, "missions": int(len(g))}
        for c in metric_cols:
            mean, half = t_interval(g[c].tolist())
            arow[c] = mean
            arow[c + "_ci"] = half
        agg.append(arow)
    label = provenance_label(synthetic, cfg.backend, cfg.trace_source)
    if synthetic and not cfg.synthetic:
        log.warning("config is non-synthetic but some runs used synthetic traces; all outputs are labelled synthetic")
    met = pd.DataFrame(agg)
    met["provenance_label"] = label
    met.to_csv(paths["metrics"], index=False)

    comp = []
    for spec in cfg.comparisons:
        a, b, metric = spec["a"], spec["b"], spec["metric"]
        ga = per[per.config_key == a].set_index("mission")
        gb = per[per.config_key == b].set_index("mission")
        common = sorted(set(ga.index) & set(gb.index))
        if not common or metric not in per.columns:
            log.warning("comparison %s vs %s on %s: no paired missions or unknown metric", a, b, metric)
            continue
        res = paired_bootstrap(ga.loc[common, metric].tolist(), gb.loc[common, metric].tolist(), cfg.bootstrap,
                               seed=cfg.seed)
        comp.append({"a": a, "b": b, "metric": metric, **res, "provenance_label": label})
    pd.DataFrame(comp, columns=["a", "b", "metric", "mean_diff", "lo", "hi", "n", "provenance_label"]).to_csv(
        paths["comparisons"], index=False)
    dump_json({"experiment": cfg.name, "config_hash": cfg.config_hash, "synthetic": synthetic,
               "provenance_label": label, "backend": cfg.backend, "trace_source": cfg.trace_source,
               "n_runs": len(runs), "run_code_versions": sorted({str(r.meta.get("code_version")) for r in runs}),
               "source_sha256": source_digest(), "config": cfg.raw,
               "environment": environment()}, paths["provenance"])
    log.info("wrote %s", ", ".join(str(p) for p in paths.values()))
    return paths
