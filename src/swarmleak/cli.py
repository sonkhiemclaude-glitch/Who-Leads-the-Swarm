"""Command-line interface: `python -m swarmleak.cli --help` or `swarmleak --help`.

Stages: gen-traces (real LLM traces; needs a served model) -> simulate -> evaluate -> tables / plot.
`demo` runs simulate + evaluate + tables + plot on a synthetic config.
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import SYNTHETIC_LABEL, __version__

log = logging.getLogger("swarmleak")


def _cfg(path: str):
    from .experiments.config import ExperimentConfig

    return ExperimentConfig.load(path)


def cmd_info(_: argparse.Namespace) -> int:
    from .sim.ns3_backend import ns3_available

    print(f"swarmleak {__version__}")
    print("Paper: Who Leads the Swarm? Traffic-Pattern Leakage of LLM-Agent Coordination in UAV Ad Hoc Networks")
    print(f"ns-3 Python bindings available: {ns3_available()}")
    return 0


def cmd_preflight(a: argparse.Namespace) -> int:
    import json

    from .experiments.preflight import preflight

    report = preflight(_cfg(a.config), a.stage)
    print(json.dumps(report, indent=2))
    return 0 if report["technical_checks_passed"] else 1


def cmd_gen_traces(a: argparse.Namespace) -> int:
    from .llm.agents import MockChatClient, OpenAICompatibleClient, generate_llm_trace
    from .utils.random import int_seed, rng_for

    cfg = _cfg(a.config)
    if cfg.trace_root is None:
        raise SystemExit("data.root must be set to the directory where traces are written")
    llm = cfg.llm
    if a.mock:
        log.warning("--mock: traces come from a deterministic mock client and are SYNTHETIC")
    elif not llm.get("endpoint") or not llm.get("model"):
        raise SystemExit("llm.endpoint and llm.model must be set (e.g. http://localhost:11434/v1, a local model tag)")
    from .data.schemas import Trace
    from .experiments.provenance import source_digest, trace_generation_digest
    from .utils.io import stable_hash

    required = {(e.scenario.n, m): e.scenario.duration for e in cfg.grid for m in e.missions}
    pairs = sorted(required)
    if a.n is not None:
        pairs = [pair for pair in pairs if pair[0] == a.n]
    if a.missions is not None:
        pairs = [pair for pair in pairs if pair[1] in a.missions]
    if not pairs:
        raise SystemExit("no requested (n, mission) pairs exist in the configured grid")
    generation = {"llm": llm, "seed": cfg.seed, "mock": bool(a.mock), "generator_sha256": trace_generation_digest()}
    fingerprint = stable_hash(generation)
    for n, m in pairs:
        path = cfg.trace_root / cfg.trace_pattern.format(n=n, mission=m)
        if path.exists() and not a.overwrite:
            existing = Trace.load_json(path)
            if existing.generator.get("generation_hash") != fingerprint:
                raise SystemExit(f"trace collection differs: {path}; use a separate trace directory")
            log.info("exists (validated): %s", path)
            continue
        audit_path = str(path.with_suffix(".calls.jsonl"))
        if not a.mock:
            from pathlib import Path

            Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
            # Preserve any failed attempt's log rather than append a different collection.
            if Path(audit_path).exists():
                import time

                Path(audit_path).rename(audit_path + f".failed-{time.time_ns()}")
        client = (MockChatClient(seed=int_seed(cfg.seed, "mock-llm", n, m)) if a.mock else
                  OpenAICompatibleClient(str(llm["endpoint"]), str(llm["model"]), float(llm.get("temperature", 0.7)),
                                         int(llm.get("max_tokens", 400)), float(llm.get("timeout_s", 120)), audit_path,
                                         structured_outputs=bool(llm.get("structured_outputs", False))))
        tr = generate_llm_trace(client, n, required[(n, m)], rng_for(cfg.seed, "trace", n, m),
                                float(llm.get("latency_scale", 0.2)), float(llm.get("report_llm_prob", 1.0)))
        tr.generator.update({"mock": bool(a.mock), "mission": m, "generation_hash": fingerprint,
                             "generation": generation, "source_sha256": source_digest()})
        tr.save_json(path)
        log.info("wrote %s (%d messages)", path, len(tr.messages))
    return 0


def cmd_simulate(a: argparse.Namespace) -> int:
    from .experiments.runner import run_grid

    cfg = _cfg(a.config)
    if a.jobs is not None:
        cfg.jobs = a.jobs
    if cfg.synthetic:
        log.warning("this configuration is synthetic: outputs are '%s'", SYNTHETIC_LABEL)
    run_grid(cfg, overwrite=a.overwrite, only=a.only)
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    from .experiments.evaluation import evaluate_runs
    from .experiments.tables import write_markdown_summary

    cfg = _cfg(a.config)
    paths = evaluate_runs(cfg)
    write_markdown_summary(paths["metrics"], cfg.output_dir / "summary.md")
    return 0


def cmd_tables(a: argparse.Namespace) -> int:
    from .experiments.tables import write_latex_tables

    if not a.pending and not a.metrics:
        raise SystemExit("--metrics is required unless --pending")
    write_latex_tables(a.out, a.metrics, pending=a.pending)
    return 0


def cmd_plot(a: argparse.Namespace) -> int:
    from .experiments.plots import tradeoff_plot

    tradeoff_plot(a.metrics, a.out)
    return 0


def cmd_demo(a: argparse.Namespace) -> int:
    from .experiments.evaluation import evaluate_runs
    from .experiments.plots import tradeoff_plot
    from .experiments.runner import run_grid
    from .experiments.tables import write_latex_tables, write_markdown_summary

    cfg = _cfg(a.config)
    if not cfg.synthetic:
        raise SystemExit("demo is for synthetic configs only (backend toy or synthetic traces)")
    run_grid(cfg, overwrite=a.overwrite)
    paths = evaluate_runs(cfg)
    write_markdown_summary(paths["metrics"], cfg.output_dir / "summary.md")
    write_latex_tables(cfg.output_dir / "tables", paths["metrics"])
    tradeoff_plot(paths["metrics"], cfg.output_dir / "tradeoff.png", n=cfg.base_scenario.n, speed=cfg.base_scenario.speed)
    (cfg.output_dir / "README.txt").write_text(
        f"{SYNTHETIC_LABEL.upper()}\nGenerated by `swarmleak demo --config {a.config}`.\n"
        "These files test the pipeline. They are not results of the paper.\n", encoding="utf-8")
    print(f"demo outputs in {cfg.output_dir} ({SYNTHETIC_LABEL})")
    return 0


def cmd_bound(a: argparse.Namespace) -> int:
    from .theory.horizon_bound import prop1_bound, prop1_exact, top_k_pmax

    pmax = a.pmax if a.pmax is not None else top_k_pmax(3, a.n - 1)
    print(f"Proposition 1 bound: {prop1_bound(a.alpha0, a.pe, pmax):.4f} (p_max={pmax:.4f})")
    print(f"Exact value under independence and a uniform successor: {prop1_exact(a.alpha0, a.pe, a.n):.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swarmleak", description=__doc__)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="version and backend availability").set_defaults(fn=cmd_info)
    pf = sub.add_parser("preflight", help="read-only environment/data/cache checks (no LLM requests)")
    pf.add_argument("--config", required=True)
    pf.add_argument("--stage", choices=["gen-traces", "simulate", "evaluate"], default="simulate")
    pf.set_defaults(fn=cmd_preflight)
    g = sub.add_parser("gen-traces", help="generate LLM-agent traces into data.root (needs llm.endpoint)")
    g.add_argument("--config", required=True)
    g.add_argument("--n", type=int)
    g.add_argument("--missions", type=int, nargs="+")
    g.add_argument("--mock", action="store_true", help="use the offline mock client (synthetic traces)")
    g.add_argument("--overwrite", action="store_true")
    g.set_defaults(fn=cmd_gen_traces)
    s = sub.add_parser("simulate", help="run the simulation grid -> output_dir/runs/*.npz")
    s.add_argument("--config", required=True)
    s.add_argument("--jobs", type=int)
    s.add_argument("--only", nargs="+", help="only these defenses")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=cmd_simulate)
    e = sub.add_parser("evaluate", help="attack + metrics -> metrics.csv, per_mission.csv, comparisons.csv")
    e.add_argument("--config", required=True)
    e.set_defaults(fn=cmd_evaluate)
    t = sub.add_parser("tables", help="LaTeX tables E1-E4 from metrics.csv")
    t.add_argument("--metrics")
    t.add_argument("--out", required=True)
    t.add_argument("--pending", action="store_true")
    t.set_defaults(fn=cmd_tables)
    pl = sub.add_parser("plot", help="trade-off figure from metrics.csv")
    pl.add_argument("--metrics", required=True)
    pl.add_argument("--out", required=True)
    pl.set_defaults(fn=cmd_plot)
    d = sub.add_parser("demo", help="synthetic smoke test: simulate + evaluate + tables + plot")
    d.add_argument("--config", required=True)
    d.add_argument("--overwrite", action="store_true")
    d.set_defaults(fn=cmd_demo)
    b = sub.add_parser("bound", help="evaluate Proposition 1")
    b.add_argument("--alpha0", type=float, required=True)
    b.add_argument("--pe", type=float, required=True)
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--pmax", type=float)
    b.set_defaults(fn=cmd_bound)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .utils.logging import setup_logging

    setup_logging(args.log_level)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())

