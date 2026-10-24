#!/usr/bin/env python3
"""Run one stage of an experiment.

    python scripts/run_experiment.py --config configs/experiment_template.yaml --stage gen-traces
    python scripts/run_experiment.py --config configs/experiment_template.yaml --stage simulate --jobs 8
    python scripts/run_experiment.py --config configs/experiment_template.yaml --stage evaluate
    python scripts/run_experiment.py --config configs/experiment_template.yaml --stage tables
    python scripts/run_experiment.py --config configs/experiment_template.yaml --stage plot

Stages map to the paper: gen-traces (Sec. 4.1), simulate (Sec. 4.2-4.3, 5), evaluate (Sec. 4.4, 3.3-3.4),
tables/plot (Sec. 7). Nothing is downloaded. Paper-level runs need ns-3 and real traces (see
docs/mac_studio_experiment_guide.md).
"""
from __future__ import annotations

import argparse
import sys

from swarmleak.cli import main
from swarmleak.experiments.config import ExperimentConfig

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["gen-traces", "simulate", "evaluate", "tables", "plot"])
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--mock", action="store_true", help="gen-traces with the offline mock client (synthetic)")
    a = ap.parse_args()
    cfg = ExperimentConfig.load(a.config)
    metrics = str(cfg.output_dir / "metrics.csv")
    argv = {
        "gen-traces": ["gen-traces", "--config", a.config] + (["--mock"] if a.mock else []),
        "simulate": ["simulate", "--config", a.config] + (["--jobs", str(a.jobs)] if a.jobs else []),
        "evaluate": ["evaluate", "--config", a.config],
        "tables": ["tables", "--metrics", metrics, "--out", str(cfg.output_dir / "tables")],
        "plot": ["plot", "--metrics", metrics, "--out", str(cfg.output_dir / "tradeoff.png")],
    }[a.stage]
    sys.exit(main(argv))
