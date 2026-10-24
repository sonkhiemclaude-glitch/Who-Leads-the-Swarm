#!/usr/bin/env python3
"""Trade-off figure and LaTeX tables from an evaluated output directory.

    python scripts/make_plots.py --input outputs/synthetic_demo --output outputs/synthetic_demo/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

from swarmleak.experiments.plots import tradeoff_plot
from swarmleak.experiments.tables import write_latex_tables, write_markdown_summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="directory containing metrics.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=None, help="swarm size of the trade-off scenario (default: most frequent)")
    a = ap.parse_args()
    metrics = Path(a.input) / "metrics.csv"
    if not metrics.exists():
        raise SystemExit(f"{metrics} not found; run the evaluate stage first")
    out = Path(a.output)
    import pandas as pd

    n = a.n if a.n is not None else int(pd.read_csv(metrics)["n"].mode()[0])
    tradeoff_plot(metrics, out / "tradeoff.png", n=n)
    write_latex_tables(out / "tables", metrics)
    write_markdown_summary(metrics, out / "summary.md")
    print(f"wrote figures and tables to {out}")
