#!/usr/bin/env python3
"""Write tables E1-E4 into the manuscript's `tables/` folder.

    # placeholders (current manuscript state)
    python scripts/write_manuscript_tables.py --pending --out ../../manuscript/tables
    # from real ns-3 + LLM-trace results (refuses synthetic results)
    python scripts/write_manuscript_tables.py --metrics outputs/paper_grid/metrics.csv --out ../../manuscript/tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarmleak.experiments.tables import write_latex_tables

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pending", action="store_true")
    a = ap.parse_args()
    if a.pending:
        write_latex_tables(a.out, pending=True)
    else:
        if not a.metrics:
            raise SystemExit("--metrics is required")
        prov = json.loads((Path(a.metrics).parent / "provenance.json").read_text())
        if prov.get("synthetic", True):
            raise SystemExit("refusing to write synthetic results into the manuscript (provenance.json: synthetic=true)")
        write_latex_tables(a.out, a.metrics)
    print(f"wrote tables to {a.out}")
