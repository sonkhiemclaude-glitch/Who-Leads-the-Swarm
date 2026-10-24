"""Tables E1-E4 (Section 7) generated from metrics.csv; never hard-coded.

`write_latex_tables` produces the four LaTeX tables used by the manuscript (`manuscript/tables`).
With `pending=True` it writes the red PENDING placeholders instead. The caption always carries the
provenance label read from metrics.csv, so synthetic outputs cannot be mistaken for paper results.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..theory.horizon_bound import prop1_bound, top_k_pmax

NAMES = {"none": "No defense", "pad": "Fixed padding", "rot": "Rotation only", "cover_uniform": "Uniform cover",
         "cover_targeted": "Targeted cover", "full": "Full protocol", "full_anyhop": "Full, any-hop cover",
         "const": "Padding + constant-rate cover"}
MAIN = [("none", 0.0), ("pad", 0.0), ("rot", 0.0), ("cover_uniform", 0.2), ("cover_uniform", 1.0),
        ("cover_targeted", 0.2), ("cover_targeted", 1.0), ("full", 0.05), ("full", 0.2), ("full", 1.0),
        ("const", 0.2), ("const", 1.0), ("full_anyhop", 0.2)]
PENDING = r"\textcolor{red}{pending}"


def _fmt(m: float, h: float, pct: bool = True, nd: int = 1) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "--"
    k = 100.0 if pct else 1.0
    return f"{k * m:.{nd}f}$\\pm${k * h:.{nd}f}" if h == h else f"{k * m:.{nd}f}"


class _Lookup:
    def __init__(self, df: pd.DataFrame | None) -> None:
        self.df = df

    def get(self, defense: str, budget: float, routing: str = "olsr", n: int = 20, speed: float = 15.0,
            mobility: str = "gm", monitors: str = "quad") -> pd.Series | None:
        if self.df is None:
            return None
        d = self.df
        sel = d[(d.defense == defense) & ((d.budget - budget).abs() < 1e-9) & (d.routing == routing) & (d.n == n)
                & ((d.speed - speed).abs() < 1e-9) & (d.mobility == mobility) & (d.monitors == monitors)]
        return None if sel.empty else sel.iloc[0]

    def cell(self, col: str, *key: Any, pct: bool = True, nd: int = 1, scen: dict[str, Any] | None = None) -> str:
        row = self.get(*key, **(scen or {}))
        if row is None or col not in row:
            return PENDING if self.df is None else "--"
        return _fmt(float(row[col]), float(row.get(col + "_ci", float("nan"))), pct, nd)


def _table(path: Path, name: str, caption: str, head: list[str], rows: list[str]) -> None:
    tab = ("\\begin{tabular}{@{}l" + "c" * (len(head) - 1) + "@{}}\\toprule\n" + " & ".join(head) + " \\\\\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
    body = ("\\begin{table}[htbp]\\centering\\small\n\\caption{" + caption + "}\n\\label{tab:" + name + "}\n"
            "\\resizebox{\\textwidth}{!}{" + tab + "}\\end{table}\n")
    (path / f"{name}.tex").write_text(body, encoding="utf-8")


def write_latex_tables(out_dir: str | Path, metrics_csv: str | Path | None = None, pending: bool = False) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = None if pending else pd.read_csv(metrics_csv)  # type: ignore[arg-type]
    L = _Lookup(df)
    if pending:
        note = r"\textcolor{red}{PENDING: generated from real LLM-agent traces once available.}"
    else:
        labels = sorted(set(df["provenance_label"])) if df is not None and "provenance_label" in df else ["unknown"]
        note = "Source: " + "; ".join(labels).replace("_", r"\_") + "."
    rows = []
    for lab, key in [("OLSR, GM, 4 monitors", ("none", 0.0)), ("AODV", ("none", 0.0, "aodv")),
                     ("Group mobility (RPGM)", ("none", 0.0)), ("One edge monitor", ("none", 0.0))]:
        kw: dict[str, Any] = {"mobility": "rpgm"} if "RPGM" in lab else ({"monitors": "edge"} if "edge" in lab else {})
        rows.append(f"{lab} & {L.cell('role_acc_h0', *key, scen=kw)} & {L.cell('role_acc_heur', *key, scen=kw)} & "
                    f"{L.cell('role_chance', *key, scen=kw)} & {L.cell('phase_acc', *key, scen=kw)} & "
                    f"{L.cell('phase_majority', *key, scen=kw)} & {L.cell('plan_f1', *key, scen=kw)} \\\\")
    _table(out, "e1", "E1: leakage without defense (\\%, mean $\\pm$ 95\\% CI over test missions). "
           "Heuristic: node with most transmitted bytes. " + note,
           ["Setting", "Role (RF)", "Role (heuristic)", "Chance", "Phase", "Majority", "Plan F1"], rows)
    rows = []
    for lab, n, v in [("$N=10$", 10, 15.0), ("$N=30$", 30, 15.0), ("5 m/s", 20, 5.0), ("25 m/s", 20, 25.0)]:
        rows.append(f"{lab} & {L.cell('role_acc_h0', 'none', 0.0, 'olsr', n, v)} & {L.cell('phase_acc', 'none', 0.0, 'olsr', n, v)} & "
                    f"{L.cell('plan_f1', 'none', 0.0, 'olsr', n, v)} & {L.cell('role_acc_h30', 'full', 0.2, 'olsr', n, v)} & "
                    f"{L.cell('phase_acc', 'full', 0.2, 'olsr', n, v)} & {L.cell('plan_f1', 'full', 0.2, 'olsr', n, v)} \\\\")
    _table(out, "e2", "E2: swarm size and speed, OLSR, 6 missions per setting (\\%). Full protocol at $B=0.2$~J; "
           "its role column is at $\\Delta=30$~s. " + note,
           ["Setting", "Role", "Phase", "Plan F1", "Role (full)", "Phase (full)", "Plan F1 (full)"], rows)
    rows = []
    for d, b in MAIN:
        nm = NAMES[d] + (f", $B$={b:g}" if b else "")
        rows.append(f"{nm} & {L.cell('role_acc_h0', d, b)} & {L.cell('role_acc_h10', d, b)} & {L.cell('role_acc_h30', d, b)} & "
                    f"{L.cell('phase_acc', d, b)} & {L.cell('plan_f1', d, b)} & {L.cell('dE_j_mean', d, b, pct=False, nd=2)} \\\\")
    for lab, d, b, kw in [("No defense, RPGM", "none", 0.0, {"mobility": "rpgm"}),
                          ("Full $B$=0.2, RPGM", "full", 0.2, {"mobility": "rpgm"}),
                          ("No defense, edge monitor", "none", 0.0, {"monitors": "edge"}),
                          ("Full $B$=0.2, edge monitor", "full", 0.2, {"monitors": "edge"})]:
        rows.append(f"{lab} & {L.cell('role_acc_h0', d, b, scen=kw)} & {L.cell('role_acc_h10', d, b, scen=kw)} & "
                    f"{L.cell('role_acc_h30', d, b, scen=kw)} & {L.cell('phase_acc', d, b, scen=kw)} & "
                    f"{L.cell('plan_f1', d, b, scen=kw)} & {L.cell('dE_j_mean', d, b, pct=False, nd=2, scen=kw)} \\\\")
    _table(out, "e3", "E3: attacker accuracy against the adaptive adversary (\\%) and measured incremental radio energy "
           "per node (J), OLSR, $N=20$. " + note,
           ["Configuration", "Role $\\Delta$=0", "Role $\\Delta$=10 s", "Role $\\Delta$=30 s", "Phase", "Plan F1",
            "$\\Delta E$ (J)"], rows)
    rows = []
    for d, b, rt in [(d, b, "olsr") for d, b in MAIN] + [("none", 0.0, "aodv"), ("full", 0.2, "aodv"), ("full_anyhop", 0.2, "aodv")]:
        nm = NAMES[d] + (f", $B$={b:g}" if b else "") + f" ({rt.upper()})"
        rows.append(f"{nm} & {L.cell('pdr', d, b, rt)} & {L.cell('lat_ms_median', d, b, rt, pct=False)} & "
                    f"{L.cell('lat_ms_p95', d, b, rt, pct=False)} & {L.cell('bcast_fps', d, b, rt, pct=False)} & "
                    f"{L.cell('cover_airtime_pct', d, b, rt, pct=False, nd=2)} \\\\")
    _table(out, "e4", "E4: cost to mission traffic. PDR (\\%), latency (ms), broadcast frames observed by the monitors "
           "per second, and airtime share of cover traffic per node (\\%). " + note,
           ["Configuration", "PDR", "Latency median", "Latency p95", "Broadcast fps", "Cover airtime"], rows)
    return [out / f"e{i}.tex" for i in range(1, 5)]


def write_markdown_summary(metrics_csv: str | Path, out_path: str | Path) -> Path:
    """Human-readable summary with a Proposition 1 diagnostic for rotating configurations."""
    df = pd.read_csv(metrics_csv)
    label = "; ".join(sorted(set(df.get("provenance_label", pd.Series(["unknown"])))))
    cols = [c for c in ["role_acc_h0", "role_acc_h10", "role_acc_h30", "role_acc_heur", "phase_acc", "phase_majority",
                        "plan_f1", "pdr", "lat_ms_median", "dE_j_mean", "cover_airtime_pct"] if c in df]
    lines = [f"# Results summary\n\n**Provenance: {label}**\n", "| config | missions | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 2) + "|"]
    for _, r in df.iterrows():
        cells = [f"{r[c]:.3f} ± {r.get(c + '_ci', float('nan')):.3f}" for c in cols]
        lines.append(f"| {r['config_key']} | {int(r['missions'])} | " + " | ".join(cells) + " |")
    lines += ["", "## Proposition 1 diagnostic (rotating configurations)", "",
              "alpha0 is approximated by role_acc_h0 of the same configuration; p_E is measured on the evaluation "
              "windows; p_max = 1/3 (top-3 successor). This is a diagnostic, not a test of the proposition: the paper's "
              "alpha0 is the accuracy of the horizon-specific estimator for the current orchestrator.", "",
              "| config | horizon | measured role acc | bound | p_E | Pr[>=2 rotations] |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        for h in (10, 30):
            pe = r.get(f"p_E_h{h}")
            if pe is None or pe != pe or pe == 0:
                continue
            b = prop1_bound(min(max(r["role_acc_h0"], 0), 1), pe, top_k_pmax(3, int(r["n"]) - 1))
            lines.append(f"| {r['config_key']} | {h} s | {r[f'role_acc_h{h}']:.3f} | {b:.3f} | {pe:.3f} | "
                         f"{r.get(f'p_two_plus_h{h}', float('nan')):.3f} |")
    p = Path(out_path)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
