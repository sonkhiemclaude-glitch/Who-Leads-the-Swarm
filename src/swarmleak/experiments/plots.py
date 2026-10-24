"""Plots generated from metrics.csv (never from hard-coded numbers). The provenance label is
written into the figure title, so synthetic smoke-test figures are always marked as such."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# categorical palette validated for CVD separation (see docs/implementation_notes.md)
SERIES = {"cover_uniform": ("#2a78d6", "o", "Uniform cover"), "cover_targeted": ("#eb6834", "s", "Targeted cover"),
          "full": ("#1baf7a", "^", "Full protocol")}
SINGLE = [("none", "No defense"), ("pad", "Padding"), ("rot", "Rotation"), ("full_anyhop", "Full, any-hop"),
          ("const", "Const-rate")]
GREY = "#6b6b66"
LABEL_OFFSETS = {"none": (-4, 9), "rot": (6, -14), "pad": (8, -12), "full_anyhop": (6, 4), "const": (-40, 9)}


def tradeoff_plot(metrics_csv: str | Path, out_png: str | Path, routing: str = "olsr", n: int | None = None,
                  speed: float = 15.0) -> Path:
    """Energy trade-off for one scenario; n defaults to the most frequent swarm size in metrics.csv."""
    df = pd.read_csv(metrics_csv)
    if n is None:
        n = int(df["n"].mode()[0])
    label = "; ".join(sorted(set(df.get("provenance_label", pd.Series(["unknown"])))))
    d = df[(df.routing == routing) & (df.n == n) & ((df.speed - speed).abs() < 1e-9) & (df.mobility == "gm")
           & (df.monitors == "quad")]
    if d.empty:
        raise ValueError("no rows for the requested scenario in metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    for ax, key, lab in [(axes[0], "role_acc_h30", "Role accuracy at Δ = 30 s"), (axes[1], "plan_f1", "Plan-change F1")]:
        if key not in d:
            ax.set_visible(False)
            continue
        for defense, (c, m, name) in SERIES.items():
            g = d[d.defense == defense].sort_values("dE_j_mean")
            if len(g):
                ax.errorbar(g.dE_j_mean, g[key], yerr=g.get(key + "_ci"), color=c, marker=m, markersize=8,
                            linewidth=2, capsize=3, label=name)
        for defense, name in SINGLE:
            for _, r in d[d.defense == defense].iterrows():
                ax.errorbar([r.dE_j_mean], [r[key]], yerr=[r.get(key + "_ci", 0)], color=GREY, marker="D",
                            markersize=7, capsize=3, linestyle="none")
                txt = f"{name} B={r.budget:g}" if defense == "const" else name
                ax.annotate(txt, (r.dE_j_mean, r[key]), textcoords="offset points", xytext=LABEL_OFFSETS[defense], fontsize=8,
                            color="#3d3d3a")
        ax.set_xlabel("Measured incremental radio energy per node (J)")
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#e5e4df", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="outside lower center", ncol=3)
    fig.suptitle(label, fontsize=9)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
