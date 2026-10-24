"""Mission-grouped cross-validated evaluation of the adaptive adversary (Section 4.4).

For one configuration (defense, budget, scenario), runs of different missions are split into
folds; the adversary is trained on the other missions' runs of the SAME configuration (adaptive
adversary) and tested on the held-out missions. Returns one metric row per test mission.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..constants import (
    CV_FOLDS,
    HORIZONS_S,
    PHASE_WINDOW,
    PLAN_MERGE_S,
    PLAN_THRESHOLD,
    PLAN_TOLERANCE_S,
    PLAN_WINDOW,
    RF_MIN_LEAF,
    RF_TREES,
    ROLE_WINDOW,
)
from ..data.schemas import RunRecord
from ..metrics.leakage import detect_events, majority_rate, phase_accuracy, plan_change_f1, role_accuracy_at_horizon
from ..metrics.rotation_stats import rotation_stats
from ..utils.random import int_seed, rng_for
from .classifiers import argmax_random_ties, heuristic_role_estimate, make_rf, positive_proba
from .datasets import orch_at, phase_dataset, plan_dataset, role_dataset

log = logging.getLogger(__name__)


@dataclass
class AttackConfig:
    role_window: tuple[float, float] = ROLE_WINDOW
    phase_window: tuple[float, float] = PHASE_WINDOW
    plan_window: tuple[float, float] = PLAN_WINDOW
    horizons: tuple[float, ...] = HORIZONS_S
    plan_tolerance_s: float = PLAN_TOLERANCE_S
    plan_threshold: float = PLAN_THRESHOLD
    plan_merge_s: float = PLAN_MERGE_S
    rf_trees: int = RF_TREES
    rf_min_leaf: int = RF_MIN_LEAF
    folds: int = CV_FOLDS
    n_jobs: int = 1
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> AttackConfig:
        d = dict(d or {})
        for k in ("role_window", "phase_window", "plan_window", "horizons"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def mission_folds(n_runs: int, n_folds: int) -> list[list[int]]:
    """Contiguous mission-disjoint folds; honor the requested count up to n_runs."""
    if n_runs < 2:
        raise ValueError("cross-validation needs at least 2 missions")
    if n_folds < 2:
        raise ValueError("cross-validation needs at least 2 folds")
    k = min(n_folds, n_runs)
    return [list(map(int, f)) for f in np.array_split(np.arange(n_runs), k)]


def evaluate_config(runs: list[RunRecord], cfg: AttackConfig, seed: int = 0) -> list[dict[str, float]]:
    """Cross-validated leakage metrics, one dict per test mission."""
    if len({r.n for r in runs}) != 1:
        raise ValueError("all runs of a configuration must have the same swarm size")
    W, step = cfg.role_window
    role = [role_dataset(r, W, step, cfg.horizons) for r in runs]
    phase = [phase_dataset(r, *cfg.phase_window) for r in runs]
    plan = [plan_dataset(r, *cfg.plan_window) for r in runs]
    out: list[dict[str, float]] = []
    for fold_id, test in enumerate(mission_folds(len(runs), cfg.folds)):
        train = [i for i in range(len(runs)) if i not in test]
        s = int_seed(seed, "rf", fold_id)
        Xr = np.vstack([role[i].X for i in train])
        role_models = {h: make_rf(s, cfg.rf_trees, cfg.rf_min_leaf, cfg.n_jobs).fit(
            Xr, np.concatenate([role[i].y[h] for i in train])) for h in cfg.horizons}
        Xp = np.vstack([phase[i][0] for i in train])
        yp = np.concatenate([phase[i][1] for i in train])
        phase_model = make_rf(s, cfg.rf_trees, cfg.rf_min_leaf, cfg.n_jobs).fit(Xp, yp)
        Xc = np.vstack([plan[i][0] for i in train])
        yc = np.concatenate([plan[i][1] for i in train])
        plan_model = make_rf(s, cfg.rf_trees, cfg.rf_min_leaf, cfg.n_jobs).fit(Xc, yc)
        for i in test:
            r, rd = runs[i], role[i]
            tie_rng = rng_for(seed, "ties", int(r.meta.get("mission", i)))
            row: dict[str, float] = {"mission": float(r.meta.get("mission", i)), "fold": float(fold_id)}
            ends = rd.window_start + W
            for h in cfg.horizons:
                p = positive_proba(role_models[h], rd.X)
                est = np.array([argmax_random_ties(p[rd.window_id == k], tie_rng) for k in range(len(rd.window_start))])
                row[f"role_acc_h{h:g}"] = role_accuracy_at_horizon(est, orch_at(r.orch, ends + h))
                if h > 0:
                    st = rotation_stats(r.orch, ends, h)
                    row[f"p_E_h{h:g}"] = st["p_E"]
                    row[f"p_two_plus_h{h:g}"] = st["p_two_plus"]
            heur = np.array([heuristic_role_estimate(rd.X[rd.window_id == k], tie_rng) for k in range(len(rd.window_start))])
            row["role_acc_heur"] = role_accuracy_at_horizon(heur, orch_at(r.orch, ends))
            row["role_chance"] = 1.0 / r.n
            row["rotation_mean_interval_s"] = rotation_stats(r.orch, ends, 0.0)["mean_interval_s"]
            Xq, yq = phase[i]
            row["phase_acc"] = phase_accuracy(phase_model.predict(Xq), yq) if len(Xq) else float("nan")
            row["phase_majority"] = majority_rate(yq)
            Xw, _, ts = plan[i]
            ev = detect_events(ts, positive_proba(plan_model, Xw), cfg.plan_window[0], cfg.plan_threshold, cfg.plan_merge_s)
            sc = plan_change_f1(ev, np.asarray(r.plan_changes), cfg.plan_tolerance_s)
            row.update(plan_f1=sc.f1, plan_prec=sc.precision, plan_rec=sc.recall)
            out.append(row)
    log.info("evaluated %d missions", len(out))
    return out
