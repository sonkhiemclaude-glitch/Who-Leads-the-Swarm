"""Leakage metrics (Section 3.3, Definitions 1-3) and the plan-change detection rule (Section 4.4)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..constants import PLAN_MERGE_S, PLAN_THRESHOLD, PLAN_TOLERANCE_S

log = logging.getLogger(__name__)


def role_accuracy_at_horizon(estimates: np.ndarray, truth_at_horizon: np.ndarray) -> float:
    """Definition 1: Acc_role(D) = (1/K) sum_k 1[o_hat_k = o(t_k + W + D)]. NaN if K = 0."""
    e, t = np.asarray(estimates), np.asarray(truth_at_horizon)
    if e.shape != t.shape:
        raise ValueError("estimates and truth must have the same shape")
    if e.size == 0:
        log.warning("role accuracy over zero windows is undefined")
        return float("nan")
    return float(np.mean(e == t))


def phase_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    """Definition 2: fraction of windows whose phase is predicted correctly. NaN if no windows."""
    p, t = np.asarray(pred), np.asarray(truth)
    if p.shape != t.shape:
        raise ValueError("pred and truth must have the same shape")
    return float(np.mean(p == t)) if p.size else float("nan")


def majority_rate(labels: np.ndarray) -> float:
    """Share of the most frequent label (reference for phase accuracy)."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return float("nan")
    _, cnt = np.unique(labels, return_counts=True)
    return float(np.max(cnt)) / float(np.sum(cnt))


def detect_events(window_starts: np.ndarray, prob: np.ndarray, W: float, threshold: float = PLAN_THRESHOLD,
                  merge_s: float = PLAN_MERGE_S) -> np.ndarray:
    """Section 4.4: local maxima of the per-window probability with p >= threshold, merged when
    closer than merge_s; event time = window start + W/2. Plateaus keep their first point."""
    ts, p = np.asarray(window_starts, float), np.asarray(prob, float)
    if ts.shape != p.shape:
        raise ValueError("window_starts and prob must have the same shape")
    ev: list[float] = []
    for i in range(len(p)):
        left = p[i - 1] if i > 0 else -np.inf
        right = p[i + 1] if i < len(p) - 1 else -np.inf
        if p[i] >= threshold and p[i] >= left and p[i] >= right and not (i > 0 and p[i] == left):
            t = ts[i] + W / 2
            if not ev or t - ev[-1] > merge_s:
                ev.append(t)
    return np.array(ev)


@dataclass(frozen=True)
class EventScore:
    tp: int
    n_detected: int
    n_true: int
    precision: float
    recall: float
    f1: float


def match_events(detections: np.ndarray, truth: np.ndarray, tol: float = PLAN_TOLERANCE_S) -> int:
    """Definition 3: detections processed in temporal order; each is matched to the nearest
    unmatched true instant if it lies within tol. Returns the number of matched pairs (TP)."""
    det = np.sort(np.asarray(detections, float))
    tru = np.asarray(truth, float)
    used = np.zeros(len(tru), bool)
    tp = 0
    for e in det:
        if not len(tru):
            break
        d = np.abs(tru - e)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1
    return tp


def plan_change_f1(detections: np.ndarray, truth: np.ndarray, tol: float = PLAN_TOLERANCE_S) -> EventScore:
    """Precision TP/|T_hat|, recall TP/|T|, F1 harmonic mean (Definition 3).
    Undefined parts are NaN (audit M4): precision with no detections, recall with no true events;
    F1 is 0 when both are defined and TP = 0, NaN when both sets are empty."""
    det, tru = np.asarray(detections, float), np.asarray(truth, float)
    tp = match_events(det, tru, tol)
    prec = tp / len(det) if len(det) else float("nan")
    rec = tp / len(tru) if len(tru) else float("nan")
    if len(det) == 0 and len(tru) == 0:
        f1 = float("nan")
    elif tp == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return EventScore(tp, len(det), len(tru), prec, rec, f1)
