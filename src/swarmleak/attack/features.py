"""Window features from the merged monitor frame log (Section 4.3).

Role (per UAV and window): frames and bytes transmitted; frames and bytes received as MAC
receiver; distinct MAC receivers (fan-out); distinct transmitters towards the UAV (fan-in);
transmitted-size histogram (edges 200/500/1000/1400 B); mean transmitted size; burstiness (largest
share of the UAV's transmitted frames in any one second); broadcast frames transmitted; fan-out of
frames >= 1400 B; fan-in of frames < 500 B. Each of these 16 features is also standardised across
the UAVs of the window -> 32 columns.

Window aggregate (phase and plan): unicast frames, bytes, broadcast frames, size histogram, mean
size, coefficient of variation of inter-frame times, active transmitters, concentration of
transmissions, largest fan-out of large frames, number of large frames -> 14 columns.
Plan features append first differences from the previous window -> 28 columns.
"""
from __future__ import annotations

import numpy as np

from ..constants import LARGE_FRAME_B, SIZE_BINS_B, SMALL_FRAME_B

ROLE_FEATURE_NAMES = ("tx_frames", "tx_bytes", "rx_frames", "rx_bytes", "fan_out", "fan_in",
                      "hist_0_200", "hist_200_500", "hist_500_1000", "hist_1000_1400", "hist_1400_plus",
                      "mean_tx_size", "burstiness", "bcast_tx", "fan_out_large", "fan_in_small")
WINDOW_FEATURE_NAMES = ("uni_frames", "uni_bytes", "bcast_frames", "hist_0_200", "hist_200_500",
                        "hist_500_1000", "hist_1000_1400", "hist_1400_plus", "mean_size", "iat_cv",
                        "active_tx", "tx_concentration", "max_fan_out_large", "large_frames")
TX_BYTES_COL = 1  # used by the heuristic attacker (most transmitted unicast bytes)


def window(frames: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return frames[(frames["t"] >= t0) & (frames["t"] < t1)]


def role_features(frames: np.ndarray, n: int, t0: float, t1: float) -> np.ndarray:
    """Per-UAV role features for window [t0, t1): shape [n, 32] (16 raw + 16 standardised)."""
    w = window(frames, t0, t1)
    uni = w[w["rx"] >= 0]
    bc = w[w["rx"] < 0]
    X = np.zeros((n, len(ROLE_FEATURE_NAMES)))
    for i in range(n):
        tx = uni[uni["tx"] == i]
        rx = uni[uni["rx"] == i]
        X[i, 0] = len(tx)
        X[i, 1] = tx["size"].sum()
        X[i, 2] = len(rx)
        X[i, 3] = rx["size"].sum()
        X[i, 4] = len(np.unique(tx["rx"]))
        X[i, 5] = len(np.unique(rx["tx"]))
        X[i, 6:11] = np.histogram(tx["size"], SIZE_BINS_B)[0]
        X[i, 11] = tx["size"].mean() if len(tx) else 0.0
        if len(tx) > 1:
            sec = np.floor(tx["t"]).astype(int)
            X[i, 12] = float(np.max(np.bincount(sec - sec.min()))) / len(tx)
        X[i, 13] = int((bc["tx"] == i).sum())
        X[i, 14] = len(np.unique(tx[tx["size"] >= LARGE_FRAME_B]["rx"]))
        X[i, 15] = len(np.unique(rx[rx["size"] < SMALL_FRAME_B]["tx"]))
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return np.hstack([X, (X - mu) / sd])


def window_features(frames: np.ndarray, n: int, t0: float, t1: float) -> np.ndarray:
    """Window aggregate features: shape [14]."""
    w = window(frames, t0, t1)
    uni = w[w["rx"] >= 0]
    f = np.zeros(len(WINDOW_FEATURE_NAMES))
    f[0] = len(uni)
    f[1] = uni["size"].sum()
    f[2] = int((w["rx"] < 0).sum())
    f[3:8] = np.histogram(uni["size"], SIZE_BINS_B)[0]
    f[8] = uni["size"].mean() if len(uni) else 0.0
    if len(uni) > 2:
        it = np.diff(uni["t"])
        f[9] = it.std() / (it.mean() + 1e-9)
    txs = uni["tx"][uni["tx"] >= 0]
    c = np.bincount(txs, minlength=n) if len(txs) else np.zeros(n)
    f[10] = int((c > 0).sum())
    f[11] = float(np.max(c)) / (float(np.sum(c)) + 1e-9) if len(c) else 0.0
    big = uni[uni["size"] >= LARGE_FRAME_B]
    if len(big):
        f[12] = max(len(np.unique(big[big["tx"] == i]["rx"])) for i in np.unique(big["tx"]))
    f[13] = len(big)
    return f
