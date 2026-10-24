"""Cost metrics (Section 3.4): PDR, latency, energy, airtime, observed broadcast frames."""
from __future__ import annotations

import numpy as np

from ..data.schemas import RunRecord
from ..protocol.energy import DEFAULT_RADIO, RadioModel


def logical_delivery(r: RunRecord) -> tuple[np.ndarray, np.ndarray]:
    """Per real logical message (grouped by gid): delivered flag (all chunks arrived) and latency
    (first chunk send -> last chunk arrival; NaN if undelivered). Handovers count as real."""
    real = r.msg_real.astype(bool)
    g, rx, t = r.msg_gid[real], r.rx_time[real], r.msg_t[real]
    if len(g) == 0:
        return np.zeros(0, bool), np.zeros(0)
    order = np.argsort(g, kind="stable")
    g, rx, t = g[order], rx[order], t[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    delivered = np.minimum.reduceat((~np.isnan(rx)).astype(int), starts).astype(bool)
    last = np.maximum.reduceat(np.where(np.isnan(rx), -np.inf, rx), starts)
    first = np.minimum.reduceat(t, starts)
    lat = np.where(delivered, last - first, np.nan)
    return delivered, lat


def network_metrics(r: RunRecord, radio: RadioModel = DEFAULT_RADIO) -> dict[str, float]:
    delivered, lat = logical_delivery(r)
    ok = lat[delivered]
    T, n = r.duration, r.n
    real = r.msg_real.astype(bool)
    cover_air = float(sum(radio.airtime(int(s)) for s in r.msg_size[~real]))
    real_air = float(sum(radio.airtime(int(s)) for s in r.msg_size[real]))
    F = r.frames[(r.frames["t"] >= 0) & (r.frames["t"] < T)]
    return {
        "pdr": float(delivered.mean()) if len(delivered) else float("nan"),
        "lat_ms_median": float(np.median(ok) * 1e3) if len(ok) else float("nan"),
        "lat_ms_p95": float(np.percentile(ok, 95) * 1e3) if len(ok) else float("nan"),
        "energy_j_mean": float(np.mean(r.energy_j)),
        "energy_j_max": float(np.max(r.energy_j)),
        "bcast_fps": float((F["rx"] < 0).sum() / T),
        "frames_fps": float(len(F) / T),
        "retry_frac": float(F["retry"].mean()) if len(F) else 0.0,
        "n_cover": int((~real).sum()),
        "n_real_logical": int(len(delivered)),
        "cover_airtime_pct": 100.0 * cover_air / (n * T),
        "real_airtime_pct": 100.0 * real_air / (n * T),
    }


def incremental_energy(defended: RunRecord, undefended: RunRecord) -> float:
    """Measured incremental radio energy per UAV, paired with the undefended run of the same
    mission, backend and seed (Section 3.4; audit M21)."""
    if defended.energy_j.shape != undefended.energy_j.shape:
        raise ValueError("paired runs must have the same number of UAVs")
    return float(np.mean(defended.energy_j - undefended.energy_j))
