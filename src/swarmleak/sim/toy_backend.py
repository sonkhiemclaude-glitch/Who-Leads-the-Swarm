"""TOY replay backend for tests and smoke runs. NOT a network simulator. NOT for paper results.

It replaces ns-3 with a deliberately simple geometric model so that the full pipeline (mapping,
defenses, frame logs, features, classifiers, metrics, tables, plots) can be tested quickly and
offline:
- unit-disk links of `link_range_m` evaluated at 1 s position samples; shortest-path forwarding;
- per-hop frame loss with up to 7 MAC retries (retry flag set on retransmissions);
- fragments at the IP MTU; frame size = fragment payload + 64 B headers;
- periodic HELLO broadcasts (OLSR 2 s, AODV 1 s); AODV additionally floods a route request when
  the source has no cached route (3 s lifetime) to the destination;
- monitors decode frames whose transmitter is within `monitor_range_m`;
- energy = idle current over the whole run + incremental tx/rx per frame (no overhearing).
No contention, interference, routing convergence or propagation physics is modelled.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..constants import I_IDLE_A, I_RX_A, I_TX_A, PER_FRAGMENT_OVERHEAD_B, REFERENCE_LOSS_DB, SUPPLY_VOLTAGE_V
from ..data.schemas import FRAME_DTYPE, PhysicalMessage
from ..mobility.topology import neighbours
from ..protocol.energy import RadioModel
from .base import SimOutput, SimSettings

HELLO_SIZE_B = 60
RREQ_SIZE_B = 72
MAX_RETRIES = 7


class ToyBackend:
    name = "toy"
    synthetic = True

    def run(self, msgs: list[PhysicalMessage], traj: np.ndarray, settings: SimSettings) -> SimOutput:
        opts = {"link_range_m": 320.0, "monitor_range_m": 330.0, "hop_loss": 0.02, **settings.extra}
        rng = np.random.default_rng([settings.seed, settings.run, 7357])
        radio = RadioModel(mtu_b=settings.mtu_b)
        steps, n, _ = traj.shape
        duration = float(steps - 1)
        mons = np.array(settings.monitors, float)
        adj_cache: dict[int, np.ndarray] = {}
        seq = np.zeros(n, int)
        e_inc = np.zeros(n)
        frames: list[tuple] = []
        route_cache: dict[tuple[int, int], float] = {}

        def adj(t: float) -> np.ndarray:
            k = min(max(int(t), 0), steps - 1)
            if k not in adj_cache:
                adj_cache[k] = neighbours(traj[k], opts["link_range_m"])
            return adj_cache[k]

        def emit(t: float, tx: int, rx: int, size: int, retry: int) -> None:
            k = min(max(int(t), 0), steps - 1)
            p = traj[k, tx]
            d = np.linalg.norm(mons - p, axis=1)
            s = int(seq[tx] % 4096)
            for mi in np.flatnonzero(d < opts["monitor_range_m"]):
                rssi = settings.tx_power_dbm - (REFERENCE_LOSS_DB + 10 * settings.loss_exponent * np.log10(max(d[mi], 1.0)))
                frames.append((t, size, tx, rx, s, retry, int(mi), rssi))
            seq[tx] += 1

        def airtime_frame(payload: int) -> float:
            return radio.airtime(payload) / radio.fragments(payload)

        def broadcast(t: float, tx: int, size: int) -> None:
            emit(t, tx, -1, size, 0)
            a = airtime_frame(size)
            e_inc[tx] += SUPPLY_VOLTAGE_V * (I_TX_A - I_IDLE_A) * a
            for j in np.flatnonzero(adj(t)[tx]):
                e_inc[j] += SUPPLY_VOLTAGE_V * (I_RX_A - I_IDLE_A) * a

        period = 2.0 if settings.routing == "olsr" else 1.0
        for i in range(n):
            t = float(rng.uniform(0, period))
            while t < duration:
                broadcast(t, i, HELLO_SIZE_B)
                t += period * float(rng.uniform(0.95, 1.05))

        def path(t: float, s: int, d: int) -> list[int] | None:
            a = adj(t)
            prev = {s: -1}
            q = deque([s])
            while q:
                u = q.popleft()
                if u == d:
                    break
                for vv in np.flatnonzero(a[u]):
                    v = int(vv)
                    if v not in prev:
                        prev[v] = u
                        q.append(v)
            if d not in prev:
                return None
            p = [d]
            while p[-1] != s:
                p.append(prev[p[-1]])
            return p[::-1]

        rx_time = np.full(len(msgs), np.nan)
        for m in sorted(range(len(msgs)), key=lambda i: msgs[i].t):
            msg = msgs[m]
            t = msg.t
            if settings.routing == "aodv" and route_cache.get((msg.src, msg.dst), -1e9) < t - 3.0:
                for i in range(n):  # route request flood: every UAV rebroadcasts once
                    broadcast(t + 0.002 * i, i, RREQ_SIZE_B)
                route_cache[(msg.src, msg.dst)] = t
                t += 0.002 * n
            p = path(t, msg.src, msg.dst)
            if p is None:
                continue
            nfrag = radio.fragments(msg.size)
            frag_payload = int(np.ceil(msg.size / nfrag))
            delivered = True
            for u, v in zip(p, p[1:]):
                for _ in range(nfrag):
                    a = airtime_frame(frag_payload)
                    ok = False
                    for attempt in range(MAX_RETRIES + 1):
                        emit(t, u, v, frag_payload + PER_FRAGMENT_OVERHEAD_B, int(attempt > 0))
                        e_inc[u] += SUPPLY_VOLTAGE_V * (I_TX_A - I_IDLE_A) * a
                        e_inc[v] += SUPPLY_VOLTAGE_V * (I_RX_A - I_IDLE_A) * a
                        t += a + float(rng.uniform(0, 0.002))
                        if rng.random() >= opts["hop_loss"]:
                            ok = True
                            break
                    if not ok:
                        delivered = False
                        break
                if not delivered:
                    break
            if delivered:
                rx_time[m] = t
        F = np.array(frames, dtype=FRAME_DTYPE) if frames else np.zeros(0, dtype=FRAME_DTYPE)
        F = F[np.argsort(F["t"], kind="stable")]
        energy = SUPPLY_VOLTAGE_V * I_IDLE_A * (duration + settings.warmup_s) + e_inc
        return SimOutput(F, rx_time, energy, self.name, True)
