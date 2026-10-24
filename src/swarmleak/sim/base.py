"""Backend interface (Section 4.2-4.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..constants import DEFAULT_MONITORS, IP_MTU_B, PATH_LOSS_EXPONENT, TX_POWER_DBM, WARMUP_S
from ..data.schemas import PhysicalMessage
from ..utils.validation import check_choice


@dataclass
class SimSettings:
    routing: str = "olsr"
    monitors: tuple[tuple[float, float, float], ...] = DEFAULT_MONITORS
    warmup_s: float = WARMUP_S
    tx_power_dbm: float = TX_POWER_DBM
    loss_exponent: float = PATH_LOSS_EXPONENT
    mtu_b: int = IP_MTU_B
    seed: int = 1  # ns-3 RngSeed / toy backend seed
    run: int = 1  # ns-3 RngRun
    extra: dict = field(default_factory=dict)  # backend-specific options

    def __post_init__(self) -> None:
        check_choice("routing", self.routing, ("olsr", "aodv"))
        if not self.monitors:
            raise ValueError("at least one monitor is required")


@dataclass
class SimOutput:
    frames: np.ndarray  # FRAME_DTYPE, raw (not yet de-duplicated)
    rx_time: np.ndarray  # per physical message, NaN if undelivered (mission clock, warm-up removed)
    energy_j: np.ndarray  # per UAV, total radio energy (includes idle)
    backend: str
    synthetic_backend: bool


class Backend(Protocol):
    name: str
    synthetic: bool

    def run(self, msgs: list[PhysicalMessage], traj: np.ndarray, settings: SimSettings) -> SimOutput: ...


def get_backend(name: str) -> Backend:
    check_choice("backend", name, ("ns3", "toy"))
    if name == "toy":
        from .toy_backend import ToyBackend

        return ToyBackend()
    from .ns3_backend import Ns3Backend

    return Ns3Backend()
