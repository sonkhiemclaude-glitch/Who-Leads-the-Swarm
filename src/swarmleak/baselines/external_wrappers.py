"""Adapter stubs for methods the paper names but does not implement.

None of these is implemented. Each documents the input/output schema the pipeline would need, so
that a real implementation can be plugged in later. Calling `run` raises NotImplementedError.
See docs/baseline_adapters.md and TODO_IMPLEMENTATION.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.schemas import PhysicalMessage


class ExternalBaselineNotAvailable(NotImplementedError):
    pass


@dataclass
class SequenceModelAttacker:
    """Sequence-model attacker over frame streams (Discussion: 'Strength of the adversary').

    Input:  frame log windows, array FRAME_DTYPE per window, plus n_uavs.
    Output: per-window score vector over UAVs (float[n_uavs]) for role; class probabilities for phase;
            per-window plan-change probability.
    Status: not implemented; the paper marks it pending / future work.
    """

    name: str = "sequence_model"

    def fit(self, windows: list[np.ndarray], labels: list[np.ndarray]) -> SequenceModelAttacker:
        raise ExternalBaselineNotAvailable("sequence-model attacker is not implemented (TODO_IMPLEMENTATION.md, T-ATT-1)")

    def score(self, window: np.ndarray, n_uavs: int) -> np.ndarray:
        raise ExternalBaselineNotAvailable("sequence-model attacker is not implemented")


@dataclass
class AnonymousRoutingDefense:
    """ANODR / MASK-style anonymous routing (Related work; Discussion).

    Would require a routing-layer implementation in ns-3 (pseudonymous per-hop MAC/route ids).
    Input/Output when implemented: same as the ns-3 backend (physical messages -> SimOutput) with
    pseudonymous tx/rx in the frame log.
    Status: intentionally out of scope in the paper; adapter only.
    """

    variant: str = "ANODR"

    def run(self, msgs: list[PhysicalMessage]) -> None:
        raise ExternalBaselineNotAvailable(f"{self.variant} is not implemented (out of scope)")


@dataclass
class TamarawDefense:
    """Exact Tamaraw (constant-rate with delayed real traffic, Cai et al. 2014).

    The implemented `const` baseline does not delay real traffic. An exact Tamaraw would need a
    per-link queue releasing packets at fixed intervals.
    Input: list[PhysicalMessage]; Output: list[PhysicalMessage] with delayed real messages + padding.
    """

    def run(self, msgs: list[PhysicalMessage]) -> list[PhysicalMessage]:
        raise ExternalBaselineNotAvailable("exact Tamaraw is not implemented; use defense 'const'")


@dataclass
class WtfPadDefense:
    """WTF-PAD adaptive padding (Juarez et al. 2016). Needs histogram distributions from real traces.
    Input/Output: list[PhysicalMessage] -> list[PhysicalMessage]."""

    def run(self, msgs: list[PhysicalMessage]) -> list[PhysicalMessage]:
        raise ExternalBaselineNotAvailable("WTF-PAD is not implemented")
