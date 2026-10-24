"""Per-UAV memory of message sizes it has sent or received (Section 5.3).

Cover sizes are drawn from sizes of the same kind the UAV has observed *before* the cover time,
or from a pre-shared default profile until it has `min_samples` samples, so the scheduler never
uses future information.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..constants import DEFAULT_SIZE_PROFILE_B, SIZE_MEMORY_MIN_SAMPLES

KIND_MAP = {"feedback": "report", "tool_result": "report", "tool_call": "ack"}


class SizeMemory:
    def __init__(self, n: int, profile: dict[str, int] | None = None,
                 min_samples: int = SIZE_MEMORY_MIN_SAMPLES) -> None:
        self.profile = dict(profile or DEFAULT_SIZE_PROFILE_B)
        self.min_samples = int(min_samples)
        self._mem: list[dict[str, list[tuple[float, int]]]] = [defaultdict(list) for _ in range(n)]

    def see(self, node: int, kind: str, size: int, t: float) -> None:
        k = KIND_MAP.get(kind, kind)
        if k in self.profile:
            self._mem[node][k].append((float(t), int(size)))

    def draw(self, node: int, kind: str, rng: np.random.Generator, t: float) -> int:
        if kind not in self.profile:
            raise ValueError(f"no size profile for kind {kind!r}")
        hist = [sz for (ts, sz) in self._mem[node][kind] if ts < t]
        if len(hist) >= self.min_samples:
            return int(rng.choice(hist))
        return int(np.clip(rng.lognormal(np.log(self.profile[kind]), 0.35), 80, 12000))
