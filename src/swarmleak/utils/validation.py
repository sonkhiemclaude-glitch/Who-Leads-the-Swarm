"""Input validation helpers that raise clear errors."""
from __future__ import annotations

from collections.abc import Iterable


def check_probability(name: str, value: float) -> float:
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return float(value)


def check_positive(name: str, value: float, allow_zero: bool = False) -> float:
    v = float(value)
    if v < 0 or (v == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}, got {value}")
    return v


def check_interval(name: str, interval: Iterable[float]) -> tuple[float, float]:
    lo, hi = (float(x) for x in interval)
    if lo > hi:
        raise ValueError(f"{name} must satisfy low <= high, got ({lo}, {hi})")
    return lo, hi


def check_choice(name: str, value: str, choices: Iterable[str]) -> str:
    c = tuple(choices)
    if value not in c:
        raise ValueError(f"{name} must be one of {c}, got {value!r}")
    return value
