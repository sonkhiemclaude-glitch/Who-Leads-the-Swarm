"""Deterministic seed derivation.

Every random stream is derived from (base seed, purpose string, integers) through numpy's
SeedSequence, so runs are reproducible and streams for different purposes are independent.
Python's built-in hash() is not used because it is salted per process.
"""
from __future__ import annotations

import zlib

import numpy as np


def _purpose_key(purpose: str) -> int:
    return zlib.crc32(purpose.encode("utf-8")) & 0xFFFFFFFF


def derive_seed_sequence(base_seed: int, purpose: str, *ints: int) -> np.random.SeedSequence:
    if base_seed < 0 or any(i < 0 for i in ints):
        raise ValueError("seeds and keys must be non-negative integers")
    return np.random.SeedSequence([int(base_seed), _purpose_key(purpose), *[int(i) for i in ints]])


def rng_for(base_seed: int, purpose: str, *ints: int) -> np.random.Generator:
    """Generator for one purpose, e.g. rng_for(seed, "mobility", mission)."""
    return np.random.default_rng(derive_seed_sequence(base_seed, purpose, *ints))


def int_seed(base_seed: int, purpose: str, *ints: int) -> int:
    """A 31-bit integer seed (for libraries that need an int, e.g. scikit-learn, ns-3 RngRun)."""
    return int(derive_seed_sequence(base_seed, purpose, *ints).generate_state(1)[0] % (2**31 - 1)) + 1
