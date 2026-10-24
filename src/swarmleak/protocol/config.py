"""Protocol and defense configuration (Section 5, Table 2)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from ..constants import (
    ACK_DELAY_S,
    BURST_GAP_S,
    CONST_COVER_SIZE_B,
    DECOY_DELIVERY_ASSUMPTION_S,
    DECOY_START_S,
    DEFAULT_SIZE_PROFILE_B,
    LINK_GAP_S,
    N_COTIMED_DECOYS,
    PAD_CHUNK_B,
    RANGE_ESTIMATE_M,
    ROTATION_DELAY_S,
    ROTATION_TOP_K,
    SIZE_MEMORY_MIN_SAMPLES,
    TARGETED_SHARES,
    TOKEN_BUCKET_CAPACITY,
    TOKEN_BUCKET_RATE_HZ,
)
from ..utils.validation import check_choice


@dataclass
class ProtocolConfig:
    range_est_m: float = RANGE_ESTIMATE_M
    rotation_delay_s: tuple[float, float] = ROTATION_DELAY_S
    rotation_top_k: int = ROTATION_TOP_K
    size_profile_b: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SIZE_PROFILE_B))
    size_memory_min: int = SIZE_MEMORY_MIN_SAMPLES
    burst_gap_s: tuple[float, float] = BURST_GAP_S
    ack_delay_s: tuple[float, float] = ACK_DELAY_S
    decoy_delivery_s: float = DECOY_DELIVERY_ASSUMPTION_S
    decoy_start_s: tuple[float, float] = DECOY_START_S
    n_cotimed_decoys: int = N_COTIMED_DECOYS
    targeted_shares: dict[str, float] = field(default_factory=lambda: dict(TARGETED_SHARES))
    bucket_capacity: float = TOKEN_BUCKET_CAPACITY
    bucket_rate_hz: float = TOKEN_BUCKET_RATE_HZ
    link_gap_s: float = LINK_GAP_S
    pad_chunk_b: int = PAD_CHUNK_B
    const_size_b: int = CONST_COVER_SIZE_B

    def __post_init__(self) -> None:
        s = sum(self.targeted_shares.values())
        if abs(s - 1.0) > 1e-9 or set(self.targeted_shares) != {"cotimed", "spread", "mirror"}:
            raise ValueError("targeted_shares must have keys cotimed/spread/mirror summing to 1")
        if self.rotation_top_k < 1 or self.bucket_capacity < 1 or self.bucket_rate_hz <= 0:
            raise ValueError("invalid rotation_top_k / token bucket settings")
        self.rotation_delay_s = tuple(self.rotation_delay_s)  # type: ignore[assignment]
        self.burst_gap_s = tuple(self.burst_gap_s)  # type: ignore[assignment]
        self.ack_delay_s = tuple(self.ack_delay_s)  # type: ignore[assignment]
        self.decoy_start_s = tuple(self.decoy_start_s)  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ProtocolConfig:
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown protocol config keys: {sorted(unknown)}")
        return cls(**d)


@dataclass(frozen=True)
class DefenseSpec:
    """One configuration of Section 6 ('Configurations')."""

    name: str
    rotate: bool = False
    cover: str | None = None  # None | uniform | targeted | const
    pad: bool = False
    cover_hops: str = "one"  # one | any

    def __post_init__(self) -> None:
        if self.cover is not None:
            check_choice("cover", self.cover, ("uniform", "targeted", "const"))
        check_choice("cover_hops", self.cover_hops, ("one", "any"))

    @property
    def uses_budget(self) -> bool:
        return self.cover is not None


DEFENSES: dict[str, DefenseSpec] = {
    "none": DefenseSpec("none"),
    "pad": DefenseSpec("pad", pad=True),
    "rot": DefenseSpec("rot", rotate=True),
    "cover_uniform": DefenseSpec("cover_uniform", cover="uniform"),
    "cover_targeted": DefenseSpec("cover_targeted", cover="targeted"),
    "full": DefenseSpec("full", rotate=True, cover="targeted"),
    "const": DefenseSpec("const", cover="const", pad=True),
    "full_anyhop": DefenseSpec("full_anyhop", rotate=True, cover="targeted", cover_hops="any"),
}
"""Defense names used throughout configs, run files and tables."""


def get_defense(name: str) -> DefenseSpec:
    if name not in DEFENSES:
        raise ValueError(f"unknown defense {name!r}; expected one of {sorted(DEFENSES)}")
    return DEFENSES[name]
