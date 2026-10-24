"""Which baselines are real implementations of the paper's descriptions (Section 6, Configurations)."""
from __future__ import annotations

from dataclasses import dataclass

from ..protocol.config import DEFENSES


@dataclass(frozen=True)
class BaselineInfo:
    name: str
    kind: str  # defense | attacker
    status: str  # implemented | dummy_for_tests | adapter_stub
    paper_location: str
    description: str


PAPER_BASELINES: dict[str, BaselineInfo] = {
    "none": BaselineInfo("none", "defense", "implemented", "Sec. 6", "No defense"),
    "pad": BaselineInfo("pad", "defense", "implemented", "Sec. 6",
                        "Fixed-size padding of every message into 1024-byte chunks"),
    "rot": BaselineInfo("rot", "defense", "implemented", "Sec. 5.2, 6", "Role rotation only"),
    "cover_uniform": BaselineInfo("cover_uniform", "defense", "implemented", "Sec. 5.3",
                                  "Whole budget on spread decoys"),
    "cover_targeted": BaselineInfo("cover_targeted", "defense", "implemented", "Sec. 5.3, 6",
                                   "Targeted cover without rotation"),
    "const": BaselineInfo("const", "defense", "implemented", "Sec. 5.3",
                          "Padding + constant-rate 1024-B one-hop cover (BuFLO-style approximation; "
                          "real traffic is not delayed, unlike Tamaraw)"),
    "full_anyhop": BaselineInfo("full_anyhop", "defense", "implemented", "Sec. 5.4, 6",
                                "Full protocol with cover to arbitrary UAVs"),
    "heuristic": BaselineInfo("heuristic", "attacker", "implemented", "Sec. 4.4",
                              "Names the UAV transmitting the most unicast bytes in the window"),
    "chance": BaselineInfo("chance", "attacker", "implemented", "Sec. 3.3", "1/N role accuracy"),
}

PROPOSED = "full"


def undocumented_defenses() -> set[str]:
    """Defenses in the protocol registry that are neither the proposed method nor a listed baseline."""
    return set(DEFENSES) - {PROPOSED} - set(PAPER_BASELINES)
