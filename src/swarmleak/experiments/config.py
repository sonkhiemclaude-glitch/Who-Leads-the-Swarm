"""Experiment configuration (YAML) -> typed objects. See configs/*.yaml and docs/implementation_notes.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..attack.evaluate import AttackConfig
from ..constants import (
    ALTITUDE_M,
    AREA_M,
    DEFAULT_MONITORS,
    EDGE_MONITOR,
    IP_MTU_B,
    MISSION_S,
    PATH_LOSS_EXPONENT,
    TX_POWER_DBM,
    WARMUP_S,
)
from ..protocol.config import ProtocolConfig, get_defense
from ..utils.io import load_yaml, stable_hash
from ..utils.validation import check_choice

MONITOR_PRESETS = {"quad": DEFAULT_MONITORS, "edge": EDGE_MONITOR}
TRACE_SOURCES = ("surrogate", "llm", "llm_mock")
SYNTHETIC_TRACE_SOURCES = ("surrogate", "llm_mock")


@dataclass(frozen=True)
class Scenario:
    n: int = 20
    duration: float = MISSION_S
    speed: float = 15.0
    area: float = AREA_M
    altitude: tuple[float, float] = ALTITUDE_M
    mobility: str = "gm"
    monitors: str = "quad"
    routing: str = "olsr"
    warmup_s: float = WARMUP_S
    tx_power_dbm: float = TX_POWER_DBM
    loss_exponent: float = PATH_LOSS_EXPONENT
    mtu_b: int = IP_MTU_B

    def __post_init__(self) -> None:
        check_choice("mobility", self.mobility, ("gm", "rpgm"))
        check_choice("monitors", self.monitors, tuple(MONITOR_PRESETS))
        check_choice("routing", self.routing, ("olsr", "aodv"))
        if self.n < 2 or self.duration <= 60:
            raise ValueError("scenario needs n >= 2 and duration > 60 s")

    def monitor_positions(self) -> tuple[tuple[float, float, float], ...]:
        """Monitor presets are defined for the 800 m area and scaled to the configured area."""
        s = self.area / AREA_M
        return tuple((x * s, y * s, z) for x, y, z in MONITOR_PRESETS[self.monitors])

    def key(self) -> str:
        return f"{self.routing}_n{self.n}_v{self.speed:g}_{self.mobility}_{self.monitors}"


@dataclass(frozen=True)
class GridEntry:
    defense: str
    budget: float
    scenario: Scenario
    missions: tuple[int, ...]

    def tag(self, mission: int) -> str:
        return f"{self.defense}_b{self.budget:g}_{self.scenario.key()}_m{mission}"

    def config_key(self) -> str:
        return f"{self.defense}_b{self.budget:g}_{self.scenario.key()}"


@dataclass
class ExperimentConfig:
    name: str
    output_dir: Path
    seed: int
    backend: str
    jobs: int
    trace_source: str
    trace_root: Path | None
    trace_pattern: str
    llm: dict[str, Any]
    toy: dict[str, Any]
    base_scenario: Scenario
    protocol: ProtocolConfig
    attack: AttackConfig
    grid: list[GridEntry]
    comparisons: list[dict[str, Any]]
    bootstrap: int
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def synthetic(self) -> bool:
        """True when outputs cannot be paper results (toy backend or synthetic traces)."""
        return self.backend == "toy" or self.trace_source in SYNTHETIC_TRACE_SOURCES

    @property
    def config_hash(self) -> str:
        return stable_hash(self.raw)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        return cls.from_dict(load_yaml(path), base_dir=Path(path).resolve().parent)

    @classmethod
    def from_dict(cls, d: dict[str, Any], base_dir: Path | None = None) -> ExperimentConfig:
        exp = dict(d.get("experiment") or {})
        data = dict(d.get("data") or {})
        sc = dict(d.get("scenario") or {})
        if "altitude" in sc:
            sc["altitude"] = tuple(sc["altitude"])
        base = Scenario(**sc)
        backend = check_choice("experiment.backend", str(exp.get("backend", "toy")), ("toy", "ns3"))
        source = check_choice("data.trace_source", str(data.get("trace_source", "surrogate")), TRACE_SOURCES)
        root = data.get("root")
        if source == "llm" and not root:
            raise ValueError("data.root is required when data.trace_source is 'llm'")
        out_dir = Path(str(exp.get("output_dir", f"outputs/{exp.get('name', 'experiment')}")))
        grid: list[GridEntry] = []
        for i, g in enumerate(d.get("grid") or []):
            get_defense(str(g["defense"]))
            budgets = g.get("budgets", [g.get("budget", 0.0)])
            missions = g.get("missions")
            if missions is None:
                missions = list(range(1, int(g.get("n_missions", 12)) + 1))
            over = dict(g.get("scenario") or {})
            if "altitude" in over:
                over["altitude"] = tuple(over["altitude"])
            scen = Scenario(**{**base.__dict__, **over})
            for b in budgets:
                if get_defense(str(g["defense"])).uses_budget and float(b) <= 0:
                    raise ValueError(f"grid[{i}]: defense {g['defense']} needs a positive budget")
                grid.append(GridEntry(str(g["defense"]), float(b), scen, tuple(int(m) for m in missions)))
        if not grid:
            raise ValueError("config has an empty grid")
        ev = dict(d.get("evaluation") or {})
        return cls(name=str(exp.get("name", "experiment")), output_dir=out_dir, seed=int(exp.get("seed", 2026)),
                   backend=backend, jobs=int(exp.get("jobs", 1)), trace_source=source,
                   trace_root=Path(str(root)).expanduser() if root else None,
                   trace_pattern=str(data.get("pattern", "llm_n{n}_m{mission}.json")),
                   llm=dict(d.get("llm") or {}), toy=dict(d.get("toy") or {}), base_scenario=base,
                   protocol=ProtocolConfig.from_dict(d.get("protocol")), attack=AttackConfig.from_dict(d.get("attack")),
                   grid=grid, comparisons=list(ev.get("comparisons") or []), bootstrap=int(ev.get("bootstrap", 10000)),
                   raw=d)
