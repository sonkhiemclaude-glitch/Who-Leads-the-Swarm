"""Adapters for real inputs. Nothing is downloaded; files must be placed manually (see
docs/data_format.md and docs/mac_studio_experiment_guide.md).

TraceDirectoryAdapter: real LLM-agent traces, one JSON file per (swarm size, mission), produced by
    `swarmleak gen-traces` (or any generator writing the same schema) on the Mac Studio.
RunDirectoryAdapter: simulation outputs (`runs/*.npz`) written by the runner.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .schemas import RunRecord, Trace

DEFAULT_TRACE_PATTERN = "llm_n{n}_m{mission}.json"


class MissingDataError(FileNotFoundError):
    pass


@dataclass
class TraceDirectoryAdapter:
    root: Path
    pattern: str = DEFAULT_TRACE_PATTERN

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()

    def path(self, n: int, mission: int) -> Path:
        return self.root / self.pattern.format(n=n, mission=mission)

    def load(self, n: int, mission: int) -> Trace:
        p = self.path(n, mission)
        if not p.exists():
            raise MissingDataError(
                f"trace not found: {p}. Generate it on the Mac Studio with `swarmleak gen-traces` "
                f"(see docs/mac_studio_experiment_guide.md) or set data.root/data.pattern.")
        tr = Trace.load_json(p)
        if tr.n_agents != n:
            raise ValueError(f"{p}: trace has {tr.n_agents} agents, expected {n}")
        return tr

    def available(self) -> list[tuple[int, int]]:
        rx = re.compile("^" + re.escape(self.pattern).replace(r"\{n\}", r"(\d+)").replace(r"\{mission\}", r"(\d+)") + "$")
        out = []
        if self.root.exists():
            for f in sorted(self.root.iterdir()):
                m = rx.match(f.name)
                if m:
                    out.append((int(m.group(1)), int(m.group(2))))
        return out


@dataclass
class RunDirectoryAdapter:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()

    def files(self) -> list[Path]:
        return sorted(self.root.glob("*.npz"))

    def load_all(self) -> list[RunRecord]:
        fs = self.files()
        if not fs:
            raise MissingDataError(f"no run files (*.npz) in {self.root}")
        return [RunRecord.load(f) for f in fs]
