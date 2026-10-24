"""Validate cached runs before resume or aggregation; no network access."""
from __future__ import annotations

from pathlib import Path

from ..data.schemas import RunRecord
from .config import ExperimentConfig, GridEntry
from .provenance import source_digest


def expected_runs(cfg: ExperimentConfig) -> dict[str, tuple[GridEntry, int]]:
    expected: dict[str, tuple[GridEntry, int]] = {}
    for entry in cfg.grid:
        for mission in entry.missions:
            tag = entry.tag(mission)
            if tag in expected:
                raise ValueError(f"duplicate run tag in grid: {tag}")
            expected[tag] = (entry, mission)
    return expected


def validate_run(rec: RunRecord, cfg: ExperimentConfig, entry: GridEntry, mission: int,
                 trace_hash: str | None = None) -> None:
    expected = {"config_hash": cfg.config_hash, "source_sha256": source_digest(),
                "backend": cfg.backend, "trace_source": cfg.trace_source,
                "tag": entry.tag(mission), "config_key": entry.config_key(), "mission": mission}
    if trace_hash is not None:
        expected["trace_sha256"] = trace_hash
    for key, value in expected.items():
        if rec.meta.get(key) != value:
            raise ValueError(f"incompatible cached run {entry.tag(mission)}: {key} differs or is missing. "
                             "Use a separate output directory or explicitly rerun with --overwrite.")


def validate_run_files(cfg: ExperimentConfig) -> list[Path]:
    expected = expected_runs(cfg)
    root = cfg.output_dir / "runs"
    found = {p.stem: p for p in root.glob("*.npz")}
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing or extra:
        raise ValueError(f"incomplete or mixed run directory: {len(missing)} missing, {len(extra)} unexpected. "
                         f"Missing (first 5): {missing[:5]}; unexpected (first 5): {extra[:5]}")
    return [found[tag] for tag in sorted(expected)]
