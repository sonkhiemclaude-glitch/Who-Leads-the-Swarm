import copy

import pytest

from swarmleak.data.schemas import RunRecord
from swarmleak.experiments.config import ExperimentConfig
from swarmleak.experiments.evaluation import evaluate_runs
from swarmleak.experiments.preflight import preflight
from swarmleak.experiments.runner import run_grid, run_one


def config(tmp_path):
    return ExperimentConfig.from_dict({
        "experiment": {"name": "readiness", "backend": "toy", "output_dir": str(tmp_path)},
        "scenario": {"n": 3, "duration": 90},
        "grid": [{"defense": "none", "missions": [1, 2]}],
        "attack": {"rf_trees": 5, "folds": 2},
    })


def test_resume_accepts_identical_inputs_and_rejects_changed_config(tmp_path):
    cfg = config(tmp_path)
    entry = cfg.grid[0]
    path = run_one(cfg, entry, 1)
    mtime = path.stat().st_mtime_ns
    assert run_one(cfg, entry, 1) == path
    assert path.stat().st_mtime_ns == mtime
    changed = copy.deepcopy(cfg.raw)
    changed["experiment"]["seed"] = 99
    cfg2 = ExperimentConfig.from_dict(changed)
    with pytest.raises(ValueError, match="config_hash"):
        run_one(cfg2, cfg2.grid[0], 1)


def test_resume_rejects_changed_trace_content(tmp_path, monkeypatch):
    from swarmleak.experiments import runner

    cfg = config(tmp_path)
    entry = cfg.grid[0]
    run_one(cfg, entry, 1)
    original = runner.load_trace

    def altered(*args):
        tr = original(*args)
        tr.source = "changed"
        return tr

    monkeypatch.setattr(runner, "load_trace", altered)
    with pytest.raises(ValueError, match="trace_sha256"):
        run_one(cfg, entry, 1)


def test_resume_rejects_missing_source_fingerprint(tmp_path):
    cfg = config(tmp_path)
    path = run_one(cfg, cfg.grid[0], 1)
    rec = RunRecord.load(path)
    del rec.meta["source_sha256"]
    rec.save(path)
    with pytest.raises(ValueError, match="source_sha256"):
        run_one(cfg, cfg.grid[0], 1)


def test_evaluation_refuses_incomplete_grid_before_writing_metrics(tmp_path):
    cfg = config(tmp_path)
    run_one(cfg, cfg.grid[0], 1)
    with pytest.raises(ValueError, match="1 missing"):
        evaluate_runs(cfg)
    assert not (tmp_path / "metrics.csv").exists()


def test_evaluation_refuses_extra_runs(tmp_path):
    cfg = config(tmp_path)
    run_grid(cfg)
    path = next((tmp_path / "runs").glob("*.npz"))
    (path.parent / "unexpected.npz").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="1 unexpected"):
        evaluate_runs(cfg)


def test_failed_run_propagates_after_manifest_is_written(tmp_path, monkeypatch):
    from swarmleak.experiments import runner

    cfg = config(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("test backend failure")

    monkeypatch.setattr(runner, "run_one", fail)
    with pytest.raises(RuntimeError, match="2/2 runs failed"):
        run_grid(cfg)
    assert "test backend failure" in (tmp_path / "runs" / "manifest.csv").read_text()


def test_duplicate_run_tags_are_rejected(tmp_path):
    cfg = config(tmp_path)
    cfg.grid.append(cfg.grid[0])
    with pytest.raises(ValueError, match="duplicate"):
        run_grid(cfg)


def test_preflight_is_read_only_and_counts_required_traces(tmp_path):
    cfg = config(tmp_path)
    report = preflight(cfg)
    assert report["technical_checks_passed"]
    assert report["planned_runs"] == 2 and report["required_trace_files"] == 2
    assert not (tmp_path / "runs").exists()


def test_real_generation_preflight_requires_model_and_endpoint(tmp_path):
    cfg = config(tmp_path)
    report = preflight(cfg, "gen-traces")
    assert not report["technical_checks_passed"]
    assert any("llm.endpoint" in issue for issue in report["issues"])
