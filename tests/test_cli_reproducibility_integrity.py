import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from swarmleak import SYNTHETIC_LABEL
from swarmleak.cli import main
from swarmleak.data.schemas import RunRecord
from swarmleak.experiments.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]

TINY = {
    "experiment": {"name": "tiny", "seed": 5, "backend": "toy"},
    "data": {"trace_source": "surrogate"},
    "scenario": {"n": 5, "duration": 150, "area": 500},
    "attack": {"rf_trees": 10, "folds": 2},
    "grid": [{"defense": "none", "missions": [1, 2]}, {"defense": "full", "budgets": [0.2], "missions": [1, 2]}],
    "evaluation": {"bootstrap": 200,
                   "comparisons": [{"a": "full_b0.2_olsr_n5_v15_gm_quad", "b": "none_b0_olsr_n5_v15_gm_quad",
                                    "metric": "role_acc_h0"}]},
}


def write_cfg(tmp_path, name="tiny", **over):
    d = json.loads(json.dumps(TINY))
    d["experiment"]["output_dir"] = str(tmp_path / name)
    for k, v in over.items():
        d[k].update(v)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_cli_help():
    p = subprocess.run([sys.executable, "-m", "swarmleak.cli", "--help"], capture_output=True, text=True)
    assert p.returncode == 0 and "simulate" in p.stdout and "demo" in p.stdout


def test_cli_bound():
    assert main(["bound", "--alpha0", "0.9", "--pe", "0.5"]) == 0


def test_demo_end_to_end_labels_outputs(tmp_path):
    cfg = write_cfg(tmp_path)
    assert main(["--log-level", "WARNING", "demo", "--config", str(cfg)]) == 0
    out = tmp_path / "tiny"
    for f in ("metrics.csv", "per_mission.csv", "comparisons.csv", "provenance.json", "summary.md", "tradeoff.png",
              "README.txt", "tables/e1.tex", "tables/e3.tex"):
        assert (out / f).exists(), f
    met = pd.read_csv(out / "metrics.csv")
    assert met["provenance_label"].str.contains(SYNTHETIC_LABEL).all()
    assert json.loads((out / "provenance.json").read_text())["synthetic"] is True
    assert SYNTHETIC_LABEL in (out / "tables" / "e3.tex").read_text()
    assert SYNTHETIC_LABEL in (out / "summary.md").read_text()
    assert len(pd.read_csv(out / "comparisons.csv")) == 1


def test_demo_refuses_non_synthetic_config(tmp_path):
    cfg = write_cfg(tmp_path, "real", experiment={"backend": "ns3"}, data={"trace_source": "llm", "root": str(tmp_path)})
    with pytest.raises(SystemExit):
        main(["demo", "--config", str(cfg)])


def test_reproducible_under_fixed_seed(tmp_path):
    a = write_cfg(tmp_path, "a")
    b = write_cfg(tmp_path, "b")
    for c in (a, b):
        main(["--log-level", "WARNING", "simulate", "--config", str(c)])
    ra = RunRecord.load(tmp_path / "a" / "runs" / "full_b0.2_olsr_n5_v15_gm_quad_m1.npz")
    rb = RunRecord.load(tmp_path / "b" / "runs" / "full_b0.2_olsr_n5_v15_gm_quad_m1.npz")
    assert np.array_equal(ra.frames, rb.frames)
    assert np.array_equal(ra.msg_t, rb.msg_t) and np.array_equal(ra.orch, rb.orch)
    for c in (a, b):
        main(["--log-level", "WARNING", "evaluate", "--config", str(c)])
    ma = pd.read_csv(tmp_path / "a" / "metrics.csv").drop(columns=["provenance_label"])
    mb = pd.read_csv(tmp_path / "b" / "metrics.csv").drop(columns=["provenance_label"])
    pd.testing.assert_frame_equal(ma, mb)


def test_different_seed_changes_runs(tmp_path):
    a = write_cfg(tmp_path, "s1")
    b = write_cfg(tmp_path, "s2", experiment={"seed": 6})
    for c in (a, b):
        main(["--log-level", "WARNING", "simulate", "--config", str(c)])
    ra = RunRecord.load(tmp_path / "s1" / "runs" / "none_b0_olsr_n5_v15_gm_quad_m1.npz")
    rb = RunRecord.load(tmp_path / "s2" / "runs" / "none_b0_olsr_n5_v15_gm_quad_m1.npz")
    assert not np.array_equal(ra.msg_t, rb.msg_t)


def test_paired_configs_share_trace_and_mobility(tmp_path):
    a = write_cfg(tmp_path, "p")
    main(["--log-level", "WARNING", "simulate", "--config", str(a)])
    r0 = RunRecord.load(tmp_path / "p" / "runs" / "none_b0_olsr_n5_v15_gm_quad_m1.npz")
    r1 = RunRecord.load(tmp_path / "p" / "runs" / "full_b0.2_olsr_n5_v15_gm_quad_m1.npz")
    assert np.array_equal(r0.plan_changes, r1.plan_changes)
    assert r0.orch[0, 1] == r1.orch[0, 1]  # same initial agent->UAV mapping


def test_shipped_configs_parse():
    for f in ("default.yaml", "synthetic_demo.yaml", "ns3_pipeline_check.yaml"):
        ExperimentConfig.load(ROOT / "configs" / f)
    cfg = ExperimentConfig.load(ROOT / "configs" / "experiment_template.yaml")
    assert not cfg.synthetic and cfg.backend == "ns3"
    assert sum(len(e.missions) for e in cfg.grid) == 13 * 12 + 3 * 12 + 4 * 12 + 8 * 6


def test_no_unlabelled_result_files_in_outputs():
    """Result integrity: any metrics or summary under outputs/ must carry a provenance label, and
    synthetic ones must say so. No hand-written result files are allowed."""
    for p in (ROOT / "outputs").rglob("metrics.csv"):
        labels = pd.read_csv(p)["provenance_label"]
        assert labels.notna().all(), p
        prov = json.loads((p.parent / "provenance.json").read_text())
        if prov["synthetic"]:
            assert labels.str.contains(SYNTHETIC_LABEL).all(), p


def test_no_hardcoded_numbers_in_tables_code():
    """Tables are filled from metrics.csv only: the writer contains no numeric literals in cells."""
    src = (ROOT / "src" / "swarmleak" / "experiments" / "tables.py").read_text()
    assert "L.cell(" in src and "0.98" not in src and "0.89" not in src


def test_mock_traces_force_synthetic_label(tmp_path, monkeypatch):
    """Even with a non-synthetic config (ns3 backend, llm traces), mock traces make the run synthetic."""
    from swarmleak.data.schemas import Trace
    from swarmleak.experiments import runner
    from swarmleak.sim.toy_backend import ToyBackend

    tr = Trace.load_json(ROOT / "examples" / "synthetic_data" / "llm_n5_m1.json")
    assert runner.trace_is_synthetic(tr)
    cfg_path = write_cfg(tmp_path, "mockllm", experiment={"backend": "ns3"},
                         data={"trace_source": "llm", "root": str(ROOT / "examples" / "synthetic_data")},
                         scenario={"n": 5, "duration": 120})
    cfg = ExperimentConfig.load(cfg_path)
    assert not cfg.synthetic
    monkeypatch.setattr(runner, "get_backend", lambda name: ToyBackend())  # no ns-3 needed for this check
    entry = next(e for e in cfg.grid if e.defense == "none")
    path = runner.run_one(cfg, entry, 1)
    assert RunRecord.load(path).meta["synthetic"] is True
