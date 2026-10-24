import numpy as np
import pytest

from swarmleak.data.adapters import MissingDataError, TraceDirectoryAdapter
from swarmleak.llm.agents import MockChatClient, OpenAICompatibleClient, generate_llm_trace
from swarmleak.protocol.config import ProtocolConfig, get_defense
from swarmleak.protocol.energy import RadioModel
from swarmleak.protocol.mapping import build_physical
from swarmleak.sim.base import SimSettings, get_backend
from swarmleak.sim.ns3_backend import ns3_available


def test_toy_backend_runs(small_trace, small_traj):
    res = build_physical(small_trace, small_traj, get_defense("full"), 0.2, ProtocolConfig(), RadioModel(), 1, 1)
    out = get_backend("toy").run(res.messages, small_traj, SimSettings(monitors=((300, 300, 30),)))
    assert out.synthetic_backend and out.backend == "toy"
    assert len(out.rx_time) == len(res.messages)
    assert np.nanmean(~np.isnan(out.rx_time)) > 0.3
    assert len(out.frames) > 0 and (out.frames["rx"] < 0).any()  # unicast + HELLO broadcasts
    assert (out.energy_j > 0).all()


def test_toy_backend_deterministic(small_trace, small_traj):
    res = build_physical(small_trace, small_traj, get_defense("none"), 0.0, ProtocolConfig(), RadioModel(), 1, 1)
    s = SimSettings(monitors=((300, 300, 30),), seed=5, run=2)
    a = get_backend("toy").run(res.messages, small_traj, s)
    b = get_backend("toy").run(res.messages, small_traj, s)
    assert np.array_equal(a.frames, b.frames) and np.array_equal(a.rx_time, b.rx_time, equal_nan=True)


def test_bad_backend_settings():
    with pytest.raises(ValueError):
        get_backend("omnet")
    with pytest.raises(ValueError):
        SimSettings(routing="dsr")


@pytest.mark.ns3
@pytest.mark.skipif(not ns3_available(), reason="ns-3 Python bindings not installed")
def test_ns3_backend_tiny_run(small_trace, small_traj):
    """Tiny ns-3 smoke run (60 s, 6 UAVs); skipped without ns-3."""
    import subprocess
    import sys

    code = (
        "import numpy as np\n"
        "from swarmleak.data.synthetic import surrogate_trace\n"
        "from swarmleak.mobility import gauss_markov_3d\n"
        "from swarmleak.protocol.config import ProtocolConfig, get_defense\n"
        "from swarmleak.protocol.energy import RadioModel\n"
        "from swarmleak.protocol.mapping import build_physical\n"
        "from swarmleak.sim.base import SimSettings, get_backend\n"
        "tr = surrogate_trace(6, 90.0, np.random.default_rng(1))\n"
        "tj = gauss_markov_3d(6, 90.0, np.random.default_rng(2), area=(500., 500.))\n"
        "res = build_physical(tr, tj, get_defense('none'), 0.0, ProtocolConfig(), RadioModel(), 1, 1)\n"
        "out = get_backend('ns3').run(res.messages, tj, SimSettings(monitors=((250., 250., 30.),)))\n"
        "assert len(out.frames) > 0 and np.mean(~np.isnan(out.rx_time)) > 0.2, (len(out.frames),)\n"
        "print('ok', len(out.frames))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0 and "ok" in p.stdout, p.stderr[-2000:]


def test_llm_trace_with_mock_client():
    tr = generate_llm_trace(MockChatClient(seed=3), 5, 150.0, np.random.default_rng(4))
    kinds = {m.kind for m in tr.messages}
    assert {"plan", "ack", "report"} <= kinds
    assert tr.source == "llm:mock"
    # plan envelope sizes come from A2A-style JSON: a few hundred bytes
    plan_sizes = [m.size for m in tr.messages if m.kind == "plan"]
    assert 200 < np.median(plan_sizes) < 1000
    # reports use the latest task (events processed in time order): times sorted
    assert all(a.t <= b.t for a, b in zip(tr.messages, tr.messages[1:]))


def test_llm_trace_deterministic_with_mock():
    a = generate_llm_trace(MockChatClient(seed=3), 4, 120.0, np.random.default_rng(4))
    b = generate_llm_trace(MockChatClient(seed=3), 4, 120.0, np.random.default_rng(4))
    assert a.to_dict() == b.to_dict()


def test_openai_client_makes_no_call_on_construction():
    c = OpenAICompatibleClient("http://127.0.0.1:9/v1", "none")  # nothing listens; constructing must not connect
    assert c.model == "none"


def test_trace_adapter(tmp_path, small_trace):
    ad = TraceDirectoryAdapter(tmp_path)
    with pytest.raises(MissingDataError):
        ad.load(6, 1)
    small_trace.save_json(ad.path(6, 1))
    assert ad.available() == [(6, 1)]
    assert ad.load(6, 1).n_agents == 6
    small_trace.save_json(ad.path(7, 1))  # wrong swarm size for that file name
    with pytest.raises(ValueError, match="expected 7"):
        ad.load(7, 1)


def test_external_baselines_are_stubs():
    from swarmleak.baselines.external_wrappers import (
        AnonymousRoutingDefense,
        ExternalBaselineNotAvailable,
        SequenceModelAttacker,
        TamarawDefense,
        WtfPadDefense,
    )

    for obj, call in [(SequenceModelAttacker(), lambda o: o.fit([], [])), (AnonymousRoutingDefense(), lambda o: o.run([])),
                      (TamarawDefense(), lambda o: o.run([])), (WtfPadDefense(), lambda o: o.run([]))]:
        with pytest.raises(ExternalBaselineNotAvailable):
            call(obj)


def test_dummy_baseline_flagged():
    from swarmleak.baselines.dummy import DummyRandomAttacker

    d = DummyRandomAttacker(0)
    assert d.is_dummy and d.estimate(5, 4).shape == (5,)
