import numpy as np
import pytest

from swarmleak.protocol.config import DEFENSES, ProtocolConfig, get_defense
from swarmleak.protocol.cover import TokenBucket
from swarmleak.protocol.energy import RadioModel
from swarmleak.protocol.mapping import build_physical, pad_messages
from swarmleak.protocol.rotation import RotationPolicy, successor_scores
from swarmleak.protocol.size_memory import SizeMemory

CFG = ProtocolConfig()
RADIO = RadioModel()


def run(defense, trace, traj, budget=0.2, seed=3, mission=1):
    return build_physical(trace, traj, get_defense(defense), budget, CFG, RADIO, seed, mission)


def test_no_defense_is_one_to_one(small_trace, small_traj):
    res = run("none", small_trace, small_traj, budget=0)
    assert len(res.messages) == len(small_trace.messages)
    assert all(m.real for m in res.messages)
    assert len(res.orch) == 1
    # every datagram carries the 29-byte security overhead
    sizes_in = sorted(m.size for m in small_trace.messages)
    sizes_out = sorted(m.size - 29 for m in res.messages)
    assert sizes_in == sizes_out


def test_rotation_skips_mission_start_and_uses_two_transfers(small_trace, small_traj):
    res = run("rot", small_trace, small_traj, budget=0)
    kinds = [m.kind for m in res.messages]
    n_rot = len(res.orch) - 1
    assert n_rot >= 1
    assert kinds.count("handover_orch") == n_rot and kinds.count("handover_worker") == n_rot
    assert all(t > small_trace.plan_changes[0] for t, _ in res.orch[1:])
    assert all(b[1] != a[1] for a, b in zip(res.orch, res.orch[1:]))
    # handover happens 4-20 s after some plan change
    for t, _ in res.orch[1:]:
        gaps = [t - 0.05 - pc for pc in small_trace.plan_changes if pc < t]
        assert any(4.0 - 1e-9 <= g <= 20.0 + 1e-9 for g in gaps)


def test_successor_top3_and_score():
    adj = np.zeros((5, 5), bool)
    adj[1, :] = adj[:, 1] = True
    np.fill_diagonal(adj, False)
    sc = successor_scores([1, 2, 3, 4], adj.sum(1), np.array([0, 0, 0, 0, 100.0]))
    assert sc[0] == pytest.approx(1.0) and sc[3] < sc[1]
    pol = RotationPolicy(top_k=3)
    picks = {pol.choose_successor(0, adj, np.array([0, 0, 0, 0, 100.0]), np.random.default_rng(s)) for s in range(200)}
    assert picks <= {1, 2, 3} and len(picks) == 3  # uniform over the top 3, never the worst candidate


def test_budget_feasibility_and_rate_limits(small_trace, small_traj):
    for name in ("cover_uniform", "cover_targeted", "full", "const"):
        res = run(name, small_trace, small_traj, budget=0.2)
        led = res.cover.ledger
        assert (led.total <= 0.2 + 1e-12).all(), name
        for stage, cap in led.caps.items():
            assert (led.spent[stage] <= cap + 1e-12).all()
        cov = sorted((m for m in res.messages if not m.real), key=lambda m: m.t)
        last = {}
        for m in cov:  # 0.5 s per directed link
            if (m.src, m.dst) in last:
                assert m.t - last[(m.src, m.dst)] >= 0.5 - 1e-9
            last[(m.src, m.dst)] = m.t


def test_estimated_cover_energy_recomputes_from_messages(small_trace, small_traj):
    res = run("full", small_trace, small_traj, budget=0.2)
    est = np.zeros(small_trace.n_agents)
    for m in res.messages:
        if not m.real:
            payload = m.size - 29
            est[m.src] += RADIO.tx_cost(payload)
            est[m.dst] += RADIO.rx_cost(payload)
    assert np.allclose(est, res.cover.ledger.total)


def test_decoys_fire_after_their_plan(small_trace, small_traj):
    res = run("cover_targeted", small_trace, small_traj, budget=1.0)
    plans = [(m.t, m.dst) for m in res.messages if m.kind == "plan"]
    for t, j in res.decoy_starts:
        assert any(pt + 0.05 + 0.3 - 1e-9 <= t <= pt + 0.05 + 2.5 + 1e-9 and pd == j for pt, pd in plans)


def test_cover_is_one_hop_unless_anyhop(small_trace, small_traj):
    from swarmleak.mobility.topology import adjacency_at

    res = run("cover_uniform", small_trace, small_traj, budget=1.0)
    cov = [m for m in res.messages if m.kind == "cover_plan"]
    assert cov
    for m in cov:
        # the destination was a one-hop neighbour (R-hat = 250 m) when the burst started (< 2 s earlier)
        assert any(adjacency_at(small_traj, m.t - dt, 250)[m.src, m.dst] for dt in (0.0, 1.0, 2.0))
    any_res = run("full_anyhop", small_trace, small_traj, budget=1.0)
    assert any(not m.real for m in any_res.messages)


def test_zero_budget_means_no_cover(small_trace, small_traj):
    res = run("cover_targeted", small_trace, small_traj, budget=0.0)
    assert not any(not m.real for m in res.messages)


def test_padding_chunks():
    from swarmleak.data.schemas import PhysicalMessage

    out = pad_messages([PhysicalMessage(1.0, 0, 1, 2500, "plan", True, 7)], 1024)
    assert [m.size for m in out] == [1024, 1024, 1024] and {m.gid for m in out} == {7}


def test_token_bucket_monotone_refill():
    b = TokenBucket(2, 1.0)
    assert b.available(0.0) == 2
    b.take()
    b.take()
    assert b.available(5.0) == 2  # capped
    b.take()
    assert b.available(3.0) == 1  # earlier time never removes tokens (audit M11)


def test_size_memory_uses_only_past():
    sm = SizeMemory(2, {"plan": 900, "ack": 260, "report": 600}, min_samples=2)
    sm.see(0, "plan", 111, 1.0)
    sm.see(0, "plan", 111, 2.0)
    sm.see(0, "plan", 999, 10.0)
    rng = np.random.default_rng(0)
    assert {sm.draw(0, "plan", rng, 5.0) for _ in range(20)} == {111}


def test_registry_complete():
    from swarmleak.baselines.registry import undocumented_defenses

    assert undocumented_defenses() == set()
    assert {"none", "pad", "rot", "cover_uniform", "cover_targeted", "full", "const", "full_anyhop"} == set(DEFENSES)


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        ProtocolConfig(targeted_shares={"cotimed": 0.5, "spread": 0.5, "mirror": 0.5})
    with pytest.raises(ValueError):
        get_defense("nope")
