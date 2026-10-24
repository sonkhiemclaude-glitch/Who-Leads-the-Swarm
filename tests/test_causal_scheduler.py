from unittest.mock import patch

import numpy as np
import pytest

from swarmleak.data.schemas import Message, Phase, Trace
from swarmleak.data.synthetic import surrogate_trace
from swarmleak.llm.agents import MockChatClient, generate_llm_trace
from swarmleak.mobility import gauss_markov_3d
from swarmleak.protocol.config import ProtocolConfig, get_defense
from swarmleak.protocol.energy import RadioModel
from swarmleak.protocol.mapping import build_physical


def test_sparse_rotation_executes_before_later_plan_and_labels_match_mapping():
    tr = Trace([Message(t, 0, 1, 900, "plan") for t in (0.1, 10.1, 14.02, 14.07, 40.1, 70.1)],
               [Phase(0, 90, "search")], [0, 10, 40, 70], 90, 4, "surrogate").validate()
    result = build_physical(tr, np.zeros((91, 4, 3)), get_defense("rot"), 0,
                            ProtocolConfig(rotation_delay_s=(4, 4)), RadioModel(), 2026, 1)
    assert np.allclose([t for t, _ in result.orch[1:]], [14.05, 44.05, 74.05])
    for msg in result.messages:
        if msg.kind == "plan":
            host = [h for t, h in result.orch if t <= msg.t][-1]
            assert msg.src == host


def test_pending_handover_is_replaced_only_before_due():
    tr = Trace([Message(t + 0.1, 0, 1, 900, "plan") for t in (0, 10, 12, 40)],
               [Phase(0, 60, "search")], [0, 10, 12, 40], 60, 4, "surrogate").validate()
    result = build_physical(tr, np.zeros((61, 4, 3)), get_defense("rot"), 0,
                            ProtocolConfig(rotation_delay_s=(4, 4)), RadioModel(), 2026, 1)
    assert np.allclose([t for t, _ in result.orch[1:]], [16.05, 44.05])


@pytest.mark.parametrize("seed,capacity,rate", [(15, 25, 3), (1, 1, 0.1), (3, 2, 1)])
def test_cover_emissions_obey_actual_chronological_bucket_and_ledger(seed, capacity, rate):
    tr = surrogate_trace(20, 600, np.random.default_rng(seed))
    tj = gauss_markov_3d(20, 600, np.random.default_rng(seed + 100))
    radio = RadioModel()
    cfg = ProtocolConfig(bucket_capacity=capacity, bucket_rate_hz=rate)
    result = build_physical(tr, tj, get_defense("full"), 1, cfg, radio, seed, 1)
    tokens, last = np.full(20, float(capacity)), np.zeros(20)
    ledger = np.zeros(20)
    links = {}
    for msg in sorted((m for m in result.messages if not m.real), key=lambda m: m.t):
        i = msg.src
        tokens[i] = min(capacity, tokens[i] + (msg.t - last[i]) * rate)
        last[i] = msg.t
        assert tokens[i] >= 1 - 1e-9
        tokens[i] -= 1
        key = (msg.src, msg.dst)
        assert msg.t - links.get(key, -1e9) >= cfg.link_gap_s - 1e-9
        links[key] = msg.t
        ledger[msg.src] += radio.tx_cost(msg.size - radio.security_overhead_b)
        ledger[msg.dst] += radio.rx_cost(msg.size - radio.security_overhead_b)
        assert 0 <= msg.t < tr.duration
    assert np.allclose(ledger, result.cover.ledger.total)
    assert np.max(ledger) <= 1 + 1e-12


def test_slow_plan_does_not_activate_task_before_its_message():
    class SlowClient(MockChatClient):
        def __call__(self, system, user="Respond now."):
            text, _ = super().__call__(system, user)
            return text, 100.0 if "orchestrator agent" in system else 0.0

    profile = {"report_period": {"search": 10.0}, "replan_rate": {"search": 0.0}}
    with patch("swarmleak.llm.agents.mission_phases", return_value=[Phase(0, 180, "search")]), \
            patch("swarmleak.llm.agents.mission_profile", return_value=profile):
        tr = generate_llm_trace(SlowClient(), 2, 180, np.random.default_rng(1), latency_scale=1)
    plans = [m.t for m in tr.messages if m.kind == "plan"]
    assert plans and plans[0] > 100
    seen_early = False
    for report in tr.generator["report_tasks"]:
        activations = [a for a in tr.generator["task_activations"]
                       if a["worker"] == report["worker"] and a["t"] <= report["trigger_t"]]
        expected = activations[-1]["task"] if activations else "hold position"
        assert report["task"] == expected
        seen_early |= report["trigger_t"] < plans[0]
    assert seen_early


def test_truncated_or_incomplete_plans_cannot_silently_generate_real_trace():
    class IncompleteClient:
        model = "invalid"

        def __call__(self, system, user="Respond now."):
            return '{"tasks":{}}', 0.0

    with pytest.raises(ValueError, match="incomplete task map"):
        generate_llm_trace(IncompleteClient(), 4, 120, np.random.default_rng(1))
