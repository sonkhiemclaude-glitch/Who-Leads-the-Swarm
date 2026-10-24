import numpy as np
import pytest

from swarmleak.constants import PHASES
from swarmleak.data.schemas import FRAME_DTYPE, Message, Phase, Trace, dedupe_frames, phase_at
from swarmleak.data.synthetic import mission_phases, mission_profile, surrogate_trace


def test_trace_roundtrip(tmp_path, small_trace):
    p = tmp_path / "t.json"
    small_trace.save_json(p)
    back = Trace.load_json(p)
    assert back.n_agents == small_trace.n_agents
    assert len(back.messages) == len(small_trace.messages)
    assert back.messages[5] == small_trace.messages[5]


def test_trace_validation_rejects_bad_messages():
    ph = [Phase(0, 100, "search")]
    with pytest.raises(ValueError):
        Trace([Message(1.0, 0, 0, 100, "plan")], ph, [], 100.0, 3, "x").validate()
    with pytest.raises(ValueError):
        Trace([Message(1.0, 0, 5, 100, "plan")], ph, [], 100.0, 3, "x").validate()
    with pytest.raises(ValueError):
        Trace([Message(2.0, 0, 1, 100, "plan"), Message(1.0, 0, 1, 100, "plan")], ph, [], 100.0, 3, "x").validate()
    with pytest.raises(ValueError):
        Trace([Message(1.0, 0, 1, 100, "bogus")], ph, [], 100.0, 3, "x").validate()


def test_missing_fields_rejected():
    with pytest.raises(ValueError, match="missing"):
        Trace.from_dict({"messages": []})


def test_mission_phases_contiguous_and_end_with_return():
    ph = mission_phases(np.random.default_rng(0), 600)
    assert ph[0].start == 0 and ph[-1].name == "return" and ph[-1].end == 600
    for a, b in zip(ph, ph[1:]):
        assert abs(a.end - b.start) < 1e-9
    assert all(p.name in PHASES for p in ph)


def test_profile_within_range():
    prof = mission_profile(np.random.default_rng(3))
    assert 4.0 * 0.6 <= prof["report_period"]["track"] <= 4.0 * 1.6
    assert prof["replan_rate"]["return"] == 0.0


def test_surrogate_trace_structure(small_trace):
    kinds = {m.kind for m in small_trace.messages}
    assert {"plan", "ack", "report"} <= kinds
    assert all(m.src == 0 for m in small_trace.messages if m.kind == "plan")
    assert all(m.dst == 0 for m in small_trace.messages if m.kind == "report")
    assert len(small_trace.plan_changes) >= len(small_trace.phases)


def test_surrogate_deterministic():
    a = surrogate_trace(5, 150, np.random.default_rng(9))
    b = surrogate_trace(5, 150, np.random.default_rng(9))
    assert a.to_dict() == b.to_dict()


def test_phase_at_last_phase_extends():
    ph = [Phase(0, 10, "search"), Phase(10, 20, "return")]
    assert phase_at(ph, 5) == "search" and phase_at(ph, 25) == "return"


def test_dedupe_merges_monitor_copies_but_keeps_retries():
    F = np.zeros(4, dtype=FRAME_DTYPE)
    F[0] = (1.00000, 500, 2, 3, 10, 0, 0, -60)
    F[1] = (1.000001, 500, 2, 3, 10, 0, 1, -70)  # same frame at another monitor
    F[2] = (1.0012, 500, 2, 3, 10, 1, 0, -60)  # retransmission
    F[3] = (2.0, 300, 3, 2, 11, 0, 0, -60)
    out = dedupe_frames(F)
    assert len(out) == 3
    assert out["retry"].sum() == 1
