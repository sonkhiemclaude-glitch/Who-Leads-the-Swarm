"""Coordination traces from real LLM agents (Section 4.1).

An orchestrator agent and N-1 worker agents run through a scripted mission. Mission timing is
scripted (`data.synthetic.mission_phases` / `mission_profile`); the LLMs decide message content,
hence sizes, inference latency, and peer queries. Every inter-agent message is wrapped in an
A2A-style JSON-RPC 2.0 envelope (field names follow pre-1.0 A2A drafts; only the byte length is
used). A message produced by a model call is timestamped at event time + latency * lambda.

Network access happens only inside `OpenAICompatibleClient.__call__`, i.e. only when a user runs
trace generation against an endpoint they configured. Tests use `MockChatClient`.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
import uuid
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..data.schemas import Message, Trace, phase_at
from ..data.synthetic import mission_phases, mission_profile

log = logging.getLogger(__name__)

SYS_ORCH = ("You are the orchestrator agent of a {n}-UAV swarm. Mission phase: {phase}. "
            "Event: {event}. Assign one short task to each worker id in {workers}. "
            "Reply with JSON only: {{\"plan_id\": str, \"tasks\": {{\"<id>\": \"<task>\"}}}}.")
SYS_WORK = ("You are worker UAV {wid} in phase {phase}. Your task: {task}. Write a status report "
            "to the orchestrator as JSON only: {{\"status\": str, \"observations\": str, \"need_peer\": bool}}.")
SYS_PEER = "You are UAV {pid}. Peer {wid} asks for your local sensor summary in phase {phase}. Reply as JSON only."
REPLAN_EVENTS = ("target lost", "new contact", "link degradation", "battery warning")


class ChatClient(Protocol):
    model: str

    def __call__(self, system: str, user: str = "Respond now.") -> tuple[str, float]:
        """Return (completion text, latency in seconds)."""


@dataclass
class OpenAICompatibleClient:
    """Chat-completions client for an OpenAI-compatible server (Ollama, vLLM, llama.cpp)."""

    endpoint: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 400
    timeout_s: float = 120.0
    audit_path: str | None = None
    structured_outputs: bool = False

    def __call__(self, system: str, user: str = "Respond now.") -> tuple[str, float]:
        url = self.endpoint.rstrip("/") + "/chat/completions"
        response_format: dict[str, Any] = {"type": "json_object"}
        schema = None
        if self.structured_outputs:
            schema = response_schema(system)
            response_format = {"type": "json_schema", "json_schema": {
                "name": "swarm_message", "strict": True, "schema": schema}}
        body = json.dumps({"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                           "response_format": response_format,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:  # noqa: S310 (user-configured URL)
            out = json.loads(r.read())
        latency = time.perf_counter() - t0
        choice = out["choices"][0]
        text = str(choice["message"]["content"])
        if self.audit_path:
            path = Path(self.audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"model": self.model, "system": system, "user": user,
                                     "content": text, "latency_s": latency, "temperature": self.temperature,
                                     "max_tokens": self.max_tokens, "finish_reason": choice.get("finish_reason"),
                                     "response_format": response_format,
                                     "usage": out.get("usage")}) + "\n")
        if choice.get("finish_reason") == "length":
            raise ValueError("LLM completion was truncated; increase the predeclared max_tokens before collection")
        if schema is not None:
            validate_response(json.loads(text), schema)
        return text, latency


def response_schema(system: str) -> dict[str, Any]:
    """Constrain each existing agent role without changing its prompt or sampling settings."""
    def obj(properties: dict[str, Any]) -> dict[str, Any]:
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    string = {"type": "string", "minLength": 1}
    if "orchestrator agent" in system:
        match = re.search(r"of a (\d+)-UAV swarm", system)
        if match is None:
            raise ValueError("cannot determine orchestrator worker count")
        return obj({"plan_id": string, "tasks": obj({str(i): string for i in range(1, int(match[1]))})})
    if "worker UAV" in system:
        return obj({"status": string, "observations": string, "need_peer": {"type": "boolean"}})
    if "Peer " in system and "sensor summary" in system:
        return obj({"summary": string})
    raise ValueError("unknown role for structured swarm response")


def validate_response(value: Any, schema: dict[str, Any]) -> None:
    """Check the small schema subset used above, even if a server ignores constraints."""
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict) or set(value) != set(schema["properties"]):
            raise ValueError("structured response has missing or unexpected fields")
        for key, child in schema["properties"].items():
            validate_response(value[key], child)
    elif kind == "string":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("structured response must contain nonempty strings")
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError("structured response must contain a boolean")
    else:
        raise ValueError("unsupported swarm schema type")


@dataclass
class MockChatClient:
    """Deterministic offline client for tests. NOT an LLM; outputs are synthetic."""

    seed: int = 0
    model: str = "mock"
    latency_s: float = 0.5
    peer_prob: float = 0.2

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, system: str, user: str = "Respond now.") -> tuple[str, float]:
        if "orchestrator agent" in system:
            n = int(system.split("of a ")[1].split("-UAV")[0])
            txt = json.dumps({"plan_id": uuid.UUID(int=int(self._rng.integers(2**63))).hex[:8],
                              "tasks": {str(i): f"sweep sector {i}" for i in range(1, n)}})
        elif "worker UAV" in system:
            txt = json.dumps({"status": "ok", "observations": "x" * int(self._rng.integers(20, 300)),
                              "need_peer": bool(self._rng.random() < self.peer_prob)})
        else:
            txt = json.dumps({"summary": "y" * int(self._rng.integers(50, 200))})
        return txt, self.latency_s * float(self._rng.uniform(0.5, 1.5))


def a2a_envelope(sender: int, receiver: int, text: str, rng: np.random.Generator) -> str:
    """A2A-style JSON-RPC 2.0 'message/send' envelope; ids drawn from rng for determinism."""
    def rid() -> str:
        return str(uuid.UUID(int=int(rng.integers(2**63)) << 64 | int(rng.integers(2**63))))
    return json.dumps({"jsonrpc": "2.0", "id": rid(), "method": "message/send",
                       "params": {"message": {"role": "agent", "messageId": rid(),
                                              "metadata": {"from": f"uav-{sender}", "to": f"uav-{receiver}"},
                                              "parts": [{"kind": "text", "text": text}]}}})


def _json_obj(text: str) -> dict:
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
        return obj if isinstance(obj, dict) else {}
    except (ValueError, json.JSONDecodeError):
        return {}


def generate_llm_trace(client: ChatClient, n_agents: int, duration: float, rng: np.random.Generator,
                       latency_scale: float = 0.2, report_llm_prob: float = 1.0) -> Trace:
    """Generate one mission's trace (Section 4.1). Events are processed in time order."""
    if n_agents < 2:
        raise ValueError("n_agents must be >= 2")
    if not (0.0 <= report_llm_prob <= 1.0):
        raise ValueError("report_llm_prob must be in [0, 1]")
    phases = mission_phases(rng, duration)
    prof = mission_profile(rng)
    raw: list[tuple[float, int, int, int, str]] = []
    plan_changes: list[float] = []
    tasks = {w: "hold position" for w in range(1, n_agents)}
    last_report: dict[int, str | None] = {w: None for w in range(1, n_agents)}

    def send(t: float, s: int, d: int, text: str, kind: str) -> None:
        raw.append((float(t), int(s), int(d), len(a2a_envelope(s, d, text, rng).encode("utf-8")), kind))

    if not np.isfinite(latency_scale) or latency_scale < 0:
        raise ValueError("latency_scale must be finite and nonnegative")
    serial = count()
    events: list[tuple[float, int, int, str, Any]] = []
    busy_until = np.zeros(n_agents)
    task_activations: list[dict[str, Any]] = []
    report_tasks: list[dict[str, Any]] = []
    call_count = 0

    def push(t: float, priority: int, kind: str, value: Any) -> None:
        if 0 <= t < duration:
            heappush(events, (float(t), priority, next(serial), kind, value))

    def call(system: str) -> tuple[str, float]:
        nonlocal call_count
        text, latency = client(system)
        call_count += 1
        if not np.isfinite(latency) or latency < 0:
            raise ValueError("model latency must be finite and nonnegative")
        return text, latency

    for ph in phases:
        push(ph.start, 1, "plan_request", f"enter phase {ph.name}")
        t, rate = ph.start, prof["replan_rate"][ph.name]
        while rate > 0:
            t += float(rng.exponential(1 / rate))
            if t >= ph.end - 5:
                break
            push(t, 1, "plan_request", str(rng.choice(REPLAN_EVENTS)))
        per = prof["report_period"][ph.name]
        for w in range(1, n_agents):
            tw = ph.start + float(rng.uniform(0, per))
            while tw < ph.end:
                push(tw, 2, "report_request", w)
                tw += per * float(rng.uniform(0.8, 1.2))

    while events:
        t, _, _, kind, value = heappop(events)
        phase = phase_at(phases, t)
        if kind == "activate":
            w, task = value
            tasks[w] = task
            task_activations.append({"t": t, "worker": w, "task": task})
        elif kind == "plan_request":
            if busy_until[0] > t:
                push(busy_until[0], 1, kind, value)
                continue
            text, lat = call(SYS_ORCH.format(n=n_agents, phase=phase, event=value,
                                             workers=list(range(1, n_agents))))
            tk = _json_obj(text).get("tasks")
            if not isinstance(tk, dict) or any(not isinstance(tk.get(str(w)), str) or not tk[str(w)].strip()
                                               for w in range(1, n_agents)):
                raise ValueError("orchestrator returned invalid JSON or an incomplete task map")
            ts = t + lat * latency_scale
            busy_until[0] = ts
            if ts >= duration:
                continue
            plan_changes.append(ts)
            for w in rng.permutation(np.arange(1, n_agents)):
                w = int(w)
                ts += float(rng.uniform(0.02, 0.15))
                if ts >= duration:
                    break
                task = tk[str(w)]
                send(ts, 0, w, json.dumps({"plan_event": value, "task": task}), "plan")
                # Trace-level delivery assumption: task activates when its plan is sent.
                # The network replay measures losses separately and cannot feed back into generation.
                push(ts, 0, "activate", (w, task))
                ta = ts + float(rng.uniform(0.2, 1.0))
                if ta < duration:
                    send(ta, w, 0, json.dumps({"ack": True}), "ack")
        elif kind == "report_request":
            w = int(value)
            if busy_until[w] > t:
                push(busy_until[w], 2, kind, w)
                continue
            cached = last_report[w]
            task = tasks[w]
            if cached is None or rng.random() < report_llm_prob:
                text, lat = call(SYS_WORK.format(wid=w, phase=phase, task=task))
            else:
                text, lat = cached, 0.0
            obj = _json_obj(text)
            if not obj or not isinstance(obj.get("need_peer"), bool):
                raise ValueError("worker report must be a JSON object with boolean need_peer")
            ts = t + lat * latency_scale
            busy_until[w] = ts
            report_tasks.append({"trigger_t": t, "send_t": ts, "worker": w, "task": task})
            push(ts, 0, "report_complete", (w, text))
        elif kind == "report_complete":
            w, text = value
            last_report[w] = text
            send(t, w, 0, text, "report")
            if _json_obj(text)["need_peer"] and n_agents > 2:
                peer = int(rng.choice([x for x in range(1, n_agents) if x != w]))
                push(t + 0.1, 2, "peer_request", (w, peer))
        elif kind == "peer_request":
            w, peer = value
            send(t, w, peer, json.dumps({"tool": "query_peer_sensor", "phase": phase}), "tool_call")
            push(max(t, busy_until[peer]), 2, "peer_inference", (w, peer))
        else:
            w, peer = value
            if busy_until[peer] > t:
                push(busy_until[peer], 2, kind, value)
                continue
            rep, lat = call(SYS_PEER.format(pid=peer, wid=w, phase=phase))
            if not _json_obj(rep):
                raise ValueError("peer reply is not a JSON object")
            ts = t + lat * latency_scale
            busy_until[peer] = ts
            if ts < duration:
                send(ts, peer, w, rep, "tool_result")
    raw.sort(key=lambda r: r[0])
    return Trace([Message(*r) for r in raw], phases, sorted(plan_changes), float(duration), n_agents,
                 f"llm:{client.model}", prof,
                 {"latency_scale": latency_scale, "report_llm_prob": report_llm_prob, "model": client.model,
                  "scheduler": "causal_events_v2", "model_calls": call_count,
                  "task_delivery_assumption": "plan send time; network replay does not feed back",
                  "task_activations": task_activations, "report_tasks": report_tasks}).validate()
