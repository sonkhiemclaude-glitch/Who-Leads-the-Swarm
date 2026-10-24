# Data formats

No external dataset or benchmark is used. All inputs are generated: LLM-agent traces (on the Mac Studio, from a served model) or synthetic traces (tests).

## 1. Trace JSON (one mission)

Path: `<data.root>/<data.pattern>`, default `data/traces/llm_n{n}_m{mission}.json` (N agents, mission id). Produced by `swarmleak gen-traces`; any generator writing this schema works.

```json
{
  "messages": [{"t": 0.31, "src": 0, "dst": 4, "size": 412, "kind": "plan"}, ...],
  "phases": [[0.0, 97.2, "search"], [97.2, 180.4, "track"], ..., [560.0, 600.0, "return"]],
  "plan_changes": [0.29, 97.6, ...],
  "duration": 600.0,
  "n_agents": 20,
  "source": "llm:<model tag>",
  "profile": {"report_period": {"search": 11.2, ...}, "replan_rate": {"search": 0.0091, ...}},
  "generator": {"model": "<model tag>", "latency_scale": 0.2, "report_llm_prob": 1.0, "mission": 1}
}
```

| Field | Rule |
|---|---|
| `messages[].t` | seconds from mission start, ≥ 0, sorted ascending, < `duration` |
| `messages[].src/dst` | agent ids in [0, n_agents), src ≠ dst; **agent 0 is the orchestrator** |
| `messages[].size` | UTF-8 bytes of the A2A-style envelope (> 0); the simulator adds 29 B security overhead |
| `messages[].kind` | `plan`, `ack`, `report`, `feedback`, `tool_call` (peer query), `tool_result` (peer reply) |
| `phases` | contiguous `[start, end, name]`, names in {search, track, regroup, return}; last ends at `duration` |
| `plan_changes` | sorted plan-change instants (ground truth for Definition 3) |
| `generator.mock` / `source = "llm:mock"` / `"surrogate"` | marks synthetic traces; runs made from them are labelled synthetic |

The scenario's `duration` and `n` must match the trace (checked).

## 2. Run file (`<output_dir>/runs/<tag>.npz`)

Tag: `{defense}_b{budget}_{routing}_n{N}_v{speed}_{mobility}_{monitors}_m{mission}`.

| Array | Content |
|---|---|
| `frames` | structured, dtype `FRAME_DTYPE`: `t` (s, mission clock), `size` (B, MAC frame), `tx`, `rx` (UAV index; `rx = -1` broadcast/unknown), `seq`, `retry`, `mon` (monitor index), `rssi` (dBm); de-duplicated across monitors on load |
| `rx_time` | delivery time per physical message, NaN if lost |
| `energy_j` | radio energy per UAV (J), including idle |
| `msg_t, msg_src, msg_dst, msg_size, msg_real, msg_gid, msg_kind` | physical messages (UAV ids; `real = False` for cover; `gid` groups padding chunks; size includes the 29-B overhead) |
| `orch` | `[[t_start, uav], ...]` orchestrator host schedule (ground truth for role) |
| `plan_changes`, `phases` | ground truth copied from the trace |
| `meta` | JSON: config hash, code version, backend, trace source/model, synthetic flag, provenance label, scenario, defense, budget, mission, rotations, cover statistics |

## 3. Evaluation outputs

| File | Rows / columns |
|---|---|
| `per_mission.csv` | one row per run: config columns, leakage metrics (`role_acc_h{Δ}`, `role_acc_heur`, `role_chance`, `phase_acc`, `phase_majority`, `plan_f1`, `plan_prec`, `plan_rec`, `p_E_h{Δ}`, `p_two_plus_h{Δ}`, `rotation_mean_interval_s`), network metrics, `dE_j_mean`, `provenance_label` |
| `metrics.csv` | one row per configuration: mean and `_ci` (95% Student-t half-width) of every numeric metric, `missions`, `provenance_label` |
| `comparisons.csv` | paired bootstrap: `a`, `b`, `metric`, `mean_diff`, `lo`, `hi`, `n` |
| `provenance.json` | experiment, config hash, synthetic flag, label, environment versions |

## 4. Splits

Mission-grouped cross-validation (`attack.folds`, default 4): a mission's runs are never in both training and test data. Each configuration is attacked by an adversary trained on that configuration (adaptive adversary).
