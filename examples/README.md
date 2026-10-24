# Examples

`synthetic_data/` holds two small traces in the trace schema (docs/data_format.md):

- `surrogate_n5_m1.json`: parametric surrogate workload (5 agents, 120 s).
- `llm_n5_m1.json`: generated with the offline `MockChatClient`, i.e. the real-LLM code path without a model.

Both are **synthetic examples**. They exist to show the format and to exercise the adapters; they are not LLM
traffic and nothing derived from them is a paper result.

`example_config.yaml` loads `llm_n5_m1.json` through the real-trace adapter (`trace_source: llm`) and replays it
with the toy backend. The runner inspects each trace: traces from the surrogate or the mock client (source
`llm:mock`, `generator.mock`, or a SYNTHETIC note) make every run and table synthetic-labelled, even with
`backend: ns3`. Real traces must come from a served model (docs/mac_studio_experiment_guide.md).

```bash
python -m swarmleak.cli simulate --config examples/example_config.yaml
```
(One mission cannot be cross-validated; `evaluate` needs at least two missions per configuration.)
