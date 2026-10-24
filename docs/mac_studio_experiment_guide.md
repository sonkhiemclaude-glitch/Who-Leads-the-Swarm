# Mac Studio experiment guide

Nothing in this repository downloads data, models or benchmarks. Everything below is done manually on the Mac Studio. Connection details (Tailscale/SSH) are kept in a private `.env` outside the repository and are never committed.

## 0. What has to be provisioned

| Item | Why | How | Status |
|---|---|---|---|
| Python 3.11+ | package | `python3 -m venv .venv` | none |
| ns-3.44 with Python bindings | paper backend (§4.2) | see §1 | **unverified on macOS** |
| An instruction LLM served through an OpenAI-compatible API | real traces (§4.1) | see §2 | to choose |
| Disk | runs | 0.25-3 MB per run in the pilot (AODV any-hop largest); paper grid = 288 runs | none |

No datasets, benchmarks or licences are required: the traces are generated, not downloaded.

## 1. ns-3.44 with Python bindings

PyPI publishes `ns3==3.44.post0` wheels **only for Linux x86_64** (checked 2026-10-19). On an Apple-silicon Mac Studio choose one:

1. **Build from source (preferred):** download the ns-3.44 release yourself, then
   ```bash
   ./ns3 configure --enable-python-bindings --enable-modules=core,network,internet,wifi,mobility,energy,olsr,aodv,applications --build-profile=optimized
   ./ns3 build
   ```
   Python bindings use `cppyy`; check `python -c "from ns import ns; print(ns.Simulator)"`. Add the build's `bindings/python` directory to `PYTHONPATH` as the ns-3 manual describes.
2. **Linux x86_64 machine or VM:** `pip install -e ".[dev,sim]"` works directly.
3. **Docker `linux/amd64` on Apple silicon:** works through emulation but is slow; use only for checks.

The backend uses `ns3::energy::EnergySourceContainer` (ns-3 ≥ 3.42 namespaces). Other ns-3 versions are not supported.

Verify:
```bash
python -m swarmleak.cli info                 # "ns-3 Python bindings available: True"
pytest -q -m ns3                             # tiny ns-3 replay
python -m swarmleak.cli simulate --config configs/ns3_pipeline_check.yaml
python -m swarmleak.cli evaluate --config configs/ns3_pipeline_check.yaml   # synthetic (mock traces)
```

## 2. LLM endpoint and real traces

Serve a small instruction model locally (for example with Ollama: `ollama serve`, then pull a model of your choice yourself). Record the exact model tag, quantisation, server version, hardware and date for the paper.

Set in your copy of `configs/experiment_template.yaml`:
```yaml
data: {trace_source: llm, root: data/traces}
llm: {endpoint: "http://localhost:11434/v1", model: "<model tag>", latency_scale: 0.2, report_llm_prob: 1.0}
```
Generate (resumable; existing files are skipped):
```bash
python -m swarmleak.cli gen-traces --config configs/experiment_template.yaml
```
This writes `data/traces/llm_n{N}_m{mission}.json` for N ∈ {10, 20, 30} and missions 1-12. A 20-agent, 600 s mission makes roughly 2,000 model calls; use `report_llm_prob < 1` only if inference is too slow, and report the value. Repeat a subset with a second model and with `latency_scale: 1.0` for the sensitivity analysis named in the paper.

## 3. Paper grid

```bash
tmux new -s swarm
JOBS=8 bash scripts/mac_studio_run_paper_grid.sh configs/experiment_template.yaml
```
- Runs are independent processes (one ns-3 simulator per process) and resumable.
- Memory: AODV with any-hop cover needed about 3.8 GB per run in the pilot; lower `--jobs` for those (`--only full_anyhop`).
- Wall time in the pilot (Linux x86_64, 1 core per run): 20-30 s per OLSR run, about 2 min per AODV any-hop run, plus about 10 s simulator start-up.

## 4. Outputs and the manuscript

`outputs/paper_grid/` gets `runs/`, `metrics.csv`, `per_mission.csv`, `comparisons.csv`, `summary.md`, `provenance.json`, `tables/`, `tradeoff.png`. Check `provenance.json` (`synthetic: false`, backend `ns3`, trace source `llm`) before writing tables into the manuscript:
```bash
python scripts/write_manuscript_tables.py --metrics outputs/paper_grid/metrics.csv --out ../../manuscript/tables
```
The script refuses synthetic results. Copy back to the authors' machine only `outputs/paper_grid/` (and the traces if they will be released).

## 5. What must not happen

- No automatic downloads, no hidden network calls (the only network access is the configured LLM endpoint during `gen-traces`).
- No secrets in configs or in the repository; keep `.env` outside it.
- No mixing of toy/synthetic outputs with real ones in the same output directory.
