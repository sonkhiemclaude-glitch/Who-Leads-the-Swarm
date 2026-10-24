# Who Leads the Swarm

Code and data for **Who Leads the Swarm? Measuring Traffic-Pattern Leakage and Role-Rotation Defenses in LLM-Agent UAV Ad Hoc Networks**.

This repository measures passive traffic-pattern leakage in a simulated UAV ad hoc network driven by recorded LLM-agent coordination. It includes the completed evaluation data and reproducibility tools. It does not include the manuscript or PDF.

## Contents

- `src/swarmleak/`: trace collection, role rotation, cover traffic, ns-3 replay, attackers and evaluation.
- `data/primary_qwen25_7b_v3/`: 24 original Qwen2.5:7B traces and model-call logs. Files with `.failed-` suffixes are retained interrupted attempts, excluded from the completed evaluation.
- `data/size_control_20261022/`: 12 derived fixed-size replay controls, not additional LLM collections.
- `outputs/paper_primary_qwen25_7b_v3/`: 288 primary runs, metrics, comparisons and provenance.
- `outputs/paper_supplements_20261022/`: 44 supplementary runs, history-augmented attacker results, energy calibration and confirmation results.
- `configs/`: exact primary and supplementary configurations.
- `tests/`: regression tests; `validation/`: independently recomputed statistical checks.
- `docs/`: input formats and reproduction instructions.

## Installation

Python 3.11 or newer is required. Full ns-3 replay was evaluated on Linux x86_64 using ns3 3.44.post0. Analysis of saved outputs does not need ns-3 or a running LLM.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
# For actual network simulation on a supported Linux environment:
python -m pip install -e '.[sim]'
```

The recorded library environment is under `logs/requirements-linux-amd64.txt`; machine-local editable paths in this historical record are not installation requirements. Use the versions recorded in provenance when reproducing exact statistical values.

## Reproduce saved-output analysis

Run from the repository root:

```sh
python -m swarmleak.cli evaluate --config configs/mac_studio_structured_v3.yaml
python scripts/run_experiment.py --config configs/mac_studio_structured_v3.yaml --stage tables
python -m swarmleak.cli evaluate --config configs/supp_size_control.yaml
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/evaluate_paper_supplements.py
python scripts/verify_paper_results.py
python scripts/verify_supplement_statistics.py
```

Saved simulations are reused after fingerprint checks. To independently rerun simulations, copy a configuration and change its output directory before invoking `simulate`; retain the saved trace inputs and seeds. See [the full reproduction guide](docs/supplement_reproduction.txt). New trace collection requires a local Ollama-compatible endpoint and Qwen2.5:7B; replay of the included traces does not. Recollection is stochastic and need not reproduce identical traces.

## Interpretation

There are 24 original LLM traces and 332 network simulations, not 332 independent LLM collections. The default evaluation uses mission-disjoint folds. Rotation accounts for most of the observed reduction in role accuracy at a future action horizon. Fixed-size replay preserves high role leakage but reduces plan-change inference. The held-out energy calibration misses its 10% matching tolerance; equal-energy superiority is not established. The alternative history attacker exposes residual phase and plan information.

The experiment is an open-loop network replay with one model and one latency scale, not a physical UAV flight or closed-loop controller. Different horizons use independently trained attackers. Fixed-size replay changes logical datagram sizes and can change fragmentation and delivered-frame timing; it is not payload-preserving padding. The prior top-three random selector does not by itself bound posterior successor predictability after observing traffic; see the theory erratum in the reproduction guide.

## Tests

```sh
python -m pytest
```

The ns-3-dependent test is skipped when bindings are unavailable. Synthetic fixtures and mock clients are test utilities, not the reported real-trace results. Credentials, machine connection settings, and model weights are not included. No reuse licence is granted by publication of this repository.
