import importlib

MODULES = [
    "swarmleak", "swarmleak.cli", "swarmleak.constants",
    "swarmleak.data.schemas", "swarmleak.data.synthetic", "swarmleak.data.adapters",
    "swarmleak.llm.agents", "swarmleak.mobility", "swarmleak.protocol.energy", "swarmleak.protocol.config",
    "swarmleak.protocol.rotation", "swarmleak.protocol.cover", "swarmleak.protocol.mapping",
    "swarmleak.protocol.size_memory", "swarmleak.sim.base", "swarmleak.sim.toy_backend", "swarmleak.sim.ns3_backend",
    "swarmleak.attack.features", "swarmleak.attack.datasets", "swarmleak.attack.classifiers",
    "swarmleak.attack.evaluate", "swarmleak.metrics.leakage", "swarmleak.metrics.network",
    "swarmleak.metrics.confidence_intervals", "swarmleak.metrics.rotation_stats", "swarmleak.theory.horizon_bound",
    "swarmleak.baselines.registry", "swarmleak.baselines.dummy", "swarmleak.baselines.external_wrappers",
    "swarmleak.experiments.config", "swarmleak.experiments.runner", "swarmleak.experiments.evaluation",
    "swarmleak.experiments.tables", "swarmleak.experiments.plots", "swarmleak.experiments.provenance",
]


def test_all_modules_import():
    for m in MODULES:
        importlib.import_module(m)


def test_ns3_module_imports_without_ns3():
    # importing the backend must not require ns-3; only Ns3Backend.run does
    mod = importlib.import_module("swarmleak.sim.ns3_backend")
    assert hasattr(mod, "ns3_available")
