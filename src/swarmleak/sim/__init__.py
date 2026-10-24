"""Simulation backends that replay physical messages over a UAV MANET.

- `ns3`: the paper's backend (ns-3.44 Python bindings). Required for any paper-level result.
- `toy`: a small geometric packet replayer for tests and smoke runs ONLY. It is not a network
  simulator; nothing produced with it is a paper result.
"""
from .base import SimOutput, SimSettings, get_backend

__all__ = ["SimOutput", "SimSettings", "get_backend"]
