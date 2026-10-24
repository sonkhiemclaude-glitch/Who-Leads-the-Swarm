"""Run metadata and provenance labels attached to every output."""
from __future__ import annotations

import datetime as _dt
import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .. import SYNTHETIC_LABEL, __version__


def source_digest() -> str:
    """Fingerprint source bytes even in a source archive without a Git checkout."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def trace_generation_digest() -> str:
    """Scope trace-cache identity to generator code; simulator fixes do not invalidate traces."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for name in ("llm/agents.py", "data/synthetic.py", "data/schemas.py", "constants.py", "utils/random.py"):
        h.update(name.encode())
        h.update((root / name).read_bytes())
    return h.hexdigest()


def code_version() -> str:
    """Package version plus git commit if the repository is a git checkout (no network)."""
    here = Path(__file__).resolve()
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here.parent, capture_output=True,
                             text=True, timeout=5, check=False)
        rev = out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        rev = ""
    return f"{__version__}+{rev}" if rev else __version__


def environment() -> dict[str, Any]:
    import numpy
    import scipy
    import sklearn

    env = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": numpy.__version__,
           "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    try:
        from importlib.metadata import version

        env["ns3"] = version("ns3")
    except Exception:  # noqa: BLE001 - optional dependency
        env["ns3"] = None
    return env


def provenance_label(synthetic: bool, backend: str, trace_source: str, extra: str = "") -> str:
    if synthetic:
        return f"{SYNTHETIC_LABEL} (backend={backend}, traces={trace_source})"
    return f"simulation output (backend={backend}, traces={trace_source}{', ' + extra if extra else ''})"


def run_metadata(cfg_hash: str, synthetic: bool, backend: str, trace_source: str, **kw: Any) -> dict[str, Any]:
    return {"created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "code_version": code_version(), "config_hash": cfg_hash, "synthetic": bool(synthetic),
            "backend": backend, "trace_source": trace_source,
            "provenance_label": provenance_label(synthetic, backend, trace_source), **kw}
