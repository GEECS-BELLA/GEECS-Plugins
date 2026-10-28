"""A stand-in for Canopy's adapter base when Canopy is not installed (CI; this package's env).

Inside Canopy's environment the real module is used and nothing here applies.
"""

import importlib.util
import sys
import types
from dataclasses import dataclass

if importlib.util.find_spec("canopy") is None:

    @dataclass
    class AdapterHealth:
        healthy: bool
        last_check_ts_ns: int
        latency_ms: float | None = None
        last_error: str | None = None

    base = types.ModuleType("canopy.raccoon.adapters.base")
    base.AdapterHealth = AdapterHealth  # type: ignore[attr-defined]
    base.RaccoonError = type("RaccoonError", (Exception,), {})  # type: ignore[attr-defined]
    for name in ("DirectoryUnavailable", "InvalidQuery", "AdapterMisconfigured"):
        setattr(base, name, type(name, (base.RaccoonError,), {}))  # type: ignore[attr-defined]
    for name in ("canopy", "canopy.raccoon", "canopy.raccoon.adapters"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["canopy.raccoon.adapters"].base = base  # type: ignore[attr-defined]
    sys.modules["canopy.raccoon.adapters.base"] = base
