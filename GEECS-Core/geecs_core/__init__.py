"""geecs-core — the GEECS access library.

The shared foundation under every GEECS consumer: the UDP/TCP wire protocol
(``transport/``), the experiment MySQL database (``db/``), the PV naming
contract (``pv_naming``), the one exception tree (``exceptions``), and the
entry-level synchronous ``GeecsDevice`` client (``client/``).
See ``DESIGN.md`` for the layering rules.

The public face re-exports the exception tree eagerly (stdlib-only) and the
heavier entry points lazily, so ``import geecs_core.transport`` never drags
pydantic or mysql-connector along.
"""

from __future__ import annotations

from typing import Any

from geecs_core.exceptions import (
    GeecsCommandError,
    GeecsCommandFailedError,
    GeecsCommandRejectedError,
    GeecsConnectionError,
    GeecsDeviceNotFoundError,
    GeecsError,
)

__all__ = [
    "GeecsDevice",
    "GeecsDb",
    "GeecsError",
    "GeecsConnectionError",
    "GeecsCommandError",
    "GeecsCommandRejectedError",
    "GeecsCommandFailedError",
    "GeecsDeviceNotFoundError",
]

_LAZY = {
    "GeecsDb": ("geecs_core.db.geecs_db", "GeecsDb"),
    "GeecsDevice": ("geecs_core.client.geecs_device", "GeecsDevice"),
}


def __getattr__(name: str) -> Any:
    """Resolve the lazily exported entry points on first access."""
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)
