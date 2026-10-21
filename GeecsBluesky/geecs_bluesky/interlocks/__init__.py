"""Software interlocks: Bluesky suspenders armed from ``interlocks.yaml``.

- :mod:`.conditions` — the pure verdict (threshold, alignment, stale;
  fail-safe on disconnected / INVALID / missing data).
- :mod:`.suspender` — one interlock as a ``SuspenderBase`` over
  ophyd-async signals.
- :mod:`.install` — :func:`install_interlocks`, called by the worker's
  startup profile, and the :class:`InterlockRegistry` it returns.
- :mod:`.status_server` / :mod:`.detail` — the worker's two streams (the
  LabVIEW status text, the JSON detail the server window reads).
- :mod:`.wire` and :mod:`.gui` — the import-light client side: the
  framing and the PyQt6 windows (the ``gui`` extra).

The re-exports below are **lazy** (PEP 562), so ``import
geecs_bluesky.interlocks.gui`` on an operator's desktop never pulls
bluesky or the device stack.
"""

from __future__ import annotations

from typing import Any

_INSTALL_EXPORTS = (
    "CLEARED_LOG_PREFIX",
    "TRIPPED_LOG_PREFIX",
    "InterlockRegistry",
    "InterlockStatus",
    "install_interlocks",
)


def __getattr__(name: str) -> Any:  # Any: whichever install export is asked for
    """Serve the install module's names on first use."""
    if name in _INSTALL_EXPORTS:
        from geecs_bluesky.interlocks import install

        return getattr(install, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_INSTALL_EXPORTS)
