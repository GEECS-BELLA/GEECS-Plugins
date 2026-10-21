"""Software interlocks: Bluesky suspenders armed from ``interlocks.yaml``.

- :mod:`.conditions` — the pure verdict (threshold, alignment, stale;
  fail-safe on disconnected / INVALID / missing data).
- :mod:`.suspender` — one interlock as a ``SuspenderBase`` over
  ophyd-async signals.
- :mod:`.install` — :func:`install_interlocks`, called by the worker's
  startup profile, and the :class:`InterlockRegistry` it returns.
"""

from geecs_bluesky.interlocks.install import (
    CLEARED_LOG_PREFIX,
    TRIPPED_LOG_PREFIX,
    InterlockRegistry,
    InterlockStatus,
    install_interlocks,
)

__all__ = [
    "CLEARED_LOG_PREFIX",
    "TRIPPED_LOG_PREFIX",
    "InterlockRegistry",
    "InterlockStatus",
    "install_interlocks",
]
