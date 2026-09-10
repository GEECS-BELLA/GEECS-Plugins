"""Cooperative per-run reset across the device mixins.

Long-lived namespace devices (GEECS-Plugins#807 phase 2) outlive the run
that configured them, so every mixin holding per-run state exposes
``reset_run_configuration``.  A device mixes in several of them, so the
implementations must chain along the MRO instead of the first one winning.
"""

from __future__ import annotations

from typing import Any


def reset_next(parent: Any) -> None:
    """Continue a cooperative ``reset_run_configuration`` up the MRO.

    ``CaGenericDetector`` mixes in more than one per-run-state carrier, so
    each ``reset_run_configuration`` must hand off rather than shadow the
    next one.  The base classes do not define it, hence the lookup.
    """
    nxt = getattr(parent, "reset_run_configuration", None)
    if nxt is not None:
        nxt()
