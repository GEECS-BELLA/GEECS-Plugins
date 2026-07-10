"""Scan-execution backend selection — Bluesky is the only backend.

The legacy ScanManager backend was deleted in G1 of the greenfield cutover
(see ``Planning/cutover_strategy/00_overview.md``), so there is no longer a
choice to resolve. This module is retained so the import surface (and any
external callers of :func:`resolve_use_bluesky`) stays valid.
"""

from __future__ import annotations

from typing import Mapping, Optional


def resolve_use_bluesky(
    use_bluesky: Optional[bool] = None, env: Optional[Mapping[str, str]] = None
) -> bool:
    """Return True: BlueskyScanner is the only scan backend.

    The legacy ScanManager backend was removed (G1); the ``use_bluesky``
    argument and the ``GEECS_USE_BLUESKY`` env var are ignored.

    Parameters
    ----------
    use_bluesky : bool, optional
        Ignored — kept for signature compatibility.
    env : Mapping[str, str], optional
        Ignored — kept for signature compatibility.

    Returns
    -------
    bool
        Always True.
    """
    return True
