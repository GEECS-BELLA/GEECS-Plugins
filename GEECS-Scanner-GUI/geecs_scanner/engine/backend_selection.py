"""Scan-execution backend selection (legacy ScanManager vs Bluesky).

Lives in ``engine`` (PyQt5-free) so it is importable and testable without
the GUI; ``geecs_scanner.app.run_control`` is the consumer.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("", "0", "false", "no", "off")


def resolve_use_bluesky(
    use_bluesky: Optional[bool], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Resolve the backend choice, with a ``GEECS_USE_BLUESKY`` env override.

    Precedence: an explicit ``use_bluesky`` argument > the
    ``GEECS_USE_BLUESKY`` env var (``1/true/yes/on`` → Bluesky,
    ``0/false/no/off`` → legacy; quick switching without touching source,
    mirroring ``GEECS_BLUESKY_ACQUISITION_MODE``) > default legacy.  The GUI
    constructs ``RunControl`` without the argument, so the env var is the
    supported way to switch a GUI session's backend during the
    legacy → Bluesky transition.

    Parameters
    ----------
    use_bluesky : bool, optional
        Explicit caller choice; ``None`` defers to the environment.
    env : Mapping[str, str], optional
        Environment mapping, injectable for testing (defaults to
        ``os.environ``).

    Returns
    -------
    bool
        True to use the Bluesky backend.

    Raises
    ------
    ValueError
        On an unrecognised env value — raising beats silently choosing a
        backend.
    """
    if use_bluesky is not None:
        return use_bluesky
    env = os.environ if env is None else env
    raw = env.get("GEECS_USE_BLUESKY")
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"Unrecognised GEECS_USE_BLUESKY value {raw!r}; expected one of "
        "1/true/yes/on or 0/false/no/off"
    )
