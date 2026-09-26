"""How many worker processes a scan run gets: the recipe asks, the host decides.

A recipe's ``scan.workers`` is a request. The host caps it at
:func:`host_worker_cap` — a facility value, so it lives in the client
``config.ini`` (``[analysis] worker_cap``) and never in a recipe — and runs a
small scan serially regardless, because a worker costs seconds of imports and
a few hundred MB before it reads its first frame. Whatever the count, the
outputs are the same: the core yields pooled outcomes in declared order.
"""

from __future__ import annotations

import configparser
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "CONFIG_OPTION",
    "CONFIG_SECTION",
    "MIN_UNITS_FOR_POOL",
    "effective_workers",
    "host_worker_cap",
]

#: A run with fewer execution units than this stays serial whatever it asks:
#: below it the pool's start-up costs more than it saves.
MIN_UNITS_FOR_POOL = 50

#: Where the host's cap lives in ``~/.config/geecs_python_api/config.ini``.
CONFIG_SECTION = "analysis"
CONFIG_OPTION = "worker_cap"

_USER_CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini")


def default_worker_cap() -> int:
    """Every core but one, and never fewer than one."""
    return max(1, (os.cpu_count() or 2) - 1)


def host_worker_cap(config_path: Optional[Path] = None) -> int:
    """The most workers one run may use on this host.

    ``[analysis] worker_cap`` in the client config (``config_path`` defaults
    to the shared ``config.ini``); absent, unreadable or not a positive
    integer, the default is every core but one. A value that cannot be read
    is logged and ignored rather than raised, so a bad line in a config
    file never stops an analysis.
    """
    path = (config_path or _USER_CONFIG_PATH).expanduser()
    if not path.exists():
        return default_worker_cap()
    try:
        config = configparser.ConfigParser()
        config.read(path)
        value = config.getint(CONFIG_SECTION, CONFIG_OPTION, fallback=None)
    except Exception as exc:  # noqa: BLE001 — a config line, never a stop
        logger.warning(
            "Ignoring [%s] %s in %s: %s", CONFIG_SECTION, CONFIG_OPTION, path, exc
        )
        return default_worker_cap()
    if value is None:
        return default_worker_cap()
    if value < 1:
        logger.warning(
            "Ignoring [%s] %s = %s in %s: it must be a positive integer",
            CONFIG_SECTION,
            CONFIG_OPTION,
            value,
            path,
        )
        return default_worker_cap()
    return value


def effective_workers(requested: int, units: int, *, cap: Optional[int] = None) -> int:
    """The worker count a run of ``units`` execution units gets.

    ``1`` (serial, no pool) when the recipe asks for one, or when the run is
    smaller than :data:`MIN_UNITS_FOR_POOL`; otherwise the request, capped at
    ``cap`` (the host's :func:`host_worker_cap` when not given).
    """
    if requested <= 1 or units < MIN_UNITS_FOR_POOL:
        return 1
    limit = host_worker_cap() if cap is None else cap
    return max(1, min(int(requested), int(limit)))
