"""Lazily-built, cached singletons the tools call through.

Everything resolves from the standard
``~/.config/geecs_python_api/config.ini`` (the fleet contract — no new
config format): ``[Experiment] expt`` names the experiment, ``[qserver]``
the manager, ``[tiled]`` the archive, ``[Paths]``/env the configs
checkout.  Each getter builds on first use and caches; tests monkeypatch
the getters on this module (tools always call ``runtime.get_*()``
through the module attribute, never a from-import, so the patch seam
holds).

Construction is cheap and offline-safe by the underlying seams' own
contracts: an unconfigured ``[qserver]`` yields the stub client (verbs
refuse with a clear message), an unconfigured ``[tiled]`` yields a
catalog whose probe says so, and a missing experiment yields ``None``
resolvers that the tools turn into ``invalid_request`` envelopes.
"""

from __future__ import annotations

import configparser
import threading
from pathlib import Path
from typing import Any, Optional

import logging

from geecs_scan_mcp import __version__

logger = logging.getLogger("geecs_scan_mcp.runtime")

_USER_CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini")

#: The default submitted-as identity — how runs trace back to this
#: server.  A deployment can override it via ``[scan_mcp]
#: client_identity`` (e.g. ``osprey-htu-assistant``); the queue item's
#: ``user`` and the ``SubmissionRecord.client`` always use the same
#: value, or stop_scan's ownership check would never match.
CLIENT_IDENTITY = f"geecs-scan-mcp {__version__}"

#: The agent scan-size cap default (owner decision 2026-08-22:
#: conservative first posture).  Override: ``[scan_mcp] max_shots``.
DEFAULT_MAX_SHOTS = 1000

_cache: dict[str, Any] = {}
# First-use builds race under FastMCP's concurrent tool dispatch (one agent
# turn can issue parallel calls); without the lock two threads could each
# build a ZmqQueueClient and the loser's zmq receive thread would leak for
# the process lifetime — the same failure mode qs_client's own #653 lock
# prevents one level down.
_cache_lock = threading.Lock()


def clear_runtime_cache() -> None:
    """Drop every cached singleton (tests; config re-read on next call)."""
    with _cache_lock:
        _cache.clear()


def _cached(key: str, build) -> Any:
    """Return ``_cache[key]``, building it under the lock on first use."""
    with _cache_lock:
        if key not in _cache:
            _cache[key] = build()
        return _cache[key]


def _read_experiment() -> Optional[str]:
    path = _USER_CONFIG_PATH.expanduser()
    if not path.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(path)
        return parser.get("Experiment", "expt", fallback="").strip() or None
    except (OSError, configparser.Error):
        return None


def _read_scan_mcp_option(option: str) -> Optional[str]:
    """One ``[scan_mcp]`` option from the shared config, or ``None``."""
    path = _USER_CONFIG_PATH.expanduser()
    if not path.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(path)
        return parser.get("scan_mcp", option, fallback="").strip() or None
    except (OSError, configparser.Error):
        return None


def get_experiment() -> Optional[str]:
    """The configured experiment name (``[Experiment] expt``), or ``None``."""
    return _cached("experiment", _read_experiment)


def client_identity() -> str:
    """The deployment's submitted-as identity (config override or default)."""
    return _cached(
        "client_identity",
        lambda: _read_scan_mcp_option("client_identity") or CLIENT_IDENTITY,
    )


def max_shots() -> int:
    """The agent scan-size cap (``[scan_mcp] max_shots``, default 1000)."""

    def build() -> int:
        raw = _read_scan_mcp_option("max_shots")
        try:
            return int(raw) if raw else DEFAULT_MAX_SHOTS
        except ValueError:
            # A deployment that intended a STRICTER cap must not silently
            # run at the default (review finding) — warn loudly.
            logger.warning(
                "[scan_mcp] max_shots = %r is not an integer — using the default of %d",
                raw,
                DEFAULT_MAX_SHOTS,
            )
            return DEFAULT_MAX_SHOTS

    return _cached("max_shots", build)


def get_queue_client() -> Any:
    """The shared RE Manager client, stamped with this server's identity."""
    # THE one identity spelling: the queue item's user and the
    # SubmissionRecord.client must match or the ownership check never
    # fires — both read client_identity() (review finding).  Resolved
    # BEFORE the cached build: _cached holds the non-reentrant lock, and
    # client_identity() takes it too.
    identity = client_identity()

    def build() -> Any:
        from geecs_bluesky.qs_client import make_queue_client

        return make_queue_client(_read_experiment() or "", user=identity)

    return _cached("queue_client", build)


def get_resolver() -> Any:
    """A FRESH ``ConfigsRepoResolver``, or ``None`` without an experiment.

    Deliberately not cached (unlike the other singletons): the resolver
    caches its scan-variable and action catalogs internally, and config
    edits mid-session (the console editors write these files during
    operations) must appear on the next listing call — a cached resolver
    would serve four kinds fresh and two kinds stale forever.
    Construction is lazy and cheap.
    """
    experiment = get_experiment()
    if experiment is None:
        return None
    from geecs_bluesky.config_resolver import ConfigsRepoResolver

    return ConfigsRepoResolver(experiment)


def get_catalog() -> Any:
    """The Tiled scan catalog (unconfigured installs get an honest probe)."""

    def build() -> Any:
        from geecs_data_utils.tiled_catalog import TiledScanCatalog

        return TiledScanCatalog.from_config()

    return _cached("catalog", build)
