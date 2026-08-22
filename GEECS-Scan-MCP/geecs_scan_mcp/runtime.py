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
from pathlib import Path
from typing import Any, Optional

from geecs_scan_mcp import __version__

_USER_CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini")

#: The submitted-as identity the manager records on queue items, and the
#: ``SubmissionRecord.client`` prefix — how runs trace back to this server.
CLIENT_IDENTITY = f"geecs-scan-mcp {__version__}"

_cache: dict[str, Any] = {}


def clear_runtime_cache() -> None:
    """Drop every cached singleton (tests; config re-read on next call)."""
    _cache.clear()


def get_experiment() -> Optional[str]:
    """The configured experiment name (``[Experiment] expt``), or ``None``."""
    if "experiment" not in _cache:
        experiment: Optional[str] = None
        path = _USER_CONFIG_PATH.expanduser()
        if path.exists():
            parser = configparser.ConfigParser()
            try:
                parser.read(path)
                experiment = (
                    parser.get("Experiment", "expt", fallback="").strip() or None
                )
            except (OSError, configparser.Error):
                experiment = None
        _cache["experiment"] = experiment
    return _cache["experiment"]


def get_queue_client() -> Any:
    """The shared RE Manager client, stamped with this server's identity."""
    if "queue_client" not in _cache:
        from geecs_bluesky.qs_client import make_queue_client

        _cache["queue_client"] = make_queue_client(
            get_experiment() or "", user=CLIENT_IDENTITY
        )
    return _cache["queue_client"]


def get_resolver() -> Any:
    """The experiment's ``ConfigsRepoResolver``, or ``None`` without an experiment."""
    if "resolver" not in _cache:
        experiment = get_experiment()
        if experiment is None:
            _cache["resolver"] = None
        else:
            from geecs_bluesky.config_resolver import ConfigsRepoResolver

            _cache["resolver"] = ConfigsRepoResolver(experiment)
    return _cache["resolver"]


def get_catalog() -> Any:
    """The Tiled scan catalog (unconfigured installs get an honest probe)."""
    if "catalog" not in _cache:
        from geecs_data_utils.tiled_catalog import TiledScanCatalog

        _cache["catalog"] = TiledScanCatalog.from_config()
    return _cache["catalog"]
