"""Where the tool finds the appliance, the experiment and the configs-repo files.

Two homes, per the site profile (``docs/platform/site_profile.md``): the
client ``~/.config/geecs_python_api/config.ini`` (``[archiver] url``,
``[Experiment] expt``, ``[Paths] scanner_config_root_path``) and the
environment (``GEECS_ARCHIVER_URL`` overrides the file — the unit's
``site.env`` is one producer of it).  The configs-repository lookup is
GEECS-Core's (:mod:`geecs_core.configs_repo`), shared with the gateway.
Nothing here is a lab literal.
"""

from __future__ import annotations

import os
from pathlib import Path

from geecs_core.configs_repo import (
    experiment_config_path,
    read_config_entry,
    scanner_configs_base,
)
from geecs_schemas import (
    ARCHIVE_POLICY_FILENAME,
    ARCHIVER_CONFIG_FOLDER,
    DERIVED_CHANNELS_FILENAME,
    GATEWAY_CONFIG_FOLDER,
    ArchivePolicy,
    DerivedChannels,
)

DEFAULT_PORT = 17665


__all__ = [
    "archiver_url",
    "experiment_name",
    "mgmt_url",
    "retrieval_url",
    "scanner_configs_base",
    "policy_path",
    "derived_channels_path",
    "load_policy",
    "load_derived_channels",
]


def archiver_url(config_path: Path | None = None) -> str | None:
    """The appliance's base URL (``http://host:17665``), environment first.

    ``GEECS_ARCHIVER_URL`` wins when set (the service host's ``site.env`` can
    carry it); otherwise ``config.ini`` ``[archiver] url``.  A trailing slash
    is dropped so the ``/mgmt/bpl`` and ``/retrieval`` suffixes join cleanly.
    """
    env = os.environ.get("GEECS_ARCHIVER_URL", "").strip()
    if env:
        return env.rstrip("/")
    value = read_config_entry("archiver", "url", config_path)
    return value.rstrip("/") if value else None


def experiment_name(config_path: Path | None = None) -> str | None:
    """The experiment ``config.ini`` names (``[Experiment] expt``)."""
    return read_config_entry("Experiment", "expt", config_path)


def mgmt_url(base_url: str) -> str:
    """The management API root for an appliance base URL."""
    return base_url.rstrip("/") + "/mgmt/bpl"


def retrieval_url(base_url: str) -> str:
    """The data-retrieval root (Phoebus: ``pbraw://host:port/retrieval``)."""
    return base_url.rstrip("/") + "/retrieval"


def policy_path(experiment: str, base: Path | None = None) -> Path | None:
    """The experiment's ``archive_policy.yaml`` if it exists, else ``None``."""
    return experiment_config_path(
        experiment, ARCHIVER_CONFIG_FOLDER, ARCHIVE_POLICY_FILENAME, base=base
    )


def derived_channels_path(experiment: str, base: Path | None = None) -> Path | None:
    """The experiment's gateway ``derived_channels.yaml`` if it exists, else ``None``."""
    return experiment_config_path(
        experiment, GATEWAY_CONFIG_FOLDER, DERIVED_CHANNELS_FILENAME, base=base
    )


def load_policy(path: Path | None) -> ArchivePolicy:
    """Load an :class:`ArchivePolicy`; no file means the defaults."""
    return ArchivePolicy() if path is None else ArchivePolicy.from_path(path)


def load_derived_channels(path: Path | None) -> DerivedChannels | None:
    """Load the gateway's derived-channel overlay, or ``None`` when there is none."""
    return None if path is None else DerivedChannels.from_path(path)
