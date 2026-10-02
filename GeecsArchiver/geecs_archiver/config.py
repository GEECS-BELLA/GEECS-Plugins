"""Where the tool finds the appliance, the experiment and the configs repo.

Two homes, per the site profile (``docs/platform/site_profile.md``): the
client ``~/.config/geecs_python_api/config.ini`` (``[archiver] url``,
``[Experiment] expt``, ``[Paths] scanner_config_root_path``) and the
environment (``GEECS_ARCHIVER_URL`` overrides the file — the unit's
``site.env`` is one producer of it).  Nothing here is a lab literal.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

import yaml
from geecs_schemas import ArchivePolicy, DerivedChannels

CONFIG_PATH = Path.home() / ".config" / "geecs_python_api" / "config.ini"
DEFAULT_PORT = 17665

#: Where the per-experiment overlay lives inside the configs repo, beside
#: the gateway's ``gateway/derived_channels.yaml``.
ARCHIVER_FOLDER = "archiver"
POLICY_FILENAME = "archive_policy.yaml"
GATEWAY_FOLDER = "gateway"
DERIVED_CHANNELS_FILENAME = "derived_channels.yaml"


def read_config_entry(
    section: str, key: str, config_path: Path | None = None
) -> str | None:
    """One ``config.ini`` value (``None`` when the file, section or key is absent)."""
    path = config_path or CONFIG_PATH
    if not path.exists():
        return None
    parser = configparser.ConfigParser()
    parser.read(path)
    if section not in parser:
        return None
    value = parser[section].get(key)
    return value.strip() if value and value.strip() else None


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
    """The data-retrieval root for an appliance base URL (Phoebus: ``pbraw://host:port/retrieval``)."""
    return base_url.rstrip("/") + "/retrieval"


def scanner_configs_base(config_path: Path | None = None) -> Path | None:
    """Resolve the configs repo's ``scanner_configs/experiments`` directory.

    The same three-step resolution the CA gateway and GeecsBluesky use,
    without importing either:

    1. ``GEECS_SCANNER_CONFIG_DIR`` points directly at ``scanner_configs/experiments``.
    2. ``GEECS_PLUGINS_CONFIGS`` points at the configs repo root.
    3. ``config.ini`` ``[Paths] scanner_config_root_path`` points at the repo root.
    """
    env = os.environ.get("GEECS_SCANNER_CONFIG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    repo_env = os.environ.get("GEECS_PLUGINS_CONFIGS")
    if repo_env:
        return Path(repo_env).expanduser().resolve() / "scanner_configs" / "experiments"
    root = read_config_entry("Paths", "scanner_config_root_path", config_path)
    if root:
        return Path(root).expanduser().resolve() / "scanner_configs" / "experiments"
    return None


def policy_path(experiment: str, base: Path | None = None) -> Path | None:
    """The experiment's ``archive_policy.yaml`` if it exists, else ``None``."""
    base = base if base is not None else scanner_configs_base()
    if base is None:
        return None
    path = base / experiment / ARCHIVER_FOLDER / POLICY_FILENAME
    return path if path.exists() else None


def derived_channels_path(experiment: str, base: Path | None = None) -> Path | None:
    """The experiment's gateway ``derived_channels.yaml`` if it exists, else ``None``."""
    base = base if base is not None else scanner_configs_base()
    if base is None:
        return None
    path = base / experiment / GATEWAY_FOLDER / DERIVED_CHANNELS_FILENAME
    return path if path.exists() else None


def load_policy(path: Path | None) -> ArchivePolicy:
    """Load an :class:`ArchivePolicy`; no file means the defaults."""
    if path is None:
        return ArchivePolicy()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ArchivePolicy.model_validate(data)


def load_derived_channels(path: Path | None) -> DerivedChannels | None:
    """Load the gateway's derived-channel overlay, or ``None`` when there is none."""
    if path is None:
        return None
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return DerivedChannels.model_validate(data)
