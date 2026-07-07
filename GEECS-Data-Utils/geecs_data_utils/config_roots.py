"""
Shared config directory managers for GEECS plugins.

Exposes pre-configured `ConfigDirManager` instances for:
- Scan/ImageAnalysis configs (env: SCAN_ANALYSIS_CONFIG_DIR; fallback: config.ini Paths.scan_analysis_configs_path)
- `image_analysis_config`, a legacy compatibility manager that resolves from the
  same unified Scan/ImageAnalysis config root.

Deployments configured only via the legacy ImageAnalysis sources — the
``IMAGE_ANALYSIS_CONFIG_DIR`` environment variable or the ``config.ini``
``Paths.config_root`` key (pointing at ``<config_root>/image_analysis/cameras``)
— continue to resolve: when none of the unified sources are set, the legacy
sources are consulted as a deprecated fallback. A ``DeprecationWarning`` and a
warning log line identify which legacy source was used and what to migrate to.

Resolution precedence (first hit wins):

1. ``SCAN_ANALYSIS_CONFIG_DIR`` environment variable
2. ``config.ini`` ``[Paths] scan_analysis_configs_path``
3. ``IMAGE_ANALYSIS_CONFIG_DIR`` environment variable (deprecated)
4. ``config.ini`` ``[Paths] config_root`` → ``image_analysis/cameras`` (deprecated)
"""

from __future__ import annotations

import configparser
import logging
import os
import warnings
from pathlib import Path

from .config_base import ConfigDirManager

logger = logging.getLogger(__name__)

_USER_CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini")

_LEGACY_ENV_VAR = "IMAGE_ANALYSIS_CONFIG_DIR"
_MIGRATION_HINT = (
    "migrate to the SCAN_ANALYSIS_CONFIG_DIR environment variable or the "
    "config.ini [Paths] scan_analysis_configs_path key"
)


def _read_paths_option(option: str) -> str | None:
    """Read one option from the [Paths] section of the shared user config.ini."""
    config_path = _USER_CONFIG_PATH.expanduser()
    if not config_path.exists():
        return None

    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        return config.get("Paths", option, fallback=None)
    except Exception as exc:  # pragma: no cover - log only
        logger.warning("Error reading config from %s: %s", config_path, exc)
        return None


def _resolve_scan_from_ini() -> Path | None:
    """Resolve unified Scan/ImageAnalysis config dir from user config."""
    if scan_path := _read_paths_option("scan_analysis_configs_path"):
        scan_dir = Path(scan_path).expanduser().resolve()
        if scan_dir.exists():
            return scan_dir
    return None


def _warn_legacy_source(source: str, path: Path) -> None:
    """Emit a DeprecationWarning + log line for a legacy config source."""
    message = (
        f"Analysis config root resolved from deprecated legacy source "
        f"{source} ({path}); {_MIGRATION_HINT}."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    logger.warning(message)


def _resolve_legacy_image_sources() -> Path | None:
    """Resolve the deprecated ImageAnalysis config sources, warning if used."""
    if env_dir := os.getenv(_LEGACY_ENV_VAR):
        legacy_dir = Path(env_dir).expanduser().resolve()
        if legacy_dir.exists():
            _warn_legacy_source(f"env var {_LEGACY_ENV_VAR}", legacy_dir)
            return legacy_dir
        logger.warning("%s is set but does not exist: %s", _LEGACY_ENV_VAR, legacy_dir)

    if config_root := _read_paths_option("config_root"):
        legacy_dir = (
            Path(config_root).expanduser().resolve() / "image_analysis" / "cameras"
        )
        if legacy_dir.exists():
            _warn_legacy_source("config.ini Paths.config_root", legacy_dir)
            return legacy_dir

    return None


def _resolve_unified_fallback() -> Path | None:
    """Resolve the config root: new ini key first, then deprecated legacy sources."""
    if scan_dir := _resolve_scan_from_ini():
        return scan_dir
    return _resolve_legacy_image_sources()


scan_analysis_config = ConfigDirManager(
    env_var="SCAN_ANALYSIS_CONFIG_DIR",
    logger=logger,
    name="Unified analysis config",
    fallback_resolver=_resolve_unified_fallback,
    fallback_name="config.ini Paths.scan_analysis_configs_path (or legacy ImageAnalysis sources)",
)

image_analysis_config = ConfigDirManager(
    env_var="SCAN_ANALYSIS_CONFIG_DIR",
    logger=logger,
    name="Legacy ImageAnalysis config alias",
    fallback_resolver=_resolve_unified_fallback,
    fallback_name="config.ini Paths.scan_analysis_configs_path (or legacy ImageAnalysis sources)",
)

for manager in (image_analysis_config, scan_analysis_config):
    manager.bootstrap_from_env_or_fallback()

__all__ = ["image_analysis_config", "scan_analysis_config"]
