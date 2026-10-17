"""
Shared config directory manager for the unified analysis config tree.

Resolution precedence (first hit wins):

1. ``SCAN_ANALYSIS_CONFIG_DIR`` environment variable
2. ``config.ini`` ``[Paths] scan_analysis_configs_path``
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path

from .config_base import ConfigDirManager

logger = logging.getLogger(__name__)

_USER_CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini")


def _read_paths_option(option: str) -> str | None:
    """Read one option from the [Paths] section of the shared user config.ini."""
    config_path = _USER_CONFIG_PATH.expanduser()
    if not config_path.exists():
        return None

    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        return config.get("Paths", option, fallback=None)
    except Exception as exc:  # pragma: no cover  # noqa: BLE001 — log only
        logger.warning("Error reading config from %s: %s", config_path, exc)
        return None


def _resolve_scan_from_ini() -> Path | None:
    """Resolve the analysis config dir from the user config.ini."""
    if scan_path := _read_paths_option("scan_analysis_configs_path"):
        scan_dir = Path(scan_path).expanduser().resolve()
        if scan_dir.exists():
            return scan_dir
    return None


scan_analysis_config = ConfigDirManager(
    env_var="SCAN_ANALYSIS_CONFIG_DIR",
    logger=logger,
    name="Unified analysis config",
    fallback_resolver=_resolve_scan_from_ini,
    fallback_name="config.ini Paths.scan_analysis_configs_path",
)

scan_analysis_config.bootstrap_from_env_or_fallback()

__all__ = ["scan_analysis_config"]
