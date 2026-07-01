"""
Shared config directory managers for GEECS plugins.

Exposes pre-configured `ConfigDirManager` instances for:
- Scan/ImageAnalysis configs (env: SCAN_ANALYSIS_CONFIG_DIR; fallback: config.ini Paths.scan_analysis_configs_path)
- `image_analysis_config`, a legacy compatibility manager that resolves from the
  same unified Scan/ImageAnalysis config root.
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path

from .config_base import ConfigDirManager

logger = logging.getLogger(__name__)


def _resolve_scan_from_ini() -> Path | None:
    """Resolve unified Scan/ImageAnalysis config dir from user config."""
    config_path = Path("~/.config/geecs_python_api/config.ini").expanduser()
    if not config_path.exists():
        return None

    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        if scan_path := config.get(
            "Paths", "scan_analysis_configs_path", fallback=None
        ):
            scan_dir = Path(scan_path).expanduser().resolve()
            if scan_dir.exists():
                return scan_dir
    except Exception as exc:  # pragma: no cover - log only
        logger.warning("Error reading config from %s: %s", config_path, exc)

    return None


scan_analysis_config = ConfigDirManager(
    env_var="SCAN_ANALYSIS_CONFIG_DIR",
    logger=logger,
    name="Unified analysis config",
    fallback_resolver=_resolve_scan_from_ini,
    fallback_name="config.ini Paths.scan_analysis_configs_path",
)

image_analysis_config = ConfigDirManager(
    env_var="SCAN_ANALYSIS_CONFIG_DIR",
    logger=logger,
    name="Legacy ImageAnalysis config alias",
    fallback_resolver=_resolve_scan_from_ini,
    fallback_name="config.ini Paths.scan_analysis_configs_path",
)

for manager in (image_analysis_config, scan_analysis_config):
    manager.bootstrap_from_env_or_fallback()

__all__ = ["image_analysis_config", "scan_analysis_config"]
