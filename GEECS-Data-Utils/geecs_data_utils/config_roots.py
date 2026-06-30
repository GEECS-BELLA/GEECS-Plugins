"""
Shared config directory managers for GEECS plugins.

Exposes pre-configured `ConfigDirManager` instances for:
- Scan/ImageAnalysis configs (env: SCAN_ANALYSIS_CONFIG_DIR; fallback: config.ini Paths.scan_analysis_configs_path)
- Legacy ImageAnalysis configs (env: IMAGE_ANALYSIS_CONFIG_DIR; fallback: config.ini Paths.image_analysis_configs_path)
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path

from .config_base import ConfigDirManager

logger = logging.getLogger(__name__)


def _resolve_image_from_ini() -> Path | None:
    """Resolve legacy image analysis config dir from user config."""
    config_path = Path("~/.config/geecs_python_api/config.ini").expanduser()
    if not config_path.exists():
        return None

    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        if image_path := config.get(
            "Paths", "image_analysis_configs_path", fallback=None
        ):
            image_dir = Path(image_path).expanduser().resolve()
            if image_dir.exists():
                return image_dir
        if config_root := config.get("Paths", "config_root", fallback=None):
            if config_root:
                image_dir = (
                    Path(config_root).expanduser().resolve()
                    / "image_analysis"
                    / "cameras"
                )
                if image_dir.exists():
                    return image_dir
    except Exception as exc:  # pragma: no cover - log only
        logger.warning("Error reading config from %s: %s", config_path, exc)

    return None


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


image_analysis_config = ConfigDirManager(
    env_var="IMAGE_ANALYSIS_CONFIG_DIR",
    logger=logger,
    name="Legacy image analysis config",
    fallback_resolver=_resolve_image_from_ini,
    fallback_name="config.ini Paths.image_analysis_configs_path",
)

scan_analysis_config = ConfigDirManager(
    env_var="SCAN_ANALYSIS_CONFIG_DIR",
    logger=logger,
    name="Unified analysis config",
    fallback_resolver=_resolve_scan_from_ini,
    fallback_name="config.ini Paths.scan_analysis_configs_path",
)

for manager in (image_analysis_config, scan_analysis_config):
    manager.bootstrap_from_env_or_fallback()

__all__ = ["image_analysis_config", "scan_analysis_config"]
