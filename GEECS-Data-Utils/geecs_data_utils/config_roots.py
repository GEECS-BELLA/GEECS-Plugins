"""
Shared config directory managers for GEECS plugins.

Exposes pre-configured `ConfigDirManager` instances for:
- ImageAnalysis configs (env: IMAGE_ANALYSIS_CONFIG_DIR; fallback: config.ini Paths.config_root/image_analysis/cameras)
- ScanAnalysis configs (env: SCAN_ANALYSIS_CONFIG_DIR)
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path

from .config_base import ConfigDirManager

logger = logging.getLogger(__name__)


def _resolve_image_from_ini() -> Path | None:
    """Resolve image analysis config dir from ~/.config/geecs_python_api/config.ini."""
    config_path = Path("~/.config/geecs_python_api/config.ini").expanduser()
    if not config_path.exists():
        return None

    try:
        config = configparser.ConfigParser()
        config.read(config_path)
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


image_analysis_config = ConfigDirManager(
    env_var="IMAGE_ANALYSIS_CONFIG_DIR",
    logger=logger,
    name="Image analysis config",
    fallback_resolver=_resolve_image_from_ini,
    fallback_name="config.ini Paths.config_root",
)

scan_analysis_config = ConfigDirManager(
    env_var="SCAN_ANALYSIS_CONFIG_DIR",
    logger=logger,
    name="Scan analysis config",
)

for manager in (image_analysis_config, scan_analysis_config):
    manager.bootstrap_from_env_or_fallback()

__all__ = ["image_analysis_config", "scan_analysis_config"]
