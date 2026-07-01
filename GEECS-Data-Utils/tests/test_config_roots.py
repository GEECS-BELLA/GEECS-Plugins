"""Tests for shared config-root bootstrap behavior."""

from __future__ import annotations

import importlib
import os
from pathlib import Path


def test_image_analysis_config_uses_unified_scan_root(tmp_path: Path) -> None:
    """Legacy image config manager should not use IMAGE_ANALYSIS_CONFIG_DIR."""
    scan_root = tmp_path / "scan_analysis_configs"
    image_root = tmp_path / "image_analysis_configs"
    scan_root.mkdir()
    image_root.mkdir()
    old_scan = os.environ.get("SCAN_ANALYSIS_CONFIG_DIR")
    old_image = os.environ.get("IMAGE_ANALYSIS_CONFIG_DIR")

    try:
        os.environ["SCAN_ANALYSIS_CONFIG_DIR"] = str(scan_root)
        os.environ["IMAGE_ANALYSIS_CONFIG_DIR"] = str(image_root)

        import geecs_data_utils.config_roots as config_roots

        config_roots = importlib.reload(config_roots)

        assert config_roots.scan_analysis_config.base_dir == scan_root.resolve()
        assert config_roots.image_analysis_config.base_dir == scan_root.resolve()
        assert config_roots.image_analysis_config.env_var == "SCAN_ANALYSIS_CONFIG_DIR"
    finally:
        _restore_env("SCAN_ANALYSIS_CONFIG_DIR", old_scan)
        _restore_env("IMAGE_ANALYSIS_CONFIG_DIR", old_image)
        import geecs_data_utils.config_roots as config_roots

        importlib.reload(config_roots)


def _restore_env(name: str, value: str | None) -> None:
    """Restore one environment variable to its previous state."""
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
