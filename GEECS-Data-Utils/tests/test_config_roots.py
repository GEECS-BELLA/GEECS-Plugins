"""Tests for the analysis config-root bootstrap."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from geecs_data_utils.config_base import ConfigDirManager


def _fresh_manager(config_roots) -> ConfigDirManager:
    """Build a manager wired exactly like the module-level singleton."""
    return ConfigDirManager(
        env_var="SCAN_ANALYSIS_CONFIG_DIR",
        logger=logging.getLogger("test_config_roots"),
        name="Test config",
        fallback_resolver=config_roots._resolve_scan_from_ini,
        fallback_name="config.ini",
    )


@pytest.fixture
def config_roots_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Import config_roots with the env var cleared and the user ini neutralized."""
    import geecs_data_utils.config_roots as config_roots

    monkeypatch.delenv("SCAN_ANALYSIS_CONFIG_DIR", raising=False)
    monkeypatch.setattr(
        config_roots, "_USER_CONFIG_PATH", tmp_path / "nonexistent_config.ini"
    )
    return config_roots


def test_env_var_wins_over_ini_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_roots_isolated
) -> None:
    """SCAN_ANALYSIS_CONFIG_DIR beats the config.ini key."""
    env_root = tmp_path / "env_configs"
    ini_root = tmp_path / "ini_configs"
    env_root.mkdir()
    ini_root.mkdir()
    ini_path = tmp_path / "config.ini"
    ini_path.write_text(f"[Paths]\nscan_analysis_configs_path = {ini_root}\n")
    monkeypatch.setattr(config_roots_isolated, "_USER_CONFIG_PATH", ini_path)
    monkeypatch.setenv("SCAN_ANALYSIS_CONFIG_DIR", str(env_root))

    manager = _fresh_manager(config_roots_isolated)
    manager.bootstrap_from_env_or_fallback()

    assert manager.base_dir == env_root.resolve()


def test_ini_key_resolves_without_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_roots_isolated
) -> None:
    """With no env var, the config.ini key supplies the root."""
    ini_root = tmp_path / "ini_configs"
    ini_root.mkdir()
    ini_path = tmp_path / "config.ini"
    ini_path.write_text(f"[Paths]\nscan_analysis_configs_path = {ini_root}\n")
    monkeypatch.setattr(config_roots_isolated, "_USER_CONFIG_PATH", ini_path)

    manager = _fresh_manager(config_roots_isolated)
    manager.bootstrap_from_env_or_fallback()

    assert manager.base_dir == ini_root.resolve()


def test_no_sources_leaves_base_dir_unset(config_roots_isolated) -> None:
    """With no sources configured at all, bootstrap leaves base_dir as None."""
    manager = _fresh_manager(config_roots_isolated)
    manager.bootstrap_from_env_or_fallback()

    assert manager.base_dir is None
