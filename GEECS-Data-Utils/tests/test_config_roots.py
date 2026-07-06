"""Tests for shared config-root bootstrap behavior."""

from __future__ import annotations

import importlib
import logging
import os
import warnings
from pathlib import Path

import pytest

from geecs_data_utils.config_base import ConfigDirManager


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


# ----------------------------------------------------------------------
# Legacy-source fallback (deprecated migration path)
# ----------------------------------------------------------------------


def _fresh_manager(config_roots) -> ConfigDirManager:
    """Build a manager wired exactly like the module-level singletons."""
    return ConfigDirManager(
        env_var="SCAN_ANALYSIS_CONFIG_DIR",
        logger=logging.getLogger("test_config_roots"),
        name="Test config",
        fallback_resolver=config_roots._resolve_unified_fallback,
        fallback_name="unified fallback",
    )


@pytest.fixture
def config_roots_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Import config_roots with env vars cleared and the user ini neutralized."""
    import geecs_data_utils.config_roots as config_roots

    monkeypatch.delenv("SCAN_ANALYSIS_CONFIG_DIR", raising=False)
    monkeypatch.delenv("IMAGE_ANALYSIS_CONFIG_DIR", raising=False)
    # Point ini reads at a nonexistent file so the real user config
    # never leaks into these tests.
    monkeypatch.setattr(
        config_roots, "_USER_CONFIG_PATH", tmp_path / "nonexistent_config.ini"
    )
    return config_roots


def _bootstrap_recording_warnings(manager: ConfigDirManager) -> list:
    """Run bootstrap while recording all warnings it emits."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manager.bootstrap_from_env_or_fallback()
    return caught


def _deprecations(caught: list) -> list:
    """Filter recorded warnings down to DeprecationWarnings."""
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_singletons_use_unified_fallback_resolver() -> None:
    """Both module singletons must resolve via the unified fallback chain."""
    import geecs_data_utils.config_roots as config_roots

    assert (
        config_roots.scan_analysis_config.fallback_resolver
        is config_roots._resolve_unified_fallback
    )
    assert (
        config_roots.image_analysis_config.fallback_resolver
        is config_roots._resolve_unified_fallback
    )


def test_legacy_env_var_fallback_emits_deprecation_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_roots_isolated
) -> None:
    """With only IMAGE_ANALYSIS_CONFIG_DIR set, it resolves — with a warning."""
    legacy_root = tmp_path / "legacy_image_configs"
    legacy_root.mkdir()
    monkeypatch.setenv("IMAGE_ANALYSIS_CONFIG_DIR", str(legacy_root))

    manager = _fresh_manager(config_roots_isolated)
    caught = _bootstrap_recording_warnings(manager)

    assert manager.base_dir == legacy_root.resolve()
    deprecations = _deprecations(caught)
    assert deprecations, "expected a DeprecationWarning for the legacy env var"
    message = str(deprecations[0].message)
    assert "IMAGE_ANALYSIS_CONFIG_DIR" in message
    assert "SCAN_ANALYSIS_CONFIG_DIR" in message  # migration target named


def test_new_env_var_wins_over_legacy_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_roots_isolated
) -> None:
    """When both env vars are set, the unified var wins and no warning fires."""
    new_root = tmp_path / "scan_analysis_configs"
    legacy_root = tmp_path / "legacy_image_configs"
    new_root.mkdir()
    legacy_root.mkdir()
    monkeypatch.setenv("SCAN_ANALYSIS_CONFIG_DIR", str(new_root))
    monkeypatch.setenv("IMAGE_ANALYSIS_CONFIG_DIR", str(legacy_root))

    manager = _fresh_manager(config_roots_isolated)
    caught = _bootstrap_recording_warnings(manager)

    assert manager.base_dir == new_root.resolve()
    assert not _deprecations(caught)


def test_new_ini_key_wins_over_legacy_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_roots_isolated
) -> None:
    """The new ini key beats the legacy env var in the fallback chain."""
    new_root = tmp_path / "scan_analysis_configs"
    legacy_root = tmp_path / "legacy_image_configs"
    new_root.mkdir()
    legacy_root.mkdir()
    ini_path = tmp_path / "config.ini"
    ini_path.write_text(f"[Paths]\nscan_analysis_configs_path = {new_root}\n")
    monkeypatch.setattr(config_roots_isolated, "_USER_CONFIG_PATH", ini_path)
    monkeypatch.setenv("IMAGE_ANALYSIS_CONFIG_DIR", str(legacy_root))

    manager = _fresh_manager(config_roots_isolated)
    caught = _bootstrap_recording_warnings(manager)

    assert manager.base_dir == new_root.resolve()
    assert not _deprecations(caught)


def test_legacy_ini_config_root_fallback_emits_deprecation_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_roots_isolated
) -> None:
    """Old ini key Paths.config_root resolves via image_analysis/cameras."""
    config_root = tmp_path / "configs_repo"
    legacy_cameras = config_root / "image_analysis" / "cameras"
    legacy_cameras.mkdir(parents=True)
    ini_path = tmp_path / "config.ini"
    ini_path.write_text(f"[Paths]\nconfig_root = {config_root}\n")
    monkeypatch.setattr(config_roots_isolated, "_USER_CONFIG_PATH", ini_path)

    manager = _fresh_manager(config_roots_isolated)
    caught = _bootstrap_recording_warnings(manager)

    assert manager.base_dir == legacy_cameras.resolve()
    deprecations = _deprecations(caught)
    assert deprecations, "expected a DeprecationWarning for the legacy ini key"
    assert "config_root" in str(deprecations[0].message)


def test_no_sources_leaves_base_dir_unset(config_roots_isolated) -> None:
    """With no sources configured at all, bootstrap leaves base_dir as None."""
    manager = _fresh_manager(config_roots_isolated)
    caught = _bootstrap_recording_warnings(manager)

    assert manager.base_dir is None
    assert not _deprecations(caught)
