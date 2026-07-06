"""Tests for LiveTaskRunner config-root wiring.

Pins the fix for the review finding where ``image_config_dir`` set
``image_analysis_config.base_dir`` while ImageAnalysis's camera/line
loader resolves through the distinct ``scan_analysis_config`` instance —
so the documented parameter silently did nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from geecs_data_utils.config_roots import image_analysis_config, scan_analysis_config
from scan_analysis.live_task_runner import apply_config_roots


@pytest.fixture(autouse=True)
def restore_config_roots():
    """Save and restore the shared config-root singletons around each test."""
    prev_scan = scan_analysis_config.base_dir
    prev_image = image_analysis_config.base_dir
    yield
    scan_analysis_config._base_dir = prev_scan
    image_analysis_config._base_dir = prev_image
    scan_analysis_config.clear_cache(log=False)
    image_analysis_config.clear_cache(log=False)


def test_image_analysis_loader_resolves_through_scan_analysis_config() -> None:
    """ImageAnalysis's camera-config loader must use scan_analysis_config."""
    from image_analysis.config import loader

    assert loader._CONFIG_MANAGER is scan_analysis_config


def test_image_config_dir_reaches_image_analysis_loader_root(
    tmp_path: Path,
) -> None:
    """image_config_dir must land on the root the ImageAnalysis loader uses."""
    image_dir = tmp_path / "image_configs"
    image_dir.mkdir()

    apply_config_roots(image_config_dir=image_dir)

    # The loader resolves through scan_analysis_config (see test above),
    # so the override is only effective if it lands there.
    assert scan_analysis_config.base_dir == image_dir.resolve()
    # Legacy alias kept in sync for callers that still read it directly.
    assert image_analysis_config.base_dir == image_dir.resolve()


def test_config_dir_sets_scan_root_only(tmp_path: Path) -> None:
    """config_dir alone points the unified root without touching the alias."""
    scan_dir = tmp_path / "scan_configs"
    scan_dir.mkdir()
    prev_image = image_analysis_config.base_dir

    apply_config_roots(config_dir=scan_dir)

    assert scan_analysis_config.base_dir == scan_dir.resolve()
    assert image_analysis_config.base_dir == prev_image


def test_image_config_dir_wins_for_camera_lookup_when_both_given(
    tmp_path: Path,
) -> None:
    """With both dirs given, camera/line lookup resolves under image_config_dir."""
    scan_dir = tmp_path / "scan_configs"
    image_dir = tmp_path / "image_configs"
    scan_dir.mkdir()
    image_dir.mkdir()

    apply_config_roots(config_dir=scan_dir, image_config_dir=image_dir)

    assert scan_analysis_config.base_dir == image_dir.resolve()
    assert image_analysis_config.base_dir == image_dir.resolve()
