"""Tests for GeecsPathsConfig path resolution after the unified-configs migration.

The relevant change: after the migration, the image-analysis config
path derives from ``scan_analysis_configs_path / "analyzers"`` rather
than living as its own config-file key. The legacy
``image_analysis_configs_path`` key is still read (with a deprecation
warning) so users who haven't updated their ``config.ini`` keep
working.
"""

from __future__ import annotations

import logging
from pathlib import Path
import textwrap

import pytest

from geecs_data_utils.geecs_paths_config import GeecsPathsConfig


@pytest.fixture
def propagate_paths_config_logger():
    """The module sets propagate=False; flip it so caplog can capture."""
    pcfg_logger = logging.getLogger("geecs_data_utils.geecs_paths_config")
    original = pcfg_logger.propagate
    pcfg_logger.propagate = True
    try:
        yield
    finally:
        pcfg_logger.propagate = original


def _write_ini(path: Path, body: str) -> None:
    # All tests need an [Experiment] section because the loader reads
    # `expt` from it before it gets to the [Paths] block.
    full = textwrap.dedent(body).lstrip()
    if "[Experiment]" not in full:
        full += "\n[Experiment]\nexpt = TestExperiment\n"
    path.write_text(full)


@pytest.fixture
def base_dir(tmp_path) -> Path:
    """Create a fake GEECS_DATA_LOCAL_BASE_PATH that exists."""
    base = tmp_path / "data"
    base.mkdir()
    return base


@pytest.fixture
def scan_configs_with_analyzers(tmp_path) -> Path:
    """Create a fake scan_analysis_configs tree with an analyzers/ subdir."""
    root = tmp_path / "scan_analysis_configs"
    (root / "analyzers" / "HTU").mkdir(parents=True)
    return root


class TestImagePathDerivation:
    """ImageAnalysis path is derived from scan_analysis_configs_path / analyzers."""

    def test_derives_from_scan_path_when_legacy_absent(
        self, tmp_path, base_dir, scan_configs_with_analyzers
    ):
        ini = tmp_path / "config.ini"
        _write_ini(
            ini,
            f"""
            [Paths]
            GEECS_DATA_LOCAL_BASE_PATH = {base_dir}
            scan_analysis_configs_path = {scan_configs_with_analyzers}
            """,
        )

        cfg = GeecsPathsConfig(config_path=ini)
        assert cfg.scan_analysis_configs_path == scan_configs_with_analyzers
        assert cfg.image_analysis_configs_path == (
            scan_configs_with_analyzers / "analyzers"
        )

    def test_legacy_key_still_honoured_with_deprecation_warning(
        self,
        tmp_path,
        base_dir,
        scan_configs_with_analyzers,
        caplog,
        propagate_paths_config_logger,
    ):
        legacy_image_dir = tmp_path / "legacy_image_configs"
        legacy_image_dir.mkdir()

        ini = tmp_path / "config.ini"
        _write_ini(
            ini,
            f"""
            [Paths]
            GEECS_DATA_LOCAL_BASE_PATH = {base_dir}
            scan_analysis_configs_path = {scan_configs_with_analyzers}
            image_analysis_configs_path = {legacy_image_dir}
            """,
        )

        with caplog.at_level("WARNING"):
            cfg = GeecsPathsConfig(config_path=ini)

        assert cfg.image_analysis_configs_path == legacy_image_dir
        assert any(
            "image_analysis_configs_path in config.ini is deprecated" in r.message
            for r in caplog.records
        )

    def test_falls_back_to_derivation_when_legacy_path_does_not_exist(
        self, tmp_path, base_dir, scan_configs_with_analyzers, caplog
    ):
        # Legacy key set but the path it points to does not exist.
        bogus_legacy = tmp_path / "no_such_dir"

        ini = tmp_path / "config.ini"
        _write_ini(
            ini,
            f"""
            [Paths]
            GEECS_DATA_LOCAL_BASE_PATH = {base_dir}
            scan_analysis_configs_path = {scan_configs_with_analyzers}
            image_analysis_configs_path = {bogus_legacy}
            """,
        )

        with caplog.at_level("WARNING"):
            cfg = GeecsPathsConfig(config_path=ini)

        # Fell back to derivation since the legacy path was invalid.
        assert cfg.image_analysis_configs_path == (
            scan_configs_with_analyzers / "analyzers"
        )

    def test_derivation_returns_none_when_analyzers_subdir_missing(
        self, tmp_path, base_dir
    ):
        # scan_analysis_configs_path exists but lacks an analyzers/ subdir
        # (e.g. a fresh setup before migration).
        bare = tmp_path / "scan_only"
        bare.mkdir()

        ini = tmp_path / "config.ini"
        _write_ini(
            ini,
            f"""
            [Paths]
            GEECS_DATA_LOCAL_BASE_PATH = {base_dir}
            scan_analysis_configs_path = {bare}
            """,
        )

        cfg = GeecsPathsConfig(config_path=ini)
        assert cfg.scan_analysis_configs_path == bare
        # analyzers/ doesn't exist → image path is None, no crash.
        assert cfg.image_analysis_configs_path is None

    def test_explicit_constructor_argument_wins(
        self, tmp_path, base_dir, scan_configs_with_analyzers
    ):
        explicit = tmp_path / "explicit_image_configs"
        explicit.mkdir()

        ini = tmp_path / "config.ini"
        _write_ini(
            ini,
            f"""
            [Paths]
            GEECS_DATA_LOCAL_BASE_PATH = {base_dir}
            scan_analysis_configs_path = {scan_configs_with_analyzers}
            """,
        )

        cfg = GeecsPathsConfig(config_path=ini, image_analysis_configs_path=explicit)
        assert cfg.image_analysis_configs_path == explicit
