"""Tests for GeecsPathsConfig path resolution."""

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


class TestWaveKitPaths:
    """The optional ``[Paths] wavekit_*`` keys of the ``haso`` measure's host."""

    def test_keys_are_read_and_missing_paths_are_none(
        self, tmp_path, propagate_paths_config_logger, caplog
    ):
        sdk = tmp_path / "wavekit_43"
        sdk.mkdir()
        configs = tmp_path / "configs"
        configs.mkdir()
        ini = tmp_path / "config.ini"
        _write_ini(
            ini,
            f"""
            [Paths]
            GEECS_DATA_LOCAL_BASE_PATH = {tmp_path}
            wavekit_sdk_path = {sdk}
            wavekit_python_path = {tmp_path / "missing" / "python.exe"}
            wavekit_launcher = env WINEPREFIX={tmp_path / "prefix"} wine
            wavekit_configs_path = {configs}
            """,
        )
        with caplog.at_level(logging.WARNING):
            config = GeecsPathsConfig(config_path=ini)
        assert config.wavekit_sdk_path == sdk
        assert config.wavekit_configs_path == configs
        assert config.wavekit_python_path is None  # does not exist: warned, None
        assert any("python.exe path was not found" in r.message for r in caplog.records)
        assert config.wavekit_launcher == f"env WINEPREFIX={tmp_path / 'prefix'} wine"
        assert not hasattr(config, "wavekit_config_path")  # the retired key

    def test_absent_keys_leave_none(self, tmp_path):
        ini = tmp_path / "config.ini"
        _write_ini(ini, f"[Paths]\nGEECS_DATA_LOCAL_BASE_PATH = {tmp_path}\n")
        config = GeecsPathsConfig(config_path=ini)
        assert config.wavekit_sdk_path is None
        assert config.wavekit_python_path is None
        assert config.wavekit_launcher is None
        assert config.wavekit_configs_path is None
