"""Shared pytest fixtures for GEECS-Plugins integration tests.

Usage
-----
Run only unit tests (CI):
    pytest -m "not integration"

Run data integration tests (networked machine):
    pytest -m "integration and data"

Run hardware integration tests (in the lab):
    pytest -m "integration and hardware"
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Data availability fixture
# ---------------------------------------------------------------------------


def _resolve_data_root() -> Path | None:
    """Return the GEECS data root if reachable, else None."""
    try:
        from geecs_data_utils import GeecsPathsConfig

        cfg = GeecsPathsConfig()
        path = Path(cfg.get_base_path("Undulator")).parent
        if path.exists():
            return path
    except Exception:
        pass

    # Fallback: check common mount points directly
    for candidate in [
        Path("/Volumes/hdna2/data"),
        Path("Z:/data"),
        Path("/mnt/data"),
    ]:
        if candidate.exists():
            return candidate

    return None


@pytest.fixture(scope="session")
def data_root() -> Path:
    """Path to the GEECS data root.

    Skips the test if the network data drive is not mounted.
    """
    root = _resolve_data_root()
    if root is None:
        pytest.skip(
            "GEECS data drive not mounted — skipping data integration test. "
            "Run on a networked machine with data access."
        )
    return root


# ---------------------------------------------------------------------------
# Image analysis config initialisation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _init_image_analysis_config():
    """Set the image_analysis config base directory for the test session.

    Resolves the path via ScanPaths (same as the example notebooks).  If the
    configs repo is not present the fixture is a no-op — integration tests
    that actually need YAML configs will fail with a clear ValueError rather
    than a silent skip, since those tests are only run on networked machines
    that should also have the configs repo checked out.
    """
    try:
        from geecs_data_utils import ScanPaths
        from geecs_data_utils.config_roots import image_analysis_config

        config_path = ScanPaths.paths_config.image_analysis_configs_path
        if config_path and Path(config_path).exists():
            image_analysis_config.set_base_dir(config_path)
    except Exception:
        pass  # CI / offline: no configs repo; integration tests are already skipped


# ---------------------------------------------------------------------------
# Hardware availability fixture (stub — populated in a future branch)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def hardware_available():
    """Skip the test if live GEECS hardware is not reachable.

    This fixture is a stub. Hardware detection logic will be added when
    hardware integration tests are introduced (simulated and real devices).
    """
    pytest.skip(
        "Hardware integration tests not yet implemented. "
        "See: https://github.com/GEECS-BELLA/GEECS-Plugins/issues/349"
    )
