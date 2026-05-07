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
# Canonical test scans
# These are stable scans kept on the data server for integration testing.
# Add new entries here when adding new integration test cases.
# ---------------------------------------------------------------------------

CANONICAL_SCANS = {
    "undulator_2d": {
        "experiment": "Undulator",
        "year": 2025,
        "month": 2,
        "day": 20,
        "number": 14,
        "description": "Standard Undulator scan — 2D beam analysis (BeamAnalyzer)",
    },
    "undulator_1d": {
        "experiment": "Undulator",
        "year": 2025,
        "month": 8,
        "day": 19,
        "number": 1,
        "description": "Standard Undulator scan — 1D ICT / spectral analysis",
    },
    "thomson_beam": {
        "experiment": "Thomson",
        "year": 2025,
        "month": 9,
        "day": 24,
        "number": 13,
        "description": "Thomson beamline scan — BeamAnalyzer + MagSpec",
    },
}


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


@pytest.fixture(scope="session")
def canonical_scan(data_root):
    """Factory fixture: return a ScanData for a named canonical test scan.

    Usage in tests::

        def test_something(canonical_scan):
            sd = canonical_scan("undulator_2d")
    """
    from geecs_data_utils import ScanData

    def _factory(name: str):
        if name not in CANONICAL_SCANS:
            raise ValueError(
                f"Unknown canonical scan '{name}'. Available: {list(CANONICAL_SCANS)}"
            )
        info = CANONICAL_SCANS[name]
        return ScanData.from_date(
            year=info["year"],
            month=info["month"],
            day=info["day"],
            number=info["number"],
            experiment=info["experiment"],
            load_scalars=True,
        )

    return _factory


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
