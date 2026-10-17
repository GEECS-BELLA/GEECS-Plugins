"""Shared pytest fixtures for ScanAnalysis tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import numpy as np
import pandas as pd

from scan_analysis.route_compare import (
    GOLDEN_RTOL,
    Golden,
    compare_snapshots,
    read_golden,
    write_golden,
)

#: The committed golden trees the differential tests pin against: one folder
#: per case (``manifest.json`` + ``arrays.npz``), generated from the core
#: route while the legacy analyzers were still present and matched (#1039,
#: tag ``legacy-analysis-final``). Regenerate deliberately with
#: ``GEECS_UPDATE_GOLDENS=1 pytest ScanAnalysis/tests/test_core_*.py``.
GOLDEN_ROOT = Path(__file__).parent / "data" / "golden"
UPDATE_GOLDENS = "GEECS_UPDATE_GOLDENS"

# The legacy scan wrappers render per-bin figures from worker threads, which
# the interactive macOS backend cannot do (it aborts the interpreter). CI is
# headless already; this keeps a developer's local run headless too.
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def golden():
    """Return the stored tree for ``name``, or store ``snapshot`` as it when updating.

    ``golden(name, snapshot, **extra)``: with ``GEECS_UPDATE_GOLDENS`` set the
    snapshot (and the keyword notes) is written under ``tests/data/golden/<name>``
    and handed back as the golden, so the same assertions that follow then
    hold trivially for the writer and bite for everyone else.
    ``golden(name)`` only reads.
    """

    def resolve(
        name: str, snapshot: dict[str, tuple] | None = None, **extra: Any
    ) -> Golden:
        folder = GOLDEN_ROOT / name
        if snapshot is not None and os.environ.get(UPDATE_GOLDENS):
            write_golden(snapshot, folder, extra=extra)
        return read_golden(folder)

    return resolve


def measurement_snapshot(data: Any, scalars: dict[str, float]) -> dict[str, tuple]:
    """One measurement (its frame data and scalars) in ``compare_snapshots``' shape."""
    data = np.asarray(data)
    if data.dtype.kind == "f":  # values, not storage width (legacy kept float32 traces)
        data = data.astype(np.float64)
    table = pd.DataFrame({key: [float(v)] for key, v in scalars.items()})
    return {"frame": ("h5", "data", data, data.dtype), "scalars": ("table", table)}


@pytest.fixture
def measurement_golden(golden):
    """Check one measurement (frame data and scalars) against the stored golden ``name``.

    The stored values came from the legacy oracle while it was present
    (#1039); ``GEECS_UPDATE_GOLDENS`` now stores the core's. Floats are held
    to ``GOLDEN_RTOL`` of their own scale.
    """

    def check(name: str, data: Any, scalars: dict[str, float]) -> None:
        core = measurement_snapshot(data, scalars)
        problems = compare_snapshots(
            golden(name, core).files, core, average_ulps=0, rtol=GOLDEN_RTOL
        )
        assert problems == [], problems

    return check


@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication instance for Qt widget tests.

    Sets QT_QPA_PLATFORM=offscreen so tests run on headless servers.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
