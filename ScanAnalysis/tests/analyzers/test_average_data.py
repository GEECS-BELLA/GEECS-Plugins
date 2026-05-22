"""Tests for the shared ``SingleDeviceScanAnalyzer.average_data`` helper.

The helper is called from the noscan post-processing path in 1D and 2D
analyzers. It must tolerate inhomogeneous per-shot lineouts (real case:
FROG spectral phase, where ROI/weight masking yields different sample
counts per shot) rather than letting ``np.mean`` raise.
"""

from __future__ import annotations

import numpy as np

from scan_analysis.analyzers.common.single_device_scan_analyzer import (
    SingleDeviceScanAnalyzer,
)


def test_returns_none_for_empty_list():
    assert SingleDeviceScanAnalyzer.average_data([]) is None


def test_averages_homogeneous_arrays():
    arrays = [np.ones((3, 2)) * i for i in range(1, 4)]
    result = SingleDeviceScanAnalyzer.average_data(arrays)

    assert result is not None
    np.testing.assert_allclose(result, np.full((3, 2), 2.0))


def test_returns_none_for_inhomogeneous_shapes(caplog):
    arrays = [np.zeros((5, 2)), np.zeros((7, 2)), np.zeros((6, 2))]

    with caplog.at_level("WARNING"):
        result = SingleDeviceScanAnalyzer.average_data(arrays)

    assert result is None
    assert any("inhomogeneous" in record.message for record in caplog.records)
