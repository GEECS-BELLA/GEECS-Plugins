"""Integration tests for GEECS-Data-Utils ScanData.

Requires network-mounted GEECS data. Run with:
    pytest -m "integration and data" tests/integration/data/geecs_data_utils/
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.data]


def test_scan_data_loads(canonical_scan):
    """ScanData loads from the data server and returns a non-empty DataFrame."""
    sd = canonical_scan("undulator_2d")
    assert sd.data_frame is not None
    assert len(sd.data_frame) > 0


def test_scan_data_has_standard_columns(canonical_scan):
    """Standard s-file columns are present."""
    sd = canonical_scan("undulator_2d")
    cols = sd.list_columns()
    assert "Bin #" in cols
    assert "Shotnumber" in cols


def test_find_cols(canonical_scan):
    """find_cols returns matches without raising."""
    sd = canonical_scan("undulator_2d")
    results = sd.find_cols("Bin", mode="contains")
    assert len(results) > 0


def test_binning(canonical_scan):
    """Binning produces a MultiIndex DataFrame with center/err columns."""
    sd = canonical_scan("undulator_2d")
    sd.set_binning_config(bin_col="Bin #")
    binned = sd.binned_scalars
    assert binned is not None
    assert len(binned) > 0
    # MultiIndex columns: (col_name, "center") and (col_name, "err_low")
    level1 = binned.columns.get_level_values(1).unique().tolist()
    assert "center" in level1


def test_scan_folder_exists(canonical_scan):
    """The resolved scan folder actually exists on disk."""
    sd = canonical_scan("undulator_2d")
    folder = sd.paths.get_folder()
    assert folder.exists(), f"Scan folder not found: {folder}"
