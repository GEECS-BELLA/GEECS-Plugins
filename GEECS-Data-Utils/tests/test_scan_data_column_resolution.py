"""Regression tests for ScanData.find_cols / resolve_col delegation.

These guard the refactor that moved column resolution out of
:mod:`geecs_data_utils.scan_data` and into
:mod:`geecs_data_utils.data.columns`. The integration test suite under
``tests/integration/data/geecs_data_utils/`` requires real scan data on the
network mount; these unit tests cover the same surface against synthetic
DataFrames so the delegation can be verified in CI without I/O.

Specifically locks behavior that's preserved at the ScanData level (and is
not directly tested by ``test_columns.py`` against the new module):

* No-frame path returns empty list / raises clearly.
* The local-alias short-circuit in ``resolve_col`` runs before column
  matching.
* MultiIndex columns are flattened with ``" "`` before matching (matches
  the legacy behavior).
* Ambiguity logging fires when more than one column resolves and a
  candidate is picked deterministically.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from geecs_data_utils.scan_data import ScanData


def _scan_data_with(df: pd.DataFrame) -> ScanData:
    """Build a ScanData with a pre-set data_frame, bypassing path resolution.

    ScanData.__init__ requires a ScanPaths object and would try to resolve
    a folder on disk. The column-resolution methods only touch
    ``data_frame`` and ``column_aliases``, so for unit-level tests we
    construct the instance directly without ScanPaths.
    """
    sd = ScanData.__new__(ScanData)
    sd.data_frame = df
    sd.column_aliases = {}
    return sd


class TestFindColsDelegation:
    """ScanData.find_cols continues to delegate to data.columns.find_cols."""

    def test_returns_empty_when_no_frame_loaded(self):
        sd = ScanData.__new__(ScanData)
        sd.data_frame = None
        sd.column_aliases = {}
        assert sd.find_cols("anything") == []

    def test_basic_contains_match(self):
        df = pd.DataFrame(
            {"Magnet Current": [1.0], "Magnet Voltage": [2.0], "Charge": [3.0]}
        )
        sd = _scan_data_with(df)
        result = sd.find_cols("magnet")
        assert "Magnet Current" in result
        assert "Magnet Voltage" in result
        assert "Charge" not in result

    def test_case_insensitive_by_default(self):
        df = pd.DataFrame({"ChargeAlias": [1.0]})
        sd = _scan_data_with(df)
        assert sd.find_cols("charge") == ["ChargeAlias"]

    def test_multiindex_columns_are_flattened(self):
        """MultiIndex columns get joined with ' ' before matching."""
        df = pd.DataFrame({("MagSpec", "charge"): [1.0], ("Camera", "centroid"): [2.0]})
        sd = _scan_data_with(df)
        # The flat representation is "MagSpec charge" / "Camera centroid";
        # a search for "magspec" should match the first column.
        result = sd.find_cols("magspec")
        assert len(result) == 1
        assert "MagSpec" in result[0] and "charge" in result[0]


class TestResolveColDelegation:
    """ScanData.resolve_col continues to delegate (with alias short-circuit)."""

    def test_raises_when_no_frame_loaded(self):
        sd = ScanData.__new__(ScanData)
        sd.data_frame = None
        sd.column_aliases = {}
        with pytest.raises(ValueError, match="No scalar dataframe loaded"):
            sd.resolve_col("anything")

    def test_local_alias_short_circuits_resolution(self):
        """add_local_alias bypasses the matching logic entirely."""
        df = pd.DataFrame({"U_BCaveICT Charge Alias: charge_BCave": [1.0]})
        sd = _scan_data_with(df)
        sd.add_local_alias("charge", "U_BCaveICT Charge Alias: charge_BCave")

        # Even though "charge" wouldn't substring-match this column's flat
        # name in the case-sensitive sense, the alias should be used.
        assert sd.resolve_col("charge") == "U_BCaveICT Charge Alias: charge_BCave"

    def test_exact_case_insensitive_preferred_over_substring(self):
        """When both an exact-CI match and a substring match exist, exact wins."""
        df = pd.DataFrame({"Charge": [1.0], "ChargeAlias": [2.0]})
        sd = _scan_data_with(df)
        assert sd.resolve_col("charge") == "Charge"

    def test_unmatched_spec_raises_value_error(self):
        df = pd.DataFrame({"Charge": [1.0]})
        sd = _scan_data_with(df)
        with pytest.raises(ValueError):
            sd.resolve_col("nonexistent_column")

    def test_ambiguity_emits_warning(self, caplog):
        """Multiple substring matches log a warning naming the candidates."""
        df = pd.DataFrame({"Magnet Current": [1.0], "Magnet Voltage": [2.0]})
        sd = _scan_data_with(df)
        with caplog.at_level(logging.WARNING):
            picked = sd.resolve_col("magnet")
        # A column is still chosen deterministically.
        assert picked in ("Magnet Current", "Magnet Voltage")
        warning_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("matched multiple columns" in m for m in warning_msgs)
