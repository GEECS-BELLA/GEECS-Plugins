"""Tests for ScanOptions Pydantic model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from geecs_scanner.engine.models.scan_options import ScanOptions


class TestScanOptionsDefaults:
    """All fields must have sensible defaults so construction needs no args."""

    def test_construct_no_args(self):
        opts = ScanOptions()
        assert opts.rep_rate_hz == 1.0
        assert opts.enable_global_time_sync is False
        assert opts.global_time_tolerance_ms == 0

    def test_construct_with_kwargs(self):
        opts = ScanOptions(
            rep_rate_hz=10.0,
            enable_global_time_sync=True,
        )
        assert opts.rep_rate_hz == 10.0
        assert opts.enable_global_time_sync is True

    def test_g1_orphaned_knobs_are_gone(self):
        """The dead post-G1 options were removed, not silently ignored (#535)."""
        for field in (
            "master_control_ip",
            "on_shot_tdms",
            "save_direct_on_network",
            "randomized_beeps",
        ):
            assert field not in ScanOptions.model_fields


class TestScanOptionsValidation:
    """Field validators must enforce constraints."""

    def test_rep_rate_must_be_positive(self):
        with pytest.raises(ValidationError):
            ScanOptions(rep_rate_hz=0.0)

    def test_rep_rate_negative_rejected(self):
        with pytest.raises(ValidationError):
            ScanOptions(rep_rate_hz=-5.0)

    def test_rep_rate_positive_accepted(self):
        opts = ScanOptions(rep_rate_hz=0.001)
        assert opts.rep_rate_hz == pytest.approx(0.001)

    def test_tolerance_clamped_to_zero(self):
        opts = ScanOptions(global_time_tolerance_ms=-100)
        assert opts.global_time_tolerance_ms == 0

    def test_tolerance_clamped_to_max(self):
        opts = ScanOptions(global_time_tolerance_ms=999_999)
        assert opts.global_time_tolerance_ms == 60_000

    def test_tolerance_in_range_unchanged(self):
        opts = ScanOptions(global_time_tolerance_ms=500)
        assert opts.global_time_tolerance_ms == 500


class TestScanOptionsSerialisation:
    """model_dump / model_validate round-trip must be lossless."""

    def test_round_trip(self):
        original = ScanOptions(
            rep_rate_hz=5.0,
            enable_global_time_sync=True,
            global_time_tolerance_ms=200,
        )
        dumped = original.model_dump()
        recovered = ScanOptions.model_validate(dumped)
        assert recovered == original
