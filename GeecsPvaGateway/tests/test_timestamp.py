"""Pure-unit tests for the frame-timestamp ladder (no sockets — unmarked)."""

from __future__ import annotations


class TestFrameTimestampPlausibility:
    """Plausibility is checked post-epoch-conversion (PV_CONTRACT parity)."""

    def test_valid_labview_timestamp_converts(self) -> None:
        from geecs_pva_gateway.server import _LABVIEW_EPOCH_OFFSET, _frame_timestamp

        lv = _LABVIEW_EPOCH_OFFSET + 1_000_000.5
        assert _frame_timestamp({"acq_timestamp": lv}) == 1_000_000.5

    def test_small_positive_labview_value_falls_through(self) -> None:
        """A value in (0, offset] must yield receive time, never negative."""
        import time as _time

        from geecs_pva_gateway.server import _frame_timestamp

        before = _time.time()
        ts = _frame_timestamp({"acq_timestamp": 12345.0, "systimestamp": 99.0})
        assert ts >= before  # receive-time fallback, not a negative stamp

    def test_ladder_prefers_acq_then_sys(self) -> None:
        from geecs_pva_gateway.server import _LABVIEW_EPOCH_OFFSET, _frame_timestamp

        assert (
            _frame_timestamp(
                {
                    "acq_timestamp": 1.0,  # implausible → skipped
                    "systimestamp": _LABVIEW_EPOCH_OFFSET + 42.0,
                }
            )
            == 42.0
        )
