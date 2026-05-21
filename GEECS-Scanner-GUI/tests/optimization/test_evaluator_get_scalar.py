"""Unit tests for MultiDeviceScanEvaluator.get_scalar key-resolution."""

from __future__ import annotations

import pytest


def _make_evaluator():
    """Return a bare MultiDeviceScanEvaluator bypassing __init__."""
    from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
        MultiDeviceScanEvaluator,
    )

    class _Concrete(MultiDeviceScanEvaluator):
        def compute_objective(self, scalar_results, bin_number):
            return 0.0

    obj = object.__new__(_Concrete)
    obj.output_key = "f"
    obj.bin_number = 1
    obj.current_shot_numbers = []
    return obj


class TestGetScalar:
    """get_scalar must resolve device_name_metric, device_name:metric, then metric."""

    def test_underscore_key(self):
        ev = _make_evaluator()
        result = ev.get_scalar("dev", "sig", {"dev_sig": 3.0})
        assert result == pytest.approx(3.0)

    def test_colon_key(self):
        ev = _make_evaluator()
        result = ev.get_scalar("dev", "sig", {"dev:sig": 7.5})
        assert result == pytest.approx(7.5)

    def test_bare_metric_key(self):
        ev = _make_evaluator()
        result = ev.get_scalar("dev", "sig", {"sig": -1.0})
        assert result == pytest.approx(-1.0)

    def test_underscore_takes_priority_over_colon(self):
        ev = _make_evaluator()
        result = ev.get_scalar(
            "dev", "sig", {"dev_sig": 1.0, "dev:sig": 2.0, "sig": 3.0}
        )
        assert result == pytest.approx(1.0)

    def test_missing_key_raises(self):
        ev = _make_evaluator()
        with pytest.raises(KeyError, match="Could not find metric"):
            ev.get_scalar("dev", "missing", {"other": 9.9})
