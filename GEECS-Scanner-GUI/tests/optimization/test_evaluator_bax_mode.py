"""Unit tests for BAX (observables-only) evaluator mode.

When output_key is None, get_value must not enforce objective-key presence and
must pass observables through cleanly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.optimization.conftest import FakeDataLogger, make_log_entries


def _make_bax_evaluator(observable_results: dict):
    """
    Build a MultiDeviceScanEvaluator shell with output_key=None.

    compute_observables returns *observable_results*.
    _get_value is patched to bypass real analysis.
    """
    from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
        MultiDeviceScanEvaluator,
    )

    class _BaxConcrete(MultiDeviceScanEvaluator):
        def compute_objective(self, scalar_results, bin_number):
            # Should never be called in BAX mode
            raise AssertionError(
                "compute_objective must not be called when output_key=None"
            )

        def compute_observables(self, scalar_results, bin_number):
            return observable_results

    obj = object.__new__(_BaxConcrete)
    obj.analyzer_configs = []
    obj.scan_analyzers = {}
    obj.output_key = None
    obj.objective_tag = "BAX"
    obj.bin_number = 1
    obj.current_shot_numbers = [1, 2]
    obj.current_data_bin = MagicMock()
    obj.scan_tag = None
    return obj


class TestBaxMode:
    """output_key=None evaluators must return only observables without objective logic."""

    def test_observables_returned(self):
        ev = _make_bax_evaluator({"x_CoM": 3.7})
        result = ev._get_value({})
        assert "x_CoM" in result
        assert result["x_CoM"] == pytest.approx(3.7)

    def test_no_f_key_in_output(self):
        ev = _make_bax_evaluator({"x_CoM": 1.0, "y_CoM": 2.0})
        result = ev._get_value({})
        assert "f" not in result

    def test_get_value_accepts_no_objective_key(self):
        """Full get_value pipeline must not raise when output_key=None."""
        from geecs_scanner.optimization.base_evaluator import BaseEvaluator

        entries = make_log_entries(n_shots=2, bin_num=1)
        data_logger = FakeDataLogger(entries, bin_num=1)

        class _Obs(BaseEvaluator):
            def _get_value(self, input_data):
                return {"x_CoM": 5.0}

        obj = _Obs.__new__(_Obs)
        obj.device_requirements = {}
        obj.scan_data_manager = None
        obj.data_logger = data_logger
        obj.bin_number = 0
        obj.log_df = None
        obj.current_data_bin = None
        obj.current_shot_numbers = None
        obj.objective_tag = "BAX"
        obj.output_key = None
        obj.scan_tag = None

        result = obj.get_value({})
        assert result == {"x_CoM": pytest.approx(5.0)}

    def test_multiple_observables_all_returned(self):
        ev = _make_bax_evaluator({"x_CoM": -2.0, "y_CoM": 4.5, "charge": 100.0})
        result = ev._get_value({})
        assert set(result.keys()) == {"x_CoM", "y_CoM", "charge"}
