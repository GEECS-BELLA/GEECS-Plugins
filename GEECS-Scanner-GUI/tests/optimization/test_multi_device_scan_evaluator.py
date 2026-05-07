"""Unit tests for MultiDeviceScanEvaluator per_shot support.

All tests use mocks — no live scan data or device connections required.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_result(scalars: dict):
    """Return a mock ImageAnalyzerResult with the given scalars."""
    r = MagicMock()
    r.scalars = scalars
    return r


def _make_analyzer_mock(results_by_key: dict):
    """Return a mock scan analyzer whose .results dict is results_by_key."""
    m = MagicMock()
    m.results = results_by_key
    return m


class _ConcreteEvaluator:
    """Minimal concrete subclass — just negates the 'signal' scalar."""

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        return -scalar_results.get("signal", 0.0)


# ---------------------------------------------------------------------------
# Tests for compute_objective_from_shots (default implementation)
# ---------------------------------------------------------------------------


class TestComputeObjectiveFromShotsDefault:
    """Default compute_objective_from_shots must mean-aggregate then delegate."""

    def _make_evaluator(self):
        """Build a bare MultiDeviceScanEvaluator-like object without touching __init__."""
        from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
            MultiDeviceScanEvaluator,
        )

        class _Concrete(MultiDeviceScanEvaluator):
            def compute_objective(self, scalar_results, bin_number):
                return -scalar_results.get("signal", 0.0)

        # Bypass __init__ entirely — we only test the method logic
        obj = object.__new__(_Concrete)
        obj.output_key = "f"
        obj.bin_number = 1
        obj.current_shot_numbers = [1, 2, 3]
        return obj

    def test_mean_aggregation(self):
        evaluator = self._make_evaluator()
        evaluator.compute_objective = (
            lambda scalar_results, bin_number: -scalar_results["signal"]
        )

        shots = [{"signal": 10.0}, {"signal": 20.0}, {"signal": 30.0}]
        result = evaluator.compute_objective_from_shots(shots, bin_number=1)

        assert result == pytest.approx(-20.0)

    def test_empty_list_returns_zero(self):
        evaluator = self._make_evaluator()
        evaluator.compute_objective = lambda scalar_results, bin_number: 0.0

        result = evaluator.compute_objective_from_shots([], bin_number=1)
        assert result == 0.0

    def test_partial_keys_aggregated_correctly(self):
        """Keys missing from some shots are averaged over shots that have them."""
        evaluator = self._make_evaluator()
        captured = {}

        def capture_objective(scalar_results, bin_number):
            captured.update(scalar_results)
            return 0.0

        evaluator.compute_objective = capture_objective

        shots = [{"a": 10.0, "b": 5.0}, {"a": 20.0}]
        evaluator.compute_objective_from_shots(shots, bin_number=1)

        assert captured["a"] == pytest.approx(15.0)
        assert captured["b"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Tests for _get_value routing
# ---------------------------------------------------------------------------


class TestGetValueRouting:
    """_get_value must route to compute_objective or compute_objective_from_shots."""

    def _make_evaluator_shell(self, analyzer_configs, scan_analyzers):
        """Build a MultiDeviceScanEvaluator bypassing __init__."""
        from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
            MultiDeviceScanEvaluator,
        )

        class _Concrete(MultiDeviceScanEvaluator):
            def compute_objective(self, scalar_results, bin_number):
                return 0.0

        obj = object.__new__(_Concrete)
        obj.analyzer_configs = analyzer_configs
        obj.scan_analyzers = scan_analyzers
        obj.output_key = "f"
        obj.bin_number = 1
        obj.current_shot_numbers = [1, 2, 3]
        obj.current_data_bin = MagicMock()
        obj.scan_tag = MagicMock()
        return obj

    def _make_config(self, device_name: str, mode: str):
        cfg = MagicMock()
        cfg.device_name = device_name
        cfg.analysis_mode = mode
        return cfg

    def test_per_bin_calls_compute_objective(self):
        cfg = self._make_config("dev_a", "per_bin")
        analyzer = _make_analyzer_mock({1: _make_result({"signal": 42.0})})

        evaluator = self._make_evaluator_shell([cfg], {"dev_a": analyzer})

        objective_called_with = {}

        def fake_compute_objective(scalar_results, bin_number):
            objective_called_with.update(scalar_results)
            return -scalar_results["signal"]

        evaluator.compute_objective = fake_compute_objective
        evaluator.compute_observables = lambda **kw: {}

        outputs = evaluator._get_value({})

        assert outputs["f"] == pytest.approx(-42.0)
        assert objective_called_with["signal"] == pytest.approx(42.0)

    def test_per_shot_calls_compute_objective_from_shots(self):
        cfg = self._make_config("dev_b", "per_shot")
        analyzer = _make_analyzer_mock(
            {
                1: _make_result({"signal": 10.0}),
                2: _make_result({"signal": 20.0}),
                3: _make_result({"signal": 30.0}),
            }
        )

        evaluator = self._make_evaluator_shell([cfg], {"dev_b": analyzer})

        received_list = []

        def fake_from_shots(scalar_results_list, bin_number):
            received_list.extend(scalar_results_list)
            return -np.median([d["signal"] for d in scalar_results_list])

        evaluator.compute_objective_from_shots = fake_from_shots
        evaluator.compute_observables = lambda **kw: {}

        outputs = evaluator._get_value({})

        assert outputs["f"] == pytest.approx(-20.0)
        assert len(received_list) == 3

    def test_per_shot_dict_return_passthrough(self):
        """compute_objective_from_shots returning a dict (with f_noise) is passed through."""
        cfg = self._make_config("dev_c", "per_shot")
        analyzer = _make_analyzer_mock(
            {
                1: _make_result({"signal": 10.0}),
                2: _make_result({"signal": 30.0}),
                3: _make_result({"signal": 20.0}),
            }
        )

        evaluator = self._make_evaluator_shell([cfg], {"dev_c": analyzer})

        def fake_from_shots(scalar_results_list, bin_number):
            vals = [d["signal"] for d in scalar_results_list]
            return {"f": -float(np.median(vals)), "f_noise": float(np.std(vals))}

        evaluator.compute_objective_from_shots = fake_from_shots
        evaluator.compute_observables = lambda **kw: {}

        outputs = evaluator._get_value({})

        assert outputs["f"] == pytest.approx(-20.0)
        assert "f_noise" in outputs
        assert outputs["f_noise"] > 0

    def test_mixed_mode_per_bin_scalars_merged_into_shots(self):
        """Per_bin scalars are available in every shot dict for mixed-mode setups."""
        cfg_bin = self._make_config("dev_bin", "per_bin")
        cfg_shot = self._make_config("dev_shot", "per_shot")

        analyzer_bin = _make_analyzer_mock({1: _make_result({"background": 5.0})})
        analyzer_shot = _make_analyzer_mock(
            {
                1: _make_result({"signal": 10.0}),
                2: _make_result({"signal": 20.0}),
                3: _make_result({"signal": 30.0}),
            }
        )

        evaluator = self._make_evaluator_shell(
            [cfg_bin, cfg_shot],
            {"dev_bin": analyzer_bin, "dev_shot": analyzer_shot},
        )

        received_list = []

        def fake_from_shots(scalar_results_list, bin_number):
            received_list.extend(scalar_results_list)
            return 0.0

        evaluator.compute_objective_from_shots = fake_from_shots
        evaluator.compute_observables = lambda **kw: {}

        evaluator._get_value({})

        assert all("background" in d for d in received_list), (
            "per_bin scalar 'background' should be present in every shot dict"
        )
        assert all(d["background"] == pytest.approx(5.0) for d in received_list)

    def test_per_bin_missing_result_logs_warning(self, caplog):
        cfg = self._make_config("dev_x", "per_bin")
        analyzer = _make_analyzer_mock({})  # no result for bin 1

        evaluator = self._make_evaluator_shell([cfg], {"dev_x": analyzer})
        evaluator.compute_objective = lambda scalar_results, bin_number: 0.0
        evaluator.compute_observables = lambda **kw: {}

        import logging

        with caplog.at_level(logging.WARNING):
            evaluator._get_value({})

        assert any("No per_bin result" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests for compute_observables key shadowing
# ---------------------------------------------------------------------------


class TestObservablesShadowing:
    """compute_observables must not overwrite the objective key in outputs."""

    def _make_shadowing_evaluator(self, objective_value: float = 42.0):
        from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
            MultiDeviceScanEvaluator,
        )

        class _Shadowing(MultiDeviceScanEvaluator):
            def compute_objective(self, scalar_results, bin_number):
                return objective_value

            def compute_observables(self, scalar_results, bin_number):
                # Accidentally returns "f", the same key as output_key
                return {"f": 999.0, "other": 1.0}

        obj = object.__new__(_Shadowing)
        obj.analyzer_configs = []
        obj.scan_analyzers = {}
        obj.output_key = "f"
        obj.objective_tag = "test"
        obj.bin_number = 1
        obj.current_shot_numbers = [1]
        obj.current_data_bin = MagicMock()
        obj.scan_tag = None
        return obj

    def test_objective_not_overwritten(self, caplog):
        import logging

        ev = self._make_shadowing_evaluator(objective_value=42.0)
        with caplog.at_level(logging.WARNING):
            result = ev._get_value({})

        assert result["f"] == pytest.approx(42.0)

    def test_other_observables_still_present(self, caplog):
        import logging

        ev = self._make_shadowing_evaluator()
        with caplog.at_level(logging.WARNING):
            result = ev._get_value({})

        assert "other" in result
        assert result["other"] == pytest.approx(1.0)

    def test_warning_logged(self, caplog):
        import logging

        ev = self._make_shadowing_evaluator()
        with caplog.at_level(logging.WARNING):
            ev._get_value({})

        assert any(
            "compute_observables returned objective key" in r.message
            for r in caplog.records
        )
