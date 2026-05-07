"""Unit tests for concrete MultiDeviceScanEvaluator subclasses.

Uses ImageAnalyzerResult with synthetic scalar dicts — no image files,
no scan-analyzer plumbing, no device connections required.

The seam being tested is: given an ImageAnalyzerResult with known scalars,
does each evaluator compute the right objective and observables?
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fake_cfg(device_name: str, mode: str = "per_bin"):
    """Minimal SingleDeviceScanAnalyzerConfig stand-in."""
    cfg = MagicMock()
    cfg.device_name = device_name
    cfg.analysis_mode = mode
    return cfg


def _fake_analyzer(results_by_key: dict):
    """Scan analyzer whose .results dict is provided."""
    m = MagicMock()
    m.results = results_by_key
    return m


@dataclass
class _Result:
    """Minimal stand-in for ImageAnalyzerResult — evaluators only access .scalars."""

    scalars: dict


# ---------------------------------------------------------------------------
# BeamSizeEvaluator
# ---------------------------------------------------------------------------


class TestBeamSizeEvaluator:
    """BeamSizeEvaluator must compute quadrature sum objective with calibration."""

    def _make_shell(self, device_name: str = "UC_Cam", calibration: float = 1.0):
        from geecs_scanner.optimization.evaluators.beam_size_evaluator import (
            BeamSizeEvaluator,
        )

        obj = object.__new__(BeamSizeEvaluator)
        obj.analyzer_configs = [_fake_cfg(device_name)]
        obj.scan_analyzers = {}
        obj.calibration = calibration
        obj.output_key = "f"
        obj.objective_tag = "BeamSize"
        obj.bin_number = 1
        obj.current_shot_numbers = [1]
        obj.current_data_bin = MagicMock()
        obj.scan_tag = None
        return obj

    # compute_objective -------------------------------------------------------

    def test_quadrature_sum(self):
        ev = self._make_shell(calibration=2.0)
        # (3 * 2)² + (4 * 2)² = 36 + 64 = 100
        result = ev.compute_objective({"x_fwhm": 3.0, "y_fwhm": 4.0}, bin_number=1)
        assert result == pytest.approx(100.0)

    def test_calibration_scales_both_axes(self):
        ev = self._make_shell(calibration=0.5)
        # (10 * 0.5)² + (10 * 0.5)² = 25 + 25 = 50
        result = ev.compute_objective({"x_fwhm": 10.0, "y_fwhm": 10.0}, bin_number=1)
        assert result == pytest.approx(50.0)

    def test_unit_calibration_is_pixel_quadrature(self):
        ev = self._make_shell(calibration=1.0)
        result = ev.compute_objective({"x_fwhm": 3.0, "y_fwhm": 4.0}, bin_number=1)
        assert result == pytest.approx(25.0)

    # compute_observables -----------------------------------------------------

    def test_pixel_values_unchanged(self):
        ev = self._make_shell(calibration=2.0)
        obs = ev.compute_observables({"x_fwhm": 3.0, "y_fwhm": 4.0}, bin_number=1)
        assert obs["x_fwhm_px"] == pytest.approx(3.0)
        assert obs["y_fwhm_px"] == pytest.approx(4.0)

    def test_calibrated_values_scaled(self):
        ev = self._make_shell(calibration=2.0)
        obs = ev.compute_observables({"x_fwhm": 3.0, "y_fwhm": 4.0}, bin_number=1)
        assert obs["x_fwhm_units"] == pytest.approx(6.0)
        assert obs["y_fwhm_units"] == pytest.approx(8.0)

    def test_quadrature_observable_matches_objective(self):
        ev = self._make_shell(calibration=2.0)
        obs = ev.compute_observables({"x_fwhm": 3.0, "y_fwhm": 4.0}, bin_number=1)
        objective = ev.compute_objective({"x_fwhm": 3.0, "y_fwhm": 4.0}, bin_number=1)
        assert obs["size_quadrature_units2"] == pytest.approx(objective)

    # end-to-end through _get_value -------------------------------------------

    def test_get_value_routes_image_analyzer_result_scalars(self):
        """ImageAnalyzerResult.scalars must flow through _get_value to compute_objective."""
        ev = self._make_shell(device_name="UC_Cam", calibration=1.0)
        ev.scan_analyzers = {
            "UC_Cam": _fake_analyzer({1: _Result({"x_fwhm": 3.0, "y_fwhm": 4.0})})
        }
        outputs = ev._get_value({})
        assert outputs["f"] == pytest.approx(25.0)
        assert "x_fwhm_px" in outputs  # observables also present


# ---------------------------------------------------------------------------
# MaxCountsEvaluator
# ---------------------------------------------------------------------------


class TestMaxCountsEvaluator:
    """MaxCountsEvaluator must negate image_total and pass centroid/peak through."""

    def _make_shell(self, device_name: str = "UC_Cam"):
        from geecs_scanner.optimization.evaluators.beam_sum_counts_evaluator import (
            MaxCountsEvaluator,
        )

        obj = object.__new__(MaxCountsEvaluator)
        obj.analyzer_configs = [_fake_cfg(device_name)]
        obj.scan_analyzers = {}
        obj.output_key = "f"
        obj.objective_tag = "TotalCounts"
        obj.bin_number = 1
        obj.current_shot_numbers = [1]
        obj.current_data_bin = MagicMock()
        obj.scan_tag = None
        return obj

    def test_compute_objective_negates_total(self):
        ev = self._make_shell()
        assert ev.compute_objective(
            {"image_total": 5000.0}, bin_number=1
        ) == pytest.approx(-5000.0)

    def test_compute_objective_zero_total(self):
        ev = self._make_shell()
        assert ev.compute_objective(
            {"image_total": 0.0}, bin_number=1
        ) == pytest.approx(0.0)

    def test_compute_observables_values_passed_through(self):
        ev = self._make_shell()
        scalars = {"x_CoM": 100.0, "y_CoM": 200.0, "image_peak_value": 3000.0}
        obs = ev.compute_observables(scalars, bin_number=1)
        assert obs["x_CoM"] == pytest.approx(100.0)
        assert obs["y_CoM"] == pytest.approx(200.0)
        assert obs["image_peak_value"] == pytest.approx(3000.0)

    def test_compute_observables_exact_key_set(self):
        ev = self._make_shell()
        obs = ev.compute_observables(
            {"x_CoM": 1.0, "y_CoM": 2.0, "image_peak_value": 3.0, "image_total": 4.0},
            bin_number=1,
        )
        assert set(obs.keys()) == {"x_CoM", "y_CoM", "image_peak_value"}

    def test_get_value_routes_image_analyzer_result_scalars(self):
        ev = self._make_shell(device_name="UC_Cam")
        ev.scan_analyzers = {
            "UC_Cam": _fake_analyzer(
                {
                    1: _Result(
                        {
                            "image_total": 5000.0,
                            "x_CoM": 100.0,
                            "y_CoM": 200.0,
                            "image_peak_value": 3000.0,
                        }
                    )
                }
            )
        }
        outputs = ev._get_value({})
        assert outputs["f"] == pytest.approx(-5000.0)
        assert outputs["x_CoM"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# BeamPositionEvaluator
# ---------------------------------------------------------------------------


class TestBeamPositionEvaluator:
    """BeamPositionEvaluator must extract observables from scalar_results by name."""

    def _make_shell(self, observable_names=None, device_name: str = "UC_Cam"):
        from geecs_scanner.optimization.evaluators.beam_position_evaluator import (
            BeamPositionEvaluator,
        )

        obj = object.__new__(BeamPositionEvaluator)
        obj.analyzer_configs = [_fake_cfg(device_name)]
        obj.observable_names = (
            observable_names if observable_names is not None else ["x_CoM"]
        )
        obj.output_key = None  # BAX — observables only
        obj.objective_tag = "BeamPosition"
        return obj

    def test_single_observable_happy_path(self):
        ev = self._make_shell(["x_CoM"])
        obs = ev.compute_observables({"x_CoM": 42.0}, bin_number=1)
        assert obs == {"x_CoM": pytest.approx(42.0)}

    def test_two_observables_both_extracted(self):
        ev = self._make_shell(["x_CoM", "y_CoM"])
        obs = ev.compute_observables({"x_CoM": 5.0, "y_CoM": -3.0}, bin_number=1)
        assert obs["x_CoM"] == pytest.approx(5.0)
        assert obs["y_CoM"] == pytest.approx(-3.0)

    def test_partial_failure_logs_warning_and_returns_partial(self, caplog):
        """Missing observable → warning logged, available observables still returned."""
        import logging

        ev = self._make_shell(["x_CoM", "y_CoM"])
        with caplog.at_level(logging.WARNING):
            obs = ev.compute_observables({"x_CoM": 5.0}, bin_number=1)

        assert "x_CoM" in obs
        assert "y_CoM" not in obs
        assert any("y_CoM" in r.message for r in caplog.records)

    def test_all_missing_returns_empty(self, caplog):
        import logging

        ev = self._make_shell(["x_CoM"])
        with caplog.at_level(logging.WARNING):
            obs = ev.compute_observables({}, bin_number=1)

        assert obs == {}


# ---------------------------------------------------------------------------
# BeamPositionSimulationEvaluator
# ---------------------------------------------------------------------------


class TestBeamPositionSimulationEvaluator:
    """Simulation evaluator must compute x_CoM from control/measurement setpoints."""

    def _make_shell(
        self,
        scale_factor: float = 1000.0,
        reference_setpoint: float = 1.0,
        noise_amplitude: float = 0.0,
        calibration: float = 1.0,
    ):
        from geecs_scanner.optimization.evaluators.beam_position_evaluator_simulation import (
            BeamPositionSimulationEvaluator,
        )

        obj = object.__new__(BeamPositionSimulationEvaluator)
        obj.scale_factor = scale_factor
        obj.reference_setpoint = reference_setpoint
        obj.noise_amplitude = noise_amplitude
        obj.calibration = calibration
        obj.control_variable_name = "U_S1H:Current"
        obj.measurement_variable_name = "U_EMQTripletBipolar:Current_Limit.Ch1"
        obj.output_key = None
        obj.objective_tag = "BeamPosition"
        return obj

    def _bin(self, control_val: float, measure_val: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "U_S1H:Current": [control_val],
                "U_EMQTripletBipolar:Current_Limit.Ch1": [measure_val],
            }
        )

    def test_zero_offset_gives_zero_centroid(self):
        # control == reference → offset = 0 → x_CoM = 0 regardless of measure
        ev = self._make_shell(scale_factor=1000.0, reference_setpoint=2.0)
        ev.current_data_bin = self._bin(2.0, 0.5)
        obs = ev.compute_observables({}, bin_number=1)
        assert obs["x_CoM"] == pytest.approx(0.0)

    def test_centroid_formula_known_values(self):
        # offset = 2.0 - 1.0 = 1.0, measure = 0.5
        # x_CoM = 1000 * 1.0 * (1 + 0.5) = 1500
        ev = self._make_shell(scale_factor=1000.0, reference_setpoint=1.0)
        ev.current_data_bin = self._bin(2.0, 0.5)
        obs = ev.compute_observables({}, bin_number=1)
        assert obs["x_CoM"] == pytest.approx(1500.0)

    def test_negative_offset_gives_negative_centroid(self):
        # offset = 0.5 - 1.0 = -0.5, measure = 0
        # x_CoM = 1000 * -0.5 * (1 + 0) = -500
        ev = self._make_shell(scale_factor=1000.0, reference_setpoint=1.0)
        ev.current_data_bin = self._bin(0.5, 0.0)
        obs = ev.compute_observables({}, bin_number=1)
        assert obs["x_CoM"] == pytest.approx(-500.0)

    def test_calibration_scales_output(self):
        # x_CoM before calibration = 1500, calibration = 0.001 → 1.5
        ev = self._make_shell(
            scale_factor=1000.0, reference_setpoint=1.0, calibration=0.001
        )
        ev.current_data_bin = self._bin(2.0, 0.5)
        obs = ev.compute_observables({}, bin_number=1)
        assert obs["x_CoM"] == pytest.approx(1.5)

    def test_missing_column_returns_empty_with_warning(self, caplog):
        import logging

        ev = self._make_shell()
        ev.current_data_bin = pd.DataFrame({"wrong_column": [1.0]})
        with caplog.at_level(logging.WARNING):
            obs = ev.compute_observables({}, bin_number=1)
        assert obs == {}
        assert len(caplog.records) > 0

    def test_multiple_shots_averaged(self):
        # Two shots: control [1.5, 2.5] → mean = 2.0, offset = 1.0; measure [0, 0] → x_CoM = 1000
        ev = self._make_shell(scale_factor=1000.0, reference_setpoint=1.0)
        ev.current_data_bin = pd.DataFrame(
            {
                "U_S1H:Current": [1.5, 2.5],
                "U_EMQTripletBipolar:Current_Limit.Ch1": [0.0, 0.0],
            }
        )
        obs = ev.compute_observables({}, bin_number=1)
        assert obs["x_CoM"] == pytest.approx(1000.0)
