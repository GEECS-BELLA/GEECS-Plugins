"""Unit tests for concrete BaseEvaluator subclasses.

All evaluators inherit from :class:`BaseEvaluator` directly. The scalars
dict they receive in ``compute_objective`` / ``compute_observables`` uses
device-prefixed keys for analyzer outputs (``"UC_Cam_x_fwhm"``) and bare
column names for s-file scalars (``"U_Laser:Energy"``).

Analyzer-side keys are emitted *already prefixed* by ImageAnalysis today
(via ``camera_config.name``); `BaseEvaluator._get_value` forwards them
through unchanged. That contract is pinned by
:class:`TestAnalyzerScalarPassthrough` below.

Tests bypass ``__init__`` via ``object.__new__`` so they don't need a live
ScanPaths / load_diagnostic round-trip; minimal attributes are stamped on
the instance directly.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fake_diag(device_name: str):
    """Minimal DiagnosticAnalysisConfig stand-in with ``.name`` set."""
    diag = MagicMock()
    diag.name = device_name
    return diag


# ---------------------------------------------------------------------------
# BeamSizeEvaluator
# ---------------------------------------------------------------------------


class TestBeamSizeEvaluator:
    """Quadrature sum of calibrated FWHMs; values prefixed by device name."""

    def _make_shell(self, device_name: str = "UC_Cam", calibration: float = 1.0):
        from geecs_scanner.optimization.evaluators.beam_size_evaluator import (
            BeamSizeEvaluator,
        )

        obj = object.__new__(BeamSizeEvaluator)
        obj.diagnostics = [_fake_diag(device_name)]
        obj.scan_analyzers = {}
        obj.scalar_keys = []
        obj.calibration = calibration
        obj.output_key = "f"
        obj.objective_tag = "BeamSize"
        return obj

    def _scalars(self, device: str, x: float, y: float) -> dict:
        return {f"{device}_x_fwhm": x, f"{device}_y_fwhm": y}

    def test_quadrature_sum(self):
        ev = self._make_shell(calibration=2.0)
        # (3 * 2)² + (4 * 2)² = 36 + 64 = 100
        result = ev.compute_objective(self._scalars("UC_Cam", 3.0, 4.0), bin_number=1)
        assert result == pytest.approx(100.0)

    def test_calibration_scales_both_axes(self):
        ev = self._make_shell(calibration=0.5)
        result = ev.compute_objective(self._scalars("UC_Cam", 10.0, 10.0), bin_number=1)
        assert result == pytest.approx(50.0)

    def test_unit_calibration_is_pixel_quadrature(self):
        ev = self._make_shell(calibration=1.0)
        result = ev.compute_objective(self._scalars("UC_Cam", 3.0, 4.0), bin_number=1)
        assert result == pytest.approx(25.0)

    def test_pixel_values_unchanged(self):
        ev = self._make_shell(calibration=2.0)
        obs = ev.compute_observables(self._scalars("UC_Cam", 3.0, 4.0), bin_number=1)
        assert obs["x_fwhm_px"] == pytest.approx(3.0)
        assert obs["y_fwhm_px"] == pytest.approx(4.0)

    def test_calibrated_values_scaled(self):
        ev = self._make_shell(calibration=2.0)
        obs = ev.compute_observables(self._scalars("UC_Cam", 3.0, 4.0), bin_number=1)
        assert obs["x_fwhm_units"] == pytest.approx(6.0)
        assert obs["y_fwhm_units"] == pytest.approx(8.0)

    def test_quadrature_observable_matches_objective(self):
        ev = self._make_shell(calibration=2.0)
        scalars = self._scalars("UC_Cam", 3.0, 4.0)
        obs = ev.compute_observables(scalars, bin_number=1)
        objective = ev.compute_objective(scalars, bin_number=1)
        assert obs["size_quadrature_units2"] == pytest.approx(objective)


# ---------------------------------------------------------------------------
# MaxCountsEvaluator
# ---------------------------------------------------------------------------


class TestMaxCountsEvaluator:
    """Negate image_total; observe centroid + peak."""

    def _make_shell(self, device_name: str = "UC_Cam"):
        from geecs_scanner.optimization.evaluators.beam_sum_counts_evaluator import (
            MaxCountsEvaluator,
        )

        obj = object.__new__(MaxCountsEvaluator)
        obj.diagnostics = [_fake_diag(device_name)]
        obj.scan_analyzers = {}
        obj.scalar_keys = []
        obj.output_key = "f"
        obj.objective_tag = "TotalCounts"
        return obj

    def test_compute_objective_negates_total(self):
        ev = self._make_shell()
        scalars = {"UC_Cam_image_total": 5000.0}
        assert ev.compute_objective(scalars, bin_number=1) == pytest.approx(-5000.0)

    def test_compute_objective_zero_total(self):
        ev = self._make_shell()
        scalars = {"UC_Cam_image_total": 0.0}
        assert ev.compute_objective(scalars, bin_number=1) == pytest.approx(0.0)

    def test_compute_observables_values_passed_through(self):
        ev = self._make_shell()
        scalars = {
            "UC_Cam_x_CoM": 100.0,
            "UC_Cam_y_CoM": 200.0,
            "UC_Cam_image_peak_value": 3000.0,
        }
        obs = ev.compute_observables(scalars, bin_number=1)
        assert obs["x_CoM"] == pytest.approx(100.0)
        assert obs["y_CoM"] == pytest.approx(200.0)
        assert obs["image_peak_value"] == pytest.approx(3000.0)

    def test_compute_observables_exact_key_set(self):
        """Only the three named observables, not the objective input."""
        ev = self._make_shell()
        scalars = {
            "UC_Cam_x_CoM": 1.0,
            "UC_Cam_y_CoM": 2.0,
            "UC_Cam_image_peak_value": 3.0,
            "UC_Cam_image_total": 4.0,
        }
        obs = ev.compute_observables(scalars, bin_number=1)
        assert set(obs.keys()) == {"x_CoM", "y_CoM", "image_peak_value"}


# ---------------------------------------------------------------------------
# BeamPositionEvaluator (BAX)
# ---------------------------------------------------------------------------


class TestBeamPositionEvaluator:
    """BAX evaluator: pull observable_names off the primary device's prefix."""

    def _make_shell(self, observable_names=None, device_name: str = "UC_Cam"):
        from geecs_scanner.optimization.evaluators.beam_position_evaluator import (
            BeamPositionEvaluator,
        )

        obj = object.__new__(BeamPositionEvaluator)
        obj.diagnostics = [_fake_diag(device_name)]
        obj.scan_analyzers = {}
        obj.scalar_keys = []
        obj.observable_names = (
            observable_names if observable_names is not None else ["x_CoM"]
        )
        obj.output_key = None
        obj.objective_tag = "BeamPosition"
        return obj

    def test_single_observable_happy_path(self):
        ev = self._make_shell(["x_CoM"])
        obs = ev.compute_observables({"UC_Cam_x_CoM": 42.0}, bin_number=1)
        assert obs == {"x_CoM": pytest.approx(42.0)}

    def test_two_observables_both_extracted(self):
        ev = self._make_shell(["x_CoM", "y_CoM"])
        obs = ev.compute_observables(
            {"UC_Cam_x_CoM": 5.0, "UC_Cam_y_CoM": -3.0}, bin_number=1
        )
        assert obs["x_CoM"] == pytest.approx(5.0)
        assert obs["y_CoM"] == pytest.approx(-3.0)

    def test_partial_failure_logs_warning_and_returns_partial(self, caplog):
        import logging

        ev = self._make_shell(["x_CoM", "y_CoM"])
        with caplog.at_level(logging.WARNING):
            obs = ev.compute_observables({"UC_Cam_x_CoM": 5.0}, bin_number=1)

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
    """Simulation evaluator: ``compute_observables`` reads s-file scalars from the dict."""

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
        obj.diagnostics = []
        obj.scan_analyzers = {}
        obj.scalar_keys = [
            "U_S1H:Current",
            "U_EMQTripletBipolar:Current_Limit.Ch1",
        ]
        obj.scale_factor = scale_factor
        obj.reference_setpoint = reference_setpoint
        obj.noise_amplitude = noise_amplitude
        obj.calibration = calibration
        obj.control_variable_name = "U_S1H:Current"
        obj.measurement_variable_name = "U_EMQTripletBipolar:Current_Limit.Ch1"
        obj.output_key = None
        obj.objective_tag = "BeamPosition"
        return obj

    def _scalars(self, control_val: float, measure_val: float) -> dict:
        return {
            "U_S1H:Current": control_val,
            "U_EMQTripletBipolar:Current_Limit.Ch1": measure_val,
        }

    def test_zero_offset_gives_zero_centroid(self):
        # control == reference → offset = 0 → x_CoM = 0
        ev = self._make_shell(scale_factor=1000.0, reference_setpoint=2.0)
        obs = ev.compute_observables(self._scalars(2.0, 0.5), bin_number=1)
        assert obs["x_CoM"] == pytest.approx(0.0)

    def test_centroid_formula_known_values(self):
        # offset = 2.0 - 1.0 = 1.0, measure = 0.5 → x_CoM = 1000 * 1.0 * 1.5 = 1500
        ev = self._make_shell(scale_factor=1000.0, reference_setpoint=1.0)
        obs = ev.compute_observables(self._scalars(2.0, 0.5), bin_number=1)
        assert obs["x_CoM"] == pytest.approx(1500.0)

    def test_negative_offset_gives_negative_centroid(self):
        # offset = 0.5 - 1.0 = -0.5, measure = 0 → x_CoM = 1000 * -0.5 * 1 = -500
        ev = self._make_shell(scale_factor=1000.0, reference_setpoint=1.0)
        obs = ev.compute_observables(self._scalars(0.5, 0.0), bin_number=1)
        assert obs["x_CoM"] == pytest.approx(-500.0)

    def test_calibration_scales_output(self):
        # x_CoM before calibration = 1500, calibration = 0.001 → 1.5
        ev = self._make_shell(
            scale_factor=1000.0, reference_setpoint=1.0, calibration=0.001
        )
        obs = ev.compute_observables(self._scalars(2.0, 0.5), bin_number=1)
        assert obs["x_CoM"] == pytest.approx(1.5)

    def test_missing_column_returns_empty_with_warning(self, caplog):
        import logging

        ev = self._make_shell()
        with caplog.at_level(logging.WARNING):
            obs = ev.compute_observables({"wrong_column": 1.0}, bin_number=1)
        assert obs == {}
        assert len(caplog.records) > 0


# ---------------------------------------------------------------------------
# Analyzer-scalar pass-through (pin the post-RFC-#412 inversion point)
# ---------------------------------------------------------------------------


class TestAnalyzerScalarPassthrough:
    """``_get_value`` forwards analyzer-emitted scalars through unchanged.

    Pins the *today* contract: ImageAnalysis emits keys already prefixed
    by ``camera_config.name`` (e.g. ``"UC_TopView_x_fwhm"``), so the
    framework does not add another prefix. When RFC #412 lands and moves
    prefixing into ScanAnalysis (analyzers emit bare ``"x_fwhm"`` keys),
    these assertions need to flip: the framework will then be responsible
    for prefixing, and the slot's key will become ``f"{device}_{k}"``.
    """

    def _make_evaluator_with_one_analyzer(
        self, device_name: str, analyzer_result_scalars: dict, mode: str = "per_bin"
    ):
        """Build a BaseEvaluator shell with one analyzer pre-seeded with results."""
        from geecs_scanner.optimization.base_evaluator import BaseEvaluator

        analyzer = MagicMock()
        analyzer.analysis_mode = mode
        result = MagicMock()
        result.scalars = analyzer_result_scalars
        # per_bin: results keyed by bin_number; per_shot: keyed by shot
        analyzer.results = {1: result} if mode == "per_bin" else {1: result, 2: result}

        class _Concrete(BaseEvaluator):
            def compute_objective(self, scalars, bin_number):
                return 0.0  # don't care about the value here

        obj = object.__new__(_Concrete)
        obj.diagnostics = [_fake_diag(device_name)]
        obj.scan_analyzers = {device_name: analyzer}
        obj.scalar_keys = []
        obj.output_key = "f"
        obj.objective_tag = "test"
        obj.bin_number = 1
        obj.current_shot_numbers = [1, 2]
        obj.current_data_bin = MagicMock()
        obj.scan_tag = None
        return obj

    def test_per_bin_analyzer_scalars_pass_through_unchanged(self):
        """Per-bin: analyzer emits already-prefixed keys; framework forwards verbatim."""
        # Capture what compute_objective_from_shots receives.
        captured: dict = {}

        ev = self._make_evaluator_with_one_analyzer(
            device_name="UC_TopView",
            # Analyzer-emitted keys are already prefixed (today's reality)
            analyzer_result_scalars={
                "UC_TopView_image_total": 100.0,
                "UC_TopView_x_fwhm": 5.0,
            },
            mode="per_bin",
        )

        def _capture(scalars_list, bin_number):
            captured["scalars_list"] = scalars_list
            return 0.0

        ev.compute_objective_from_shots = _capture
        ev._get_value({})

        # Every slot got both keys verbatim — no double-prefix.
        for slot in captured["scalars_list"]:
            assert "UC_TopView_image_total" in slot
            assert "UC_TopView_x_fwhm" in slot
            # And the framework did NOT add another layer of prefixing.
            assert "UC_TopView_UC_TopView_image_total" not in slot
            assert "UC_TopView_UC_TopView_x_fwhm" not in slot

    def test_per_shot_analyzer_scalars_pass_through_unchanged(self):
        """Per-shot: same contract — keys forwarded verbatim."""
        captured: dict = {}

        ev = self._make_evaluator_with_one_analyzer(
            device_name="UC_TopView",
            analyzer_result_scalars={"UC_TopView_x_fwhm": 5.0},
            mode="per_shot",
        )

        def _capture(scalars_list, bin_number):
            captured["scalars_list"] = scalars_list
            return 0.0

        ev.compute_objective_from_shots = _capture
        ev._get_value({})

        for slot in captured["scalars_list"]:
            assert "UC_TopView_x_fwhm" in slot
            assert "UC_TopView_UC_TopView_x_fwhm" not in slot
