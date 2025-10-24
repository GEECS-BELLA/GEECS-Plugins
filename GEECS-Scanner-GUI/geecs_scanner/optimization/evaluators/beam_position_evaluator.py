"""
Example evaluator for tracking beam centroid position using MultiDeviceScanEvaluator.

Classes
-------
BeamPositionEvaluator
    Evaluator that reports calibrated centroid observables alongside the optimization objective.
    Includes a lightweight simulation mode that derives observables directly from
    requested setpoints for offline testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Callable, Optional

if TYPE_CHECKING:
    pass

import logging
import random

from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
    MultiDeviceScanEvaluator,
)

logger = logging.getLogger(__name__)


class BeamPositionEvaluator(MultiDeviceScanEvaluator):
    """
    Evaluator for tracking beam position.

    Parameters
    ----------
    calibration : float, default=24.4e-3
        Spatial calibration factor in mm/pixel (or desired units/pixel).
    simulate : bool, default=False
        If True, skip analyzer execution and synthesize observables from the
        setpoints provided to ``get_value``.
    simulation_model : callable, optional
        Optional callable ``f(input_data) -> dict | float`` used in simulation mode.
        When omitted, a simple placeholder model based on the provided setpoints
        is used.
    simulation_control_key : str, optional
        Name of the control variable to use in the default simulation model.
    simulation_measurement_key : str, optional
        Name of the measurement variable to use in the default simulation model.
    simulation_control_nominal : float, default=0.0
        Nominal value for the control variable used by the placeholder model.
    simulation_measurement_nominal : float, default=1.0
        Nominal value for the measurement variable used by the placeholder model.
    simulation_base_offset : float, default=0.0
        Baseline centroid (in pixels) when both variables sit at their nominal values.
    simulation_base_slope : float, default=1.0
        Baseline slope (in pixels per measurement unit) relating the measurement
        variable to the centroid.
    simulation_control_slope_gain : float, default=0.5
        Linear gain (in pixels per measurement unit per control unit) that modulates
        the slope as the control variable moves away from its nominal value.
    simulation_noise_std : float, default=0.0
        Standard deviation of optional Gaussian noise (in pixels) added to the
        simulated centroid.
    **kwargs
        Additional keyword arguments passed to MultiDeviceScanEvaluator,
        including 'analyzers', 'scan_data_manager', and 'data_logger'.
    """

    def __init__(
        self,
        calibration: float = 24.4e-3,
        simulate: bool = True,
        simulation_model: Optional[
            Callable[[Dict[str, float]], Dict[str, float] | float]
        ] = None,
        simulation_control_key: Optional[str] = None,
        simulation_measurement_key: Optional[str] = None,
        simulation_control_nominal: float = 0.0,
        simulation_measurement_nominal: float = 1.0,
        simulation_base_offset: float = 0.0,
        simulation_base_slope: float = 1.0,
        simulation_control_slope_gain: float = 0.5,
        simulation_noise_std: float = 0.0,
        **kwargs,
    ):
        """Initialize the beam position evaluator."""
        super().__init__(**kwargs)
        self.calibration = calibration
        # Get device name from first analyzer config
        self.device_name = self.analyzer_configs[0].device_name
        self.observable_key = "x_cen"
        self.simulate = simulate
        self.simulation_model = simulation_model
        self.simulation_control_key = simulation_control_key
        self.simulation_measurement_key = simulation_measurement_key
        self.simulation_control_nominal = simulation_control_nominal
        self.simulation_measurement_nominal = simulation_measurement_nominal
        self.simulation_base_offset = simulation_base_offset
        self.simulation_base_slope = simulation_base_slope
        self.simulation_control_slope_gain = simulation_control_slope_gain
        self.simulation_noise_std = max(0.0, simulation_noise_std)

    def get_value(self, input_data: Dict) -> Dict:
        """
        Evaluate the objective or, if requested, simulate it from setpoints.

        When ``simulate`` is True, bypass analyzer execution and instead
        derive the observable(s) directly from the provided ``input_data``.
        """
        if self.simulate:
            simulated = self._simulate_from_setpoints(input_data or {})
            logger.info(
                "Simulated observables %s from input data %s",
                simulated,
                input_data,
            )
            return simulated

        return super().get_value(input_data)

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """
        Compute beam position objective from CoM measurements.

        Parameters
        ----------
        scalar_results : dict
            Dictionary of scalar results from all analyzers.
            Expected to contain x_CoM and y_CoM metrics.
        bin_number : int
            Current bin number being evaluated (not used in this implementation).

        Returns
        -------
        float
            Beam position observable value
        """
        x_mm = self._extract_position(scalar_results, axis="x")
        # Return calibrated position as the optimization objective (can be overridden if needed)
        return x_mm

    def compute_observables(
        self, scalar_results: dict, bin_number: int
    ) -> Dict[str, float]:
        """
        Provide calibrated centroid observables required by downstream generators.

        Returns a dictionary containing ``x_cen`` (and optionally ``y_cen``) expressed
        in physical units. These observable keys must match the VOCS configuration.
        """
        observables: Dict[str, float] = {}
        try:
            observables[self.observable_key] = self._extract_position(
                scalar_results, axis="x"
            )
        except KeyError as err:
            logger.warning("Unable to extract x centroid for observables: %s", err)

        # y centroid is optional; include if available
        try:
            observables["y_cen"] = self._extract_position(scalar_results, axis="y")
        except KeyError:
            # silently ignore missing y centroid
            pass

        return observables

    def _simulate_from_setpoints(
        self, input_data: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Build simulated outputs from control setpoints.

        Parameters
        ----------
        input_data : dict
            Mapping of control/measurment names to their proposed setpoints.

        Returns
        -------
        dict
            Dictionary containing at least ``self.output_key`` and
            ``self.observable_key`` entries.
        """
        if self.simulation_model is not None:
            simulated = self.simulation_model(input_data)
            if isinstance(simulated, dict):
                outputs = dict(simulated)
            else:
                outputs = {self.observable_key: float(simulated)}
        else:
            outputs = {self.observable_key: self._default_simulation_model(input_data)}

        outputs.setdefault(self.output_key, outputs[self.observable_key])
        return outputs

    def _default_simulation_model(self, input_data: Dict[str, float]) -> float:
        """
        Fallback simulation that derives a centroid from the provided setpoints.

        This default implementation takes the first numeric value (sorted by key)
        as the primary drive term, subtracts the second if available, and applies
        the evaluator calibration. It is intended only as a placeholder; supply
        ``simulation_model`` for realistic behaviour.
        """
        numeric_items = [
            (key, float(value))
            for key, value in input_data.items()
            if isinstance(value, (int, float))
        ]
        if not numeric_items:
            return 0.0
        control_val: Optional[float] = None
        measurement_val: Optional[float] = None

        if self.simulation_control_key:
            control_val = self._safe_fetch_value(
                input_data, self.simulation_control_key
            )
        if self.simulation_measurement_key:
            measurement_val = self._safe_fetch_value(
                input_data, self.simulation_measurement_key
            )

        if control_val is None or measurement_val is None:
            numeric_items.sort(key=lambda item: item[0])
            if control_val is None:
                control_val = numeric_items[0][1]
            if measurement_val is None:
                measurement_val = numeric_items[1][1] if len(numeric_items) > 1 else 0.0

        if control_val is None:
            control_val = self.simulation_control_nominal
        if measurement_val is None:
            measurement_val = self.simulation_measurement_nominal

        slope = self.simulation_base_slope + self.simulation_control_slope_gain * (
            control_val - self.simulation_control_nominal
        )
        delta_measurement = measurement_val - self.simulation_measurement_nominal
        centroid_pixels = self.simulation_base_offset + slope * delta_measurement

        if self.simulation_noise_std > 0.0:
            centroid_pixels += random.gauss(0.0, self.simulation_noise_std)

        # Convert simulated pixels to physical units for parity with live data
        return centroid_pixels * self.calibration

    @staticmethod
    def _safe_fetch_value(data: Dict[str, float], key: str) -> Optional[float]:
        """Attempt to retrieve and cast a numeric value from ``data``."""
        if key not in data:
            return None
        try:
            return float(data[key])
        except (TypeError, ValueError):
            return None

    def _extract_position(self, scalar_results: dict, axis: str = "x") -> float:
        """
        Extract the calibrated beam centroid along the requested axis.

        Parameters
        ----------
        scalar_results : dict
            Analyzer scalar results.
        axis : {"x", "y"}
            Axis for which to obtain the centroid.

        Returns
        -------
        float
            Calibrated centroid position in physical units.
        """
        axis = axis.lower()
        if axis not in {"x", "y"}:
            raise ValueError(f"Unsupported axis '{axis}'. Expected 'x' or 'y'.")
        metric = f"{axis}_CoM"
        raw_value = self.get_scalar(self.device_name, metric, scalar_results)
        return raw_value * self.calibration
