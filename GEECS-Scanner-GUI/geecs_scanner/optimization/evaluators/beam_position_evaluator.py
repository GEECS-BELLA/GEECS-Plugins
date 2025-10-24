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

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    pass

import logging

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
        setpoints provided to ``get_value`` using a fixed linear model.
    **kwargs
        Additional keyword arguments passed to MultiDeviceScanEvaluator,
        including 'analyzers', 'scan_data_manager', and 'data_logger'.
    """

    def __init__(
        self,
        calibration: float = 24.4e-3,
        simulate: bool = True,
        **kwargs,
    ):
        """Initialize the beam position evaluator."""
        super().__init__(**kwargs)
        self.calibration = calibration
        # Get device name from first analyzer config
        self.device_name = self.analyzer_configs[0].device_name
        self.observable_key = "x_CoM"
        self.simulate = simulate

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

        Returns a dictionary containing ``x_CoM`` (and optionally ``y_CoM``) expressed
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
            observables["y_CoM"] = self._extract_position(scalar_results, axis="y")
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
        outputs = {self.observable_key: self._default_simulation_model(input_data)}
        outputs.setdefault(self.output_key, outputs[self.observable_key])
        return outputs

    def _default_simulation_model(self, input_data: Dict[str, float]) -> float:
        """
        Fallback simulation that derives a centroid from the provided setpoints.

        The placeholder model implements a simple linear relation:

        ``x_CoM = offset + (base_amplitude + gain * (control - control_nominal)) * (measurement - measurement_nominal)``.

        Control and measurement values are inferred from the first two numeric
        entries in ``input_data`` (sorted by key). The result is converted to
        physical units using the evaluator calibration.
        """
        numeric_items = sorted(
            (key, float(value))
            for key, value in input_data.items()
            if isinstance(value, (int, float))
        )
        if len(numeric_items) < 2:
            raise KeyError(
                "Simulation mode requires at least two numeric setpoints to infer control and measurement values."
            )

        control_val = input_data.get("U_S1H:Current")
        measurement_val = input_data.get("U_EMQTripletBipolar:Current_Limit.Ch1")

        centroid_pixels = (measurement_val - 1) * (control_val - 1)

        return centroid_pixels * self.calibration

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
