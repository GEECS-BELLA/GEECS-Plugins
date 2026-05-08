"""Encapsulates all shot-trigger interactions with the shot-control device."""

from __future__ import annotations

import logging
from typing import Optional

from geecs_python_api.controls.devices.scan_device import ScanDevice
from geecs_scanner.engine.dialog_request import DEVICE_COMMAND_ERRORS
from geecs_scanner.utils.exceptions import TriggerError
from geecs_scanner.utils.retry import retry

logger = logging.getLogger(__name__)

_VALID_STATES = frozenset({"OFF", "SCAN", "STANDBY", "SINGLESHOT"})


class TriggerController:
    """Drive the shot-control device between trigger states.

    Extracted from ``ScanManager`` so that ``ScanStepExecutor`` receives an
    explicit typed dependency rather than dynamically-injected callables.

    Parameters
    ----------
    shot_control : ScanDevice or None
        The trigger device.  When ``None`` all trigger calls are no-ops.
    shot_control_variables : dict or None
        Variable-name → state-value mapping loaded from the shot control YAML.
    """

    def __init__(
        self,
        shot_control: Optional[ScanDevice] = None,
        shot_control_variables: Optional[dict] = None,
    ):
        self._shot_control = shot_control
        self._shot_control_variables = shot_control_variables

    # ------------------------------------------------------------------
    # Public trigger API
    # ------------------------------------------------------------------

    def trigger_on(self) -> None:
        """Set trigger state to SCAN."""
        self._set_trigger("SCAN")

    def trigger_off(self) -> None:
        """Set trigger state to OFF."""
        self._set_trigger("OFF")

    def set_standby(self) -> None:
        """Set trigger state to STANDBY."""
        self._set_trigger("STANDBY")

    def singleshot(self) -> list:
        """Fire a single-shot trigger pulse and return the results."""
        return self._set_trigger("SINGLESHOT")

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _set_trigger(self, state: str) -> list:
        """Set the trigger device to *state* for all shot-control variables.

        Parameters
        ----------
        state : str
            One of ``'OFF'``, ``'SCAN'``, ``'STANDBY'``, ``'SINGLESHOT'``.

        Returns
        -------
        list
            Return values from each ``shot_control.set()`` call.

        Raises
        ------
        TriggerError
            If any trigger variable fails after retries.
        """
        if self._shot_control is None or self._shot_control_variables is None:
            logger.debug("No shot control device, skipping 'set state %s'", state)
            return []

        if state not in _VALID_STATES:
            logger.error("Invalid trigger state: %s", state)
            return []

        device_name = self._shot_control.get_name()
        results = []
        for variable, variable_settings in self._shot_control_variables.items():
            set_value = variable_settings.get(state, "")
            if not set_value:
                continue
            try:
                result = retry(
                    lambda v=variable, sv=set_value: self._shot_control.set(
                        v, sv, exec_timeout=0.5
                    ),
                    attempts=2,
                    delay=0.25,
                    catch=DEVICE_COMMAND_ERRORS,
                    on_retry=lambda exc, n, var=variable: logger.debug(
                        "Trigger variable '%s' retry %d: %s", var, n, exc
                    ),
                )
                results.append(result)
                logger.debug("Trigger: set %s → %s", variable, set_value)
            except DEVICE_COMMAND_ERRORS as exc:
                raise TriggerError(
                    device_name,
                    f"set trigger state '{state}' on variable '{variable}'",
                    variable=variable,
                ) from exc

        logger.debug("Trigger state → %s", state)
        return results
