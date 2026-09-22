"""
Monitor condition builders for interlock system.

Provides composable, reusable condition factories for common monitoring scenarios:
- ThresholdCheck: value vs. threshold comparison
- AlignmentCheck: tolerance-based alignment checking
- MultiCheck: combines multiple conditions with OR logic
- CustomCheck: arbitrary predicate functions
- ConfigurationChangeCheck: arm/disarm gate that trips if watched settings drift
- ExternalStatusCheck: wraps a non-GEECS, one-shot UDP status signal (see
  external_status_listener.py) so it can be combined like any other check

NOTE: Staleness detection is built-in to DeviceMonitorGroup and automatically
applied by all condition checks. No separate StalenessGuard needed.
"""

import logging
from collections.abc import Callable

from .device_monitor_group import DeviceMonitorGroup

logger = logging.getLogger(__name__)


class ThresholdCheck:
    """
    Factory for threshold-based monitoring.

    Checks if a device variable crosses a threshold using specified operator.
    Returns True (unsafe) if condition is met, False (safe) otherwise.
    """

    def __init__(
        self,
        device_monitor_group: "DeviceMonitorGroup",
        device_name: str,
        variable_name: str,
        threshold: float,
        operator: str = "<",
    ):
        """
        Initialize threshold check.

        Args:
            device_monitor_group: DeviceMonitorGroup instance for state access
            device_name: Alias of device to monitor
            variable_name: Name of variable to check
            threshold: Threshold value
            operator: Comparison operator ('<', '>', '<=', '>=', '==', '!=')

        Returns
        -------
            Callable that returns bool (True=unsafe, False=safe)
        """
        self.device_monitor_group = device_monitor_group
        self.device_name = device_name
        self.variable_name = variable_name
        self.threshold = threshold

        # Map operator strings to functions
        self.operators = {
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }

        if operator not in self.operators:
            raise ValueError(
                f"Invalid operator: {operator}. Must be one of {list(self.operators.keys())}"
            )

        self.operator = operator
        self.operator_fn = self.operators[operator]

        # Track diagnostic info (what failed, value, threshold)
        self.last_failure_info = None

    def __call__(self) -> bool:
        """
        Check if threshold condition is met.

        Returns
        -------
            True if unsafe (threshold condition met), False if safe
        """
        try:
            value = self.device_monitor_group.get_value(
                self.device_name, self.variable_name
            )

            if value is None:
                logger.warning(
                    f"No data for {self.device_name}.{self.variable_name} - returning dangerous"
                )
                self.last_failure_info = {
                    "variable": self.variable_name,
                    "value": "NO DATA",
                    "threshold": self.threshold,
                    "operator": self.operator,
                }
                return True

            is_unsafe = self.operator_fn(value, self.threshold)

            # Track what failed
            if is_unsafe:
                self.last_failure_info = {
                    "variable": self.variable_name,
                    "value": value,
                    "threshold": self.threshold,
                    "operator": self.operator,
                }
            else:
                self.last_failure_info = None

            return is_unsafe

        except Exception as e:
            logger.error(
                f"Error in ThresholdCheck ({self.device_name}.{self.variable_name}): {e}"
            )
            self.last_failure_info = {
                "variable": self.variable_name,
                "value": "ERROR",
                "reason": str(e),
            }
            return True  # Fail-safe: return unsafe on error

    def get_diagnostic_info(self) -> str:
        """
        Get diagnostic info about the last failure.

        Returns
        -------
            String describing what failed, or empty string if safe
        """
        if self.last_failure_info:
            return (
                f"{self.last_failure_info['variable']}="
                f"{self.last_failure_info['value']} "
                f"(threshold: {self.last_failure_info['threshold']})"
            )
        return ""


class AlignmentCheck:
    """
    Factory for tolerance-based alignment checking.

    Checks if a value is within tolerance of a target value.
    Returns True (unsafe) if NOT aligned, False (safe) if aligned.
    """

    def __init__(
        self,
        device_monitor_group: "DeviceMonitorGroup",
        device_name: str,
        value_variable: str | list[str],
        target_variable: str | list[str],
        tolerance: float | list[float],
    ):
        """
        Initialize alignment check.

        Args:
            device_monitor_group: DeviceMonitorGroup instance for state access
            device_name: Alias of device to monitor
            value_variable: Name of value variable (e.g., 'centroidx')
            target_variable: Name of target variable (e.g., 'Target.X')
            tolerance: Maximum allowed difference

        Returns
        -------
            Callable that returns bool (True=not aligned/unsafe, False=aligned/safe)
        """
        self.device_monitor_group = device_monitor_group
        self.device_name = device_name
        self.value_variable = value_variable
        self.target_variable = target_variable
        self.tolerance = tolerance

        # Track diagnostic info
        self.last_failure_info = None

    def __call__(self) -> bool:
        """
        Check if alignment condition is met.

        Returns
        -------
            True if unsafe (not aligned), False if safe (aligned)
        """
        try:
            value_variables = (
                [self.value_variable]
                if isinstance(self.value_variable, str)
                else self.value_variable
            )
            target_variables = (
                [self.target_variable]
                if isinstance(self.target_variable, str)
                else self.target_variable
            )
            tolerances = (
                [self.tolerance]
                if isinstance(self.tolerance, (int, float))
                else self.tolerance
            )
            if not (len(value_variables) == len(target_variables) == len(tolerances)):
                raise ValueError(
                    "Alignment values, targets, and tolerances must have equal lengths"
                )

            values = [
                self.device_monitor_group.get_value(self.device_name, variable)
                for variable in value_variables
            ]
            targets = [
                self.device_monitor_group.get_value(self.device_name, variable)
                for variable in target_variables
            ]

            if any(
                value is None or target is None
                for value, target in zip(values, targets)
            ):
                logger.warning(
                    f"Missing data for {self.device_name} alignment check "
                    f"({value_variables} or {target_variables}) - returning dangerous"
                )
                self.last_failure_info = {
                    "variable": ", ".join(value_variables),
                    "value": str(values),
                    "target": str(targets),
                    "tolerance": str(tolerances),
                }
                return True

            distances = [abs(value - target) for value, target in zip(values, targets)]
            is_aligned = all(
                distance < tolerance
                for distance, tolerance in zip(distances, tolerances)
            )

            if not is_aligned:
                self.last_failure_info = {
                    "variable": ", ".join(value_variables),
                    "value": values,
                    "target": targets,
                    "distance": distances,
                    "tolerance": tolerances,
                }
            else:
                self.last_failure_info = None

            return not is_aligned  # Return True if NOT aligned (unsafe)

        except Exception as e:
            logger.error(f"Error in AlignmentCheck ({self.device_name}): {e}")
            self.last_failure_info = {
                "variable": self.value_variable,
                "value": "ERROR",
                "reason": str(e),
            }
            return True  # Fail-safe: return unsafe on error

    def get_diagnostic_info(self) -> str:
        """
        Get diagnostic info about the last failure.

        Returns
        -------
            String describing what failed, or empty string if safe
        """
        if self.last_failure_info:
            if "target" in self.last_failure_info:
                return (
                    f"{self.last_failure_info['variable']}="
                    f"{self.last_failure_info['value']} "
                    f"(target: {self.last_failure_info['target']}, "
                    f"tolerance: {self.last_failure_info['tolerance']})"
                )
            return (
                f"{self.last_failure_info['variable']}="
                f"{self.last_failure_info['value']}"
            )
        return ""


class MultiCheck:
    """
    Combines multiple condition builders with OR logic.

    Returns True (unsafe) if ANY condition returns True.
    Returns False (safe) only if ALL conditions return False.

    NOTE: Each condition automatically benefits from built-in staleness checking
    via DeviceMonitorGroup. If any device data is stale, conditions fail-safe (return True/unsafe).
    """

    def __init__(self, conditions: list[Callable[[], bool]]):
        """
        Initialize multi-condition check.

        Args:
            conditions: List of check functions

        Returns
        -------
            Callable that returns bool (True=any unsafe, False=all safe)
        """
        self.conditions = conditions

        # Track which condition failed
        self.last_failure_info = None

    def __call__(self) -> bool:
        """
        Evaluate all conditions.

        Returns
        -------
            True if ANY condition returns True, False if ALL return False
        """
        try:
            self.last_failure_info = None
            is_unsafe = False

            # Evaluate all conditions; short-circuit on first unsafe
            for condition in self.conditions:
                try:
                    if condition():
                        is_unsafe = True
                        # Store which condition failed
                        if hasattr(condition, "get_diagnostic_info"):
                            self.last_failure_info = condition.get_diagnostic_info()
                        break  # No need to check further if already unsafe
                except Exception as e:
                    logger.error(f"Error evaluating condition: {e}")
                    is_unsafe = True
                    break  # Fail-safe: return unsafe on error

            return is_unsafe  # All conditions safe

        except Exception as e:
            logger.error(f"Error in MultiCheck: {e}")
            return True  # Fail-safe: return unsafe on error

    def get_diagnostic_info(self) -> str:
        """
        Get diagnostic info from the failed condition.

        Returns
        -------
            String describing what failed, or empty string if safe
        """
        return self.last_failure_info or ""


class CustomCheck:
    """
    Allows arbitrary predicate functions for custom logic.

    Wraps a user-provided function that operates on device state.
    """

    def __init__(
        self,
        device_monitor_group: "DeviceMonitorGroup",
        device_name: str,
        predicate: Callable[[dict], bool],
    ):
        """
        Initialize custom check.

        Args:
        ----
            device_monitor_group: DeviceMonitorGroup instance for state access
            device_name: Alias of device to monitor
            predicate: Function that receives device state dict and returns
                      bool (True=unsafe, False=safe)
        """
        self.device_monitor_group = device_monitor_group
        self.device_name = device_name
        self.predicate = predicate

        # Track diagnostic info
        self.last_failure_info = None

    def __call__(self) -> bool:
        """
        Evaluate custom predicate.

        Returns
        -------
            Result of predicate function
        """
        try:
            device = self.device_monitor_group.get_device(self.device_name)
            if device is None:
                logger.warning(
                    f"Device {self.device_name} not found - returning dangerous"
                )
                self.last_failure_info = "Device not found"
                return True

            state = device.state
            if state is None:
                logger.warning(f"No state for {self.device_name} - returning dangerous")
                self.last_failure_info = "No device state"
                return True

            result = self.predicate(state)

            if result:
                self.last_failure_info = "Custom predicate returned dangerous"
            else:
                self.last_failure_info = None

            return result

        except Exception as e:
            logger.error(f"Error in CustomCheck ({self.device_name}): {e}")
            self.last_failure_info = f"Error: {str(e)}"
            return True  # Fail-safe: return unsafe on error

    def get_diagnostic_info(self) -> str:
        """
        Get diagnostic info about the last failure.

        Returns
        -------
            String describing what failed, or empty string if safe
        """
        return self.last_failure_info or ""


class ConfigurationChangeCheck:
    """Triggers when monitored device settings change after arming interlock."""

    def __init__(self, device_monitor_group, device_name, variables):
        self.device_monitor_group = device_monitor_group
        self.device_name = device_name
        self.variables = variables
        self.baseline = {}
        self.last_failure_info = ""

    def arm_interlock(self):  # noqa: D102
        values = {
            variable: self.device_monitor_group.get_value(self.device_name, variable)
            for variable in self.variables
        }

        if any(value is None for value in values.values()):
            raise RuntimeError("Cannot arm interlock: Device settings are unavailable")

        self.baseline = values
        self.last_failure_info = ""

    def __call__(self):  # noqa: D102
        if not self.baseline:
            self.last_failure_info = "Interlock not armed"
            return True  # unsafe if not armed

        for variable in self.variables:
            current = self.device_monitor_group.get_value(self.device_name, variable)

            if current != self.baseline[variable]:
                self.last_failure_info = (
                    f"{variable} changed from {self.baseline[variable]} to {current}"
                    f" update interlock thresholds and re-arm to continue."
                )
                return True  # unsafe while settings require re-arming

        self.last_failure_info = ""
        return False  # safe if all settings match baseline

    def get_diagnostic_info(self):  # noqa: D102
        return self.last_failure_info


class ExternalStatusCheck:
    """Wraps an ExternalStatusListener as an interlock condition.

    Unlike the other condition builders, this doesn't read a GEECS device
    through DeviceMonitorGroup - the "device" here is whatever unicasts UDP
    status packets to an ExternalStatusListener (see external_status_listener.py).
    Returns True (unsafe) until the listener has received its first packet
    (fail-safe on no data yet), then mirrors the listener's last known state.
    There is no staleness timeout, since the sender only transmits on change.
    """

    def __init__(self, listener, label: str | None = None):
        """
        Initialize the check.

        Args:
        ----
            listener: An ExternalStatusListener (already created; start/stop it
                separately, e.g. via InterlockConstructor.add_external_listener()).
            label: Optional human-readable name for diagnostics; defaults to the
                listener's own name.
        """
        self.listener = listener
        self.label = label or listener.name

    def __call__(self) -> bool:
        """Return True (unsafe) if no packet has arrived yet or the last one was a warning."""
        state = self.listener.get_state()
        if state is None:
            return True  # fail-safe: no packet received yet
        return state

    def get_diagnostic_info(self) -> str:
        """Describe the last received packet behind this check."""
        return self.listener.get_diagnostic_info()
