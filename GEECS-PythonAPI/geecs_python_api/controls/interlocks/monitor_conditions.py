"""
Monitor condition builders for interlock system.

Provides composable, reusable condition factories for common monitoring scenarios:
- ThresholdCheck: value vs. threshold comparison
- AlignmentCheck: tolerance-based alignment checking
- MultiCheck: combines multiple conditions with OR logic
- CustomCheck: arbitrary predicate functions

NOTE: Staleness detection is built-in to DeviceMonitorGroup and automatically
applied by all condition checks. No separate StalenessGuard needed.
"""

import logging
from typing import Callable, Any, Optional, TYPE_CHECKING

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
        device_monitor_group: 'DeviceMonitorGroup',
        device_name: str,
        variable_name: str,
        threshold: float,
        operator: str = '<'
    ):
        """
        Initialize threshold check.
        
        Args:
            device_monitor_group: DeviceMonitorGroup instance for state access
            device_name: Alias of device to monitor
            variable_name: Name of variable to check
            threshold: Threshold value
            operator: Comparison operator ('<', '>', '<=', '>=', '==', '!=')
        
        Returns:
            Callable that returns bool (True=unsafe, False=safe)
        """
        self.device_monitor_group = device_monitor_group
        self.device_name = device_name
        self.variable_name = variable_name
        self.threshold = threshold
        
        # Map operator strings to functions
        self.operators = {
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
        }
        
        if operator not in self.operators:
            raise ValueError(f"Invalid operator: {operator}. Must be one of {list(self.operators.keys())}")
        
        self.operator = operator
        self.operator_fn = self.operators[operator]
        
        # Track diagnostic info (what failed, value, threshold)
        self.last_failure_info = None
    
    def __call__(self) -> bool:
        """
        Check if threshold condition is met.
        
        Returns:
            True if unsafe (threshold condition met), False if safe
        """
        try:
            value = self.device_monitor_group.get_value(self.device_name, self.variable_name)
            
            if value is None:
                logger.warning(
                    f"No data for {self.device_name}.{self.variable_name} - returning unsafe"
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
                "reason": str(e)
            }
            return True  # Fail-safe: return unsafe on error
    
    def get_diagnostic_info(self) -> str:
        """
        Get diagnostic info about the last failure.
        
        Returns:
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
        device_monitor_group: 'DeviceMonitorGroup',
        device_name: str,
        value_variable: str,
        target_variable: str,
        tolerance: float
    ):
        """
        Initialize alignment check.
        
        Args:
            device_monitor_group: DeviceMonitorGroup instance for state access
            device_name: Alias of device to monitor
            value_variable: Name of value variable (e.g., 'centroidx')
            target_variable: Name of target variable (e.g., 'Target.X')
            tolerance: Maximum allowed difference
        
        Returns:
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
        
        Returns:
            True if unsafe (not aligned), False if safe (aligned)
        """
        try:
            value = self.device_monitor_group.get_value(self.device_name, self.value_variable)
            target = self.device_monitor_group.get_value(self.device_name, self.target_variable)
            
            if value is None or target is None:
                logger.warning(
                    f"Missing data for {self.device_name} alignment check "
                    f"({self.value_variable} or {self.target_variable}) - returning unsafe"
                )
                self.last_failure_info = {
                    "variable": f"{self.value_variable}",
                    "value": str(value),
                    "target": str(target),
                    "tolerance": self.tolerance,
                }
                return True
            
            distance = abs(value - target)
            is_aligned = distance < self.tolerance
            
            if not is_aligned:
                self.last_failure_info = {
                    "variable": self.value_variable,
                    "value": value,
                    "target": target,
                    "distance": distance,
                    "tolerance": self.tolerance,
                }
            else:
                self.last_failure_info = None
            
            return not is_aligned  # Return True if NOT aligned (unsafe)
        
        except Exception as e:
            logger.error(
                f"Error in AlignmentCheck ({self.device_name}): {e}"
            )
            self.last_failure_info = {
                "variable": self.value_variable,
                "value": "ERROR",
                "reason": str(e)
            }
            return True  # Fail-safe: return unsafe on error
    
    def get_diagnostic_info(self) -> str:
        """
        Get diagnostic info about the last failure.
        
        Returns:
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
    
    def __init__(
        self,
        conditions: list[Callable[[], bool]]
    ):
        """
        Initialize multi-condition check.
        
        Args:
            conditions: List of check functions
        
        Returns:
            Callable that returns bool (True=any unsafe, False=all safe)
        """
        self.conditions = conditions
        
        # Track which condition failed
        self.last_failure_info = None
    
    def __call__(self) -> bool:
        """
        Evaluate all conditions.
        
        Returns:
            True if ANY condition returns True, False if ALL return False
        """
        try:
            self.last_failure_info = None  # Reset before checking
            
            # Evaluate all conditions; short-circuit on first unsafe
            for condition in self.conditions:
                try:
                    if condition():
                        # Store which condition failed
                        if hasattr(condition, 'get_diagnostic_info'):
                            self.last_failure_info = condition.get_diagnostic_info()
                        return True
                except Exception as e:
                    logger.error(f"Error evaluating condition: {e}")
                    return True  # Fail-safe: return unsafe on error
            
            return False  # All conditions safe
        
        except Exception as e:
            logger.error(f"Error in MultiCheck: {e}")
            return True  # Fail-safe: return unsafe on error
    
    def get_diagnostic_info(self) -> str:
        """
        Get diagnostic info from the failed condition.
        
        Returns:
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
        device_monitor_group: 'DeviceMonitorGroup',
        device_name: str,
        predicate: Callable[[dict], bool]
    ):
        """
        Initialize custom check.
        
        Args:
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
        
        Returns:
            Result of predicate function
        """
        try:
            device = self.device_monitor_group.get_device(self.device_name)
            if device is None:
                logger.warning(f"Device {self.device_name} not found - returning unsafe")
                self.last_failure_info = "Device not found"
                return True
            
            state = device.state
            if state is None:
                logger.warning(f"No state for {self.device_name} - returning unsafe")
                self.last_failure_info = "No device state"
                return True
            
            result = self.predicate(state)
            
            if result:
                self.last_failure_info = "Custom predicate returned unsafe"
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
        
        Returns:
            String describing what failed, or empty string if safe
        """
        return self.last_failure_info or ""
