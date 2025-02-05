from __future__ import annotations
from typing import Optional, Any, Dict, Union

import time
import numpy as np
import numexpr as ne

from geecs_python_api.controls.interface.geecs_database import GeecsDatabase, load_config
from geecs_python_api.controls.devices.geecs_device import GeecsDevice

# Load experiment info globally
expt = load_config().get('Experiment', 'expt')
GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(expt)


class CompositeDeviceError(Exception):
    """Base class for composite device-related errors."""
    pass


class SetNotAllowedError(CompositeDeviceError):
    """Raised when attempting to set a variable on a 'get_only' composite device."""
    pass


class GetNotAllowedError(CompositeDeviceError):
    """Raised when attempting to get a variable from a 'set' type composite device."""
    pass


def initialize_scan_device(name: str, composite_spec_dict: Optional[Dict[str, Any]] = None,
                           virtual: bool = False) -> ScanDevice:
    """ Initializes either a scan or composite device depending on if a composite_spec_dict is given """
    if composite_spec_dict is None:
        return ScanDevice(name=name, virtual=virtual)
    else:
        return CompositeDevice(name=name, composite_spec_dict=composite_spec_dict)


class ScanDevice(GeecsDevice):
    """
    A class to represent a device for scanning.  Mostly a GEECS Device but with a few important differences:
    - the `get` function returns a dictionary instead of just the value.  This is to be compatible with subclasses
    - the `close` function includes a 1s wait between unsubscribing and closing.  This is to avoid a race condition
    """

    def __init__(self, name: str, virtual: bool = False):
        """
        Initialize a `ScanDevice`, as a regular GEECS Device
    
        Args:
            name (str): The name of the device.
            virtual (bool): Whether the device is virtual. Defaults to `False`.

        Raises:
            ValueError: If the device name is invalid
        """

        # Standard Device Initialization
        super().__init__(name, virtual)
        self.name = name

        if not virtual and not self.is_valid():
            raise ValueError(f"Device '{name}' is not a valid GeecsDevice.")

    def get(self, variable: str, use_state: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Get a variable for the standard device.

        Args:
            variable (str): The variable to get.
            use_state (bool): Whether to use cached state values instead of querying the devices. Defaults to `True`.
            **kwargs: Additional arguments passed to the `get` method of sub-devices.

        Returns:
            Dict[str, Any]: A dictionary containing the variable name and the results of a GEECS Device get
        """
        return {variable: super().get(variable, **kwargs)}

    def close(self) -> None:
        """Close devices after sleeping for 1s to avoid race condition."""
        super().unsubscribe_var_values()
        time.sleep(1)
        super().close()

    def is_composite(self) -> bool:
        """ Returns False in a check if this is a composite scan device """
        return False


class CompositeDevice(ScanDevice):
    """
    Composite variables can be of two types:
    - 'Set' type: Controls multiple devices using a relation, optionally using relative or absolute modes.
    - 'Get' type: Calculates a value using a relation based on readings from multiple devices.
    """
    def __init__(self, name: str, composite_spec_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize a `ScanDevice`, either as a standard device or as a composite device.

        Args:
            name (str): The name of the device.
            composite_spec_dict (Optional[Dict[str, Any]]): A dictionary defining a composite variable.
                If None, the ScanDevice is treated as a standard device.

                - Set Type (relative or absolute mode):
                  {
                      "components": [
                          {"device": "DeviceName1", "variable": "VariableName1", "relation": "composite_var * 1"},
                          {"device": "DeviceName2", "variable": "VariableName2", "relation": "composite_var * -1"}
                      ],
                      "mode": "relative"  # or "absolute"
                  }

                - Get Type (read-only mode):
                  {
                      "components": [
                          {"device": "DeviceName1", "variable": "VariableName1", "var_name": "var1"},
                          {"device": "DeviceName2", "variable": "VariableName2", "var_name": "var2"}
                      ],
                      "relation": "var1 + var2",
                      "mode": "get_only"
                  }

        Raises:
            ValueError: If the device name is invalid or the composite dictionary is incorrectly formatted.
            TypeError: If `composite_spec_dict` is not a dictionary.
        """
        # Composite Device Initialization (Virtual GeecsDevice)
        if not isinstance(composite_spec_dict, dict):
            raise TypeError("composite_spec_dict must be a dictionary if provided.")

        # Call `super().__init__(name, virtual=True)` to correctly inherit attributes like `state`
        super().__init__(name, virtual=True)

        self.components = composite_spec_dict['components']
        self.mode = composite_spec_dict['mode']
        self.relation = composite_spec_dict.get('relation', None)
        self.reference = {} if self.mode == "relative" else None
        self.sub_devices = {}

        # Initialize sub-devices and organize variables
        for comp in self.components:
            device_name = comp['device']
            if device_name not in self.sub_devices:
                self.sub_devices[device_name] = {
                    'instance': GeecsDevice(device_name),
                    'variables': []
                }
            self.sub_devices[device_name]['variables'].append(comp['variable'])

        # Subscribe to necessary variables for caching in `state`
        self._subscribe_to_variables()

        self.state["composite_var"] = 0

        if self.mode == "relative":
            time.sleep(1)
            self.set_reference()

    def _subscribe_to_variables(self) -> None:
        """
        Subscribe to all variables in the composite's sub-devices to enable caching in `state`.

        This ensures that variables can be accessed from the sub-device's cached state.

        Raises:
            ValueError: If subscribing to any variable on a sub-device fails.
        """
        for device_name, device_data in self.sub_devices.items():
            sub_device = device_data['instance']
            variables = device_data['variables']
            success = sub_device.subscribe_var_values(variables)
            if not success:
                raise ValueError(f"Failed to subscribe to variables {variables} on device '{device_name}'.")

        self.state["composite_var"] = 0

    def set_reference(self) -> None:
        """
        Set reference values for a composite device in relative mode.

        This initializes the `reference` dictionary with the current values of all sub-device variables,
        allowing adjustments to be made relative to these reference positions.

        Raises:
            ValueError: If the composite device is not in relative mode or if any variable's value
            cannot be retrieved from its sub-device.
        """
        if self.mode != "relative":
            raise ValueError("set_reference can only be used for composite variables in relative mode.")
        self.reference = {}
        for comp in self.components:
            device_name = comp['device']
            device_var = comp['variable']
            sub_device = self.sub_devices[device_name]['instance']
            current_value = sub_device.state.get(device_var)
            if current_value is None:
                raise ValueError(f"Failed to retrieve current value for '{device_var}' on '{device_name}'.")
            if device_name not in self.reference:
                self.reference[device_name] = {}
            self.reference[device_name][device_var] = current_value

    def set(self, variable: str, value: Any, **kwargs) -> None:
        """Set a variable for the composite or standard device."""
        if self.mode in ["relative", "absolute"] and variable == "composite_var":
            self.state["composite_var"] = value
            for comp in self.components:
                sub_device = self.sub_devices[comp['device']]['instance']
                device_var = comp['variable']
                sub_value = self._calculate_sub_value(comp, value)
                sub_device.set(device_var, sub_value, **kwargs)

            # right now, to be consistent with geecs device we are returning a value which is just the set value
            # In the future, a more clever/informative value could be returned that maybe takes into account
            # tolerances
            return value

        else:
            raise CompositeDeviceError("Set Error:  Incorrect Mode or variable not 'composite_var'")

    def _calculate_sub_value(self, comp: Dict[str, Any], value: Any) -> Any:
        """
        Calculate the value to set on a sub-device based on the composite relation.

        Uses `numexpr` to safely evaluate the relation.

        Args:
            comp (Dict[str, Any]): Component information containing device, variable, and relation.
            value (Any): The input composite variable value.

        Returns:
            Any: Computed sub-device value.
        """
        reference_value = (
            self.reference.get(comp['device'], {}).get(comp['variable'], 0)
            if self.mode == "relative" else 0
        )

        try:
            sub_value = ne.evaluate(comp['relation'], local_dict={"composite_var": value})
        except Exception as e:
            raise ValueError(f"Failed to evaluate relation: {comp['relation']}. Error: {e}")

        return reference_value + sub_value

    def get(self, variable: str, use_state: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Get a variable for the composite device.

        Uses `numexpr` to safely evaluate the composite relation.

        Args:
            variable (str): The variable to get. For composite devices, this must be `"composite_var"`.
            use_state (bool): Whether to use cached state values instead of querying the devices. Defaults to `True`.
            **kwargs: Additional arguments passed to the `get` method of sub-devices.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - `"composite_var"`: The last set value (for `set` mode) or evaluated value (for `get_only` mode).
                - Individual component values using `"DeviceName:VariableName"` keys.
        """
        result = {}

        if self.mode == "get_only" and variable == "composite_var":
            values = {}
            var_mapping = {}
            for comp in self.components:
                sub_device = self.sub_devices[comp['device']]['instance']
                device_var = comp['variable']
                key = f"{comp['device']}:{device_var}"
                var_mapping[comp['var_name']] = key
                values[key] = sub_device.state.get(device_var) if use_state else sub_device.get(device_var, **kwargs)

            # Prepare variable mapping for safe evaluation
            relation_vars = {var_name: values[mapped_key] for var_name, mapped_key in var_mapping.items()}

            try:
                composite_value = ne.evaluate(self.relation, local_dict=relation_vars)
            except Exception as e:
                raise ValueError(f"Failed to evaluate relation: {self.relation}. Error: {e}")

            self.state["composite_var"] = composite_value
            result["composite_var"] = composite_value
            result.update(values)
            return result

        elif self.mode in ["relative", "absolute"] and variable == "composite_var":
            for comp in self.components:
                sub_device = self.sub_devices[comp['device']]['instance']
                device_var = comp['variable']
                key = f"{comp['device']}:{device_var}"
                result[key] = sub_device.state.get(device_var) if use_state else sub_device.get(device_var, **kwargs)

            result["composite_var"] = self.state.get("composite_var", "NA")
            return result
        else:
            raise CompositeDeviceError("Get Error:  Incorrect Mode or variable not 'composite_var'")

    def close(self):
        """Close all sub-devices for composite devices."""
        for sub_device_data in self.sub_devices.values():
            sub_device_data['instance'].unsubscribe_var_values()
        time.sleep(1)
        for sub_device_data in self.sub_devices.values():
            sub_device_data['instance'].close()

    def is_composite(self) -> bool:
        """ Returns True in a check if this is a composite scan device """
        return True
