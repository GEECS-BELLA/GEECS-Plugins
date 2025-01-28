from __future__ import annotations
from typing import Optional, Any, Dict, Union

import time
from functools import wraps
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


class ScanDevice(GeecsDevice):
    """
    A class to represent a device or composite variable for scanning.

    Composite variables can be of two types:
    - 'Set' type: Controls multiple devices using a relation, optionally using relative or fixed modes.
    - 'Get' type: Calculates a value using a relation based on readings from multiple devices.

    Args:
        name_or_dict (Union[str, Dict[str, Any]]): Device name (string) or dictionary defining a composite variable.
        virtual (bool): Whether the device is virtual (default: False).
    """

    def __init__(self, name_or_dict: Union[str, Dict[str, Any]], virtual: bool = False):
        if isinstance(name_or_dict, str):
            # Standard GeecsDevice initialization
            super().__init__(name_or_dict, virtual)
            if self.is_valid():
                self.is_composite = False
                self.components = []
            else:
                raise ValueError(f"Device '{name_or_dict}' is not a valid GeecsDevice.")
        elif isinstance(name_or_dict, dict):
            if len(name_or_dict) != 1:
                raise ValueError("Composite device dictionary must contain exactly one top-level key.")

            # Dummy name for composite initialization
            super().__init__('virtual_composite_device', virtual=True)

            self.name, composite_info = next(iter(name_or_dict.items()))
            self.is_composite = True
            self.components = composite_info['components']
            self.mode = composite_info['mode']
            self.relation = composite_info.get('relation', None)
            self.reference = {} if self.mode == "relative" else None
            self.sub_devices = {comp['device']: GeecsDevice(comp['device']) for comp in self.components}

            # Subscribe to necessary variables for caching in `state`
            self._subscribe_to_variables()

            if self.mode == "relative":
                self.set_reference()
        else:
            raise TypeError("The first argument must be a device name (str) or a composite dictionary (dict).")

    def _subscribe_to_variables(self):
        """Subscribe to variables in all sub-devices for caching in `state`."""
        for comp in self.components:
            sub_device = self.sub_devices[comp['device']]
            device_var = comp['variable']
            success = sub_device.subscribe_var_values([device_var])
            if not success:
                raise ValueError(f"Failed to subscribe to variable '{device_var}' on device '{comp['device']}'.")

    def set_reference(self):
        """Set reference values for relative mode."""
        if self.mode != "relative":
            raise ValueError("set_reference can only be used for composite variables in relative mode.")
        self.reference = {
            comp['device']: {
                comp['variable']: self.sub_devices[comp['device']].state.get(comp['variable'])
            } for comp in self.components
        }

    def set(self, variable: str, value: Any, **kwargs):
        """Set a variable for the composite or standard device."""
        if self.is_composite and self.mode in ["relative", "fixed"] and variable == "composite_var":
            for comp in self.components:
                sub_device = self.sub_devices[comp['device']]
                device_var = comp['variable']

                reference_value = (
                    self.reference.get(comp['device'], {}).get(device_var, 0) if self.mode == "relative" else 0
                )
                sub_value = (
                    reference_value + eval(comp['relation'], {'composite_var': value})
                    if self.mode == "relative"
                    else eval(comp['relation'], {'composite_var': value})
                )
                sub_device.set(device_var, sub_value, **kwargs)
        else:
            super().set(variable, value, **kwargs)

    def get(self, variable: str, use_state: bool = True, **kwargs) -> Any:
        """
        Get a variable for the composite or standard device.

        Args:
            variable (str): Variable name.
            use_state (bool): Whether to use cached state values instead of querying the devices (default: False).
            **kwargs: Additional arguments for the `get` method.

        Returns:
            Any: Value of the variable.

        Raises:
            GetNotAllowedError: If trying to get a variable on a 'set' type composite device.
        """
        if self.is_composite and self.mode in ["relative", "fixed"]:
            raise GetNotAllowedError(
                f"Cannot get variable '{variable}' from a 'set' type composite device '{self.name}'."
            )

        if self.is_composite and self.mode == "get_only" and variable == "composite_var":
            values = {
                comp['var_name']: (
                    self.sub_devices[comp['device']].state.get(comp['variable'])
                    if use_state
                    else self.sub_devices[comp['device']].get(comp['variable'], **kwargs)
                )
                for comp in self.components
            }
            return eval(self.relation, values)
        else:
            return super().get(variable, **kwargs)

    def close(self):
        """Close all sub-devices for composite devices."""
        if self.is_composite:
            for sub_device in self.sub_devices.values():
                sub_device.unsubscribe_var_values()
                time.sleep(1) # sleep coniditon needed to deal with some race condition in GEECS Device
                sub_device.close()
                
        else:
            super().unsubscribe_var_values()
            time.sleep(1) # sleep coniditon needed to deal with some race condition in GEECS Device
            super().close()


