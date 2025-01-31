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
    - 'Set' type: Controls multiple devices using a relation, optionally using relative or absolute modes.
    - 'Get' type: Calculates a value using a relation based on readings from multiple devices.
    
    """

    def __init__(self, name_or_dict: Union[str, Dict[str, Any]], virtual: bool = False):
        
        """
        Initialize a `ScanDevice`, either as a standard device or as a composite device.
        
        Args:
            
            name_or_dict (Union[str, Dict[str, Any]]): A device name (string) or a dictionary defining 
                a composite variable. For composite variables:
                - Set Type (relative or absolute mode):
                  {
                      "composite_name": {
                          "components": [
                              {"device": "DeviceName1", "variable": "VariableName1", "relation": "composite_var * 1"},
                              {"device": "DeviceName2", "variable": "VariableName2", "relation": "composite_var * -1"}
                          ],
                          "mode": "relative"  # or "absolute"
                      }
                  }
        
        
                - Get Type (read-only mode):
                  {
                      "composite_name": {
                          "components": [
                              {"device": "DeviceName1", "variable": "VariableName1", "var_name": "var1"},
                              {"device": "DeviceName2", "variable": "VariableName2", "var_name": "var2"}
                          ],
                          "relation": "var1 + var2",
                          "mode": "get_only"
                      }
                  }
        
            virtual (bool): Whether the device is virtual. Defaults to `False`.

        Raises:
            ValueError: If the device name is invalid or the composite dictionary is incorrectly formatted.
            TypeError: If `name_or_dict` is neither a string nor a dictionary.
        """
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
            
            self.name, composite_info = next(iter(name_or_dict.items()))
            
            super().__init__(self.name, virtual=True)

            self.is_composite = True
            self.components = composite_info['components']
            self.mode = composite_info['mode']
            self.relation = composite_info.get('relation', None)
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

            if self.mode == "relative":
                time.sleep(1)
                self.set_reference()
        else:
            raise TypeError("The first argument must be a device name (str) or a composite dictionary (dict).")


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
            sub_device = self.sub_devices[device_name]['instance']  # Access the GeecsDevice instance
            current_value = sub_device.state.get(device_var)
            if current_value is None:
                raise ValueError(f"Failed to retrieve current value for variable '{device_var}' on device '{device_name}'.")
            if device_name not in self.reference:
                self.reference[device_name] = {}
            self.reference[device_name][device_var] = current_value

    
    def set(self, variable: str, value: Any, **kwargs) -> None:
        """
        Set a variable for the composite or standard device.

        For composite devices:
        - In 'relative' mode, the value is applied as an adjustment to the reference position.
        - In 'absolute' mode, the value is directly mapped to sub-device variables based on their relations.

        Updates the internal state:
        - `"composite_var"` is updated to the set value.

        Args:
            variable (str): The variable to set. For composite devices, this must be `"composite_var"`.
            value (Any): The value to set.
            **kwargs: Additional arguments passed to the `set` method of sub-devices.
        """

        if self.is_composite and self.mode in ["relative", "absolute"] and variable == "composite_var":
            # Store last set value of composite_var
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
            super().set(variable, value, **kwargs)
            
            
    def _calculate_sub_value(self, comp: Dict[str, Any], value: Any) -> Any:
        """
        Calculate the value to set on a sub-device based on the composite relation.

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
        return reference_value + eval(comp['relation'], {'composite_var': value})
    
    def get(self, variable: str, use_state: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Get a variable for the composite or standard device.

        Standardized behavior:
        - Returns a dictionary where:
          - `"composite_var"` holds the last set value (for set-types) or the evaluated value (for get-types).
          - Individual component values are always included using `"DeviceName:VariableName"` format.

        Args:
            variable (str): The variable to get. For composite devices, this must be `"composite_var"`.
            use_state (bool): Whether to use cached state values instead of querying the devices. Defaults to `True`.
            **kwargs: Additional arguments passed to the `get` method of sub-devices.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - `"composite_var"`: The last set value (for `set` mode) or evaluated value (for `get_only` mode).
                - Individual component values using `"DeviceName:VariableName"` keys.

        Raises:
            ValueError: If a sub-device value cannot be retrieved.
        """

        result = {}

        # Handle GET-ONLY composites (evaluate relation and update state)
        if self.is_composite and self.mode == "get_only" and variable == "composite_var":
            values = {}
            var_mapping = {}  # Mapping for var_name -> actual values
            for comp in self.components:
                sub_device = self.sub_devices[comp['device']]['instance']
                device_var = comp['variable']
                key = f"{comp['device']}:{device_var}"  # Standardized key
                var_mapping[comp['var_name']] = key  # Map var_name to actual device-variable key
                values[key] = (
                    sub_device.state.get(device_var)
                    if use_state
                    else sub_device.get(device_var, **kwargs)
                )

            # Map var_names in self.relation to the actual values dictionary
            relation_mapped = self.relation
            for var_name, mapped_key in var_mapping.items():
                relation_mapped = relation_mapped.replace(var_name, f"values['{mapped_key}']")

            # Compute the composite variable value (RESTRICTED EVAL SCOPE)
            composite_value = eval(relation_mapped, {}, {"values": values})

            # Store result in internal state
            self.state["composite_var"] = composite_value

            # Construct return dictionary
            result["composite_var"] = composite_value
            result.update(values)
            return result

        # Handle SET-TYPE composites (return individual component values)
        elif self.is_composite and self.mode in ["relative", "absolute"] and variable == "composite_var":
            for comp in self.components:
                sub_device = self.sub_devices[comp['device']]['instance']
                device_var = comp['variable']
                key = f"{comp['device']}:{device_var}"
                result[key] = (
                    sub_device.state.get(device_var)
                    if use_state
                    else sub_device.get(device_var, **kwargs)
                )

            # Retrieve last set value of composite_var (or set to "NA" if never set)
            result["composite_var"] = self.state.get("composite_var", "NA")
            return result

        # Standard device get method
        return {variable: super().get(variable, **kwargs)}

    def close(self) -> None:

        """
        Close all sub-devices for composite devices.

        For composite devices:
        - Unsubscribes variables for each sub-device to clean up cached state.
        - Closes each sub-device instance.

        For standard devices:
        - Unsubscribes variables and closes the device.

        Notes:
            - Includes a short sleep to handle potential race conditions in the GEECS framework.
        """

        if self.is_composite:
            for sub_device_data in self.sub_devices.values():
                sub_device_data['instance'].unsubscribe_var_values()
                time.sleep(1)  # Handle race condition in GEECS
                sub_device_data['instance'].close()
        else:
            super().unsubscribe_var_values()
            time.sleep(1)  # Handle race condition in GEECS
            super().close()


