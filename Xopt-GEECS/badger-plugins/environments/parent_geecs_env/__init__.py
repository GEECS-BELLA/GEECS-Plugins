import torch
import time
import numpy as np
from typing import Dict
from badger import environment
from badger.interface import Interface
from functools import reduce
from operator import mul

class Environment(environment.Environment):
    shots_per_step: int = 5
    max_fresh_retries: int = 3
    fresh_wait_time: float = 0.5

    _control_devices = {}
    _observable_devices = {}
    _variables = {}
    _observations = {}
    _max_retries = 3

    def get_variables(self, variable_names):
        assert self.interface, 'Must provide an interface!'
        self._control_devices = self.interface.initialize_subscribers(variable_names)
        results = self.interface.get_values(variable_names)
        print('variable_outputs: ', results)
        self.interface.close_subscribers(self._control_devices)
        values = {key: nested_dict['value'] for key, nested_dict in results.items()}
        return values

    def set_variables(self, variable_inputs: Dict[str, float], max_retries=_max_retries):
        self.interface.set_values(variable_inputs, max_retries)

    def get_observables(self, observable_names):
        return self.get_observables_geecs(observable_names)

    def get_observables_geecs(self, observable_names, device_acquisitions=None, target_function=None):
        # Manually specify device_acquisitions when observable is a single value calculated from multiple measurements
        if device_acquisitions is None:
            device_acquisitions = observable_names
        # Unless an alternative function is provided, the default target is just the median of each device acquisition
        if target_function is None:
            target_function = self.return_median_value

        self._observable_devices = self.interface.initialize_subscribers(device_acquisitions)
        all_vals = {name: [] for name in device_acquisitions}  # Initialize the dictionary

        for i in range(1, self.shots_per_step):
            result = self.interface.get_fresh_values(device_acquisitions,
                                                     max_attempts=self.max_fresh_retries,
                                                     wait_time=self.fresh_wait_time)

            print('res in env step3:', result)

            # Keys to exclude
            exclude_keys = ['fresh', 'shot number']

            # Update all_vals with values from new_dict
            for key, var in result.items():
                if var not in exclude_keys:
                    all_vals[f'{key}'].append(var['value'])
            print('all_vals: ', all_vals)

        self.interface.close_subscribers(self._observable_devices)

        print('all_vals: ', all_vals)
        return target_function(all_vals)

    # Below are two common use-cases:  simply returning the median, and returning the result of a Gaussian target func.
    @staticmethod
    def return_median_value(acquired_data):
        median_vals = {key: np.median(values) for key, values in acquired_data.items()}
        return median_vals

    def return_multi_target(self, acquired_data, target_vals, observables, sigma_vals=None, acquisition_keys=None):
        median_vals = self.return_median_value(acquired_data)

        # Check that we have the acquisition keys if there was more than one
        if acquisition_keys is None:
            if len(median_vals) > 1:
                print("When there is more than one device acquisition, need acquisition_keys to determine ordering")
                return {}
            else:
                acquisition_keys = list(median_vals.keys())

        # Make sure our targets and sigmas are lists, and make up sigmas if they weren't provided
        if isinstance(target_vals, (int, float)):
            target_vals = [target_vals]
        if sigma_vals is None:
            sigma_vals = [1 if target == 0 else target/5 for target in target_vals]
        elif isinstance(sigma_vals, (int, float)):
            sigma_vals = [target_vals]

        # Calculate the result of a simple Gaussian target function for each pair of median value & target
        return_dict = {}
        for key in acquisition_keys:
            index = acquisition_keys.index(key)
            target_function = np.exp(-np.square(median_vals.get(key) - target_vals[index])/np.square(sigma_vals[index]))
            print("Key: ", key)
            print("Value: ", median_vals.get(key))
            print("Target: ", target_vals[index])
            print("Target Value: ", target_function)
            return_dict.update({key: target_function})

        # If we are expecting one observable, multiply the target functions together
        if len(observables) == 1:
            combined_target = reduce(mul, return_dict.values(), 1)
            return_dict = {observables[0]: combined_target}

        print("Target Return: ", return_dict)
        return return_dict
