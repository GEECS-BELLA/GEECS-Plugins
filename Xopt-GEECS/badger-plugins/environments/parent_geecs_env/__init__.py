import torch
import time
import numpy as np
from typing import Dict
from badger import environment
from badger.interface import Interface


def process_observables(acquired_data):
    median_vals = {key: np.median(values) for key, values in acquired_data.items()}
    return median_vals


class Environment(environment.Environment):
    
    shots_per_step: int = 5

    _control_devices = {}
    _observable_devices = {}
    _variables = {}
    _observations = {}
    _max_retries = 3

    def get_variables(self, variable_names):
        assert self.interface, 'Must provide an interface!'
        self._control_devices = self.interface.initialize_subscribers(variable_names)
        results = self.interface.get_values(variable_names)
        print('variable_outputs: ',results)
        self.interface.close_subscribers(self._control_devices)
        values = {key: nested_dict['value'] for key, nested_dict in results.items()}
        return values

    def set_variables(self, variable_inputs: Dict[str, float],max_retries=_max_retries):
        self.interface.set_values(variable_inputs, max_retries)

    def get_observables(self, observable_names):
        return self.get_observables_geecs(observable_names)

    def get_observables_geecs(self, observable_names, device_acquisitions=None, target_function=None):
        # Manually specify device_acquisitions when observable is a single value calculated from multiple measurements
        if device_acquisitions is None:
            device_acquisitions = observable_names
        # Unless an alternative function is provided, the default target is just the median of each device acquisition
        if target_function is None:
            target_function = process_observables

        self._observable_devices = self.interface.initialize_subscribers(device_acquisitions)
        
        all_vals = {name: [] for name in device_acquisitions}  # Initialize the dictionary
        
        for i in range(1, self.shots_per_step):            
            # result = self.interface.get_values(observable_names)
            
            result = self.interface.get_fresh_values(device_acquisitions, max_attempts=3, wait_time=0.5)
            
            print('res in env step3: ', result)
            
            # Keys to exclude
            exclude_keys = ['fresh', 'shot number']
            
            # Update all_vals with values from new_dict
            for key, var in result.items():
                if var not in exclude_keys:
                    all_vals[f'{key}'].append(var['value'])
            print('all_vals: ',all_vals)
            
        self.interface.close_subscribers(self._observable_devices)
        
        print('all_vals: ',all_vals)
        objective_target = target_function(all_vals)

        return objective_target
