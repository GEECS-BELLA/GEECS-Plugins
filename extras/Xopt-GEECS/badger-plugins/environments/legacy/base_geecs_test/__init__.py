import torch
import time
import numpy as np
from typing import Dict
from badger import environment
from badger.interface import Interface


class Environment(environment.Environment):

    name = 'base_geecs_test'

    variables = {
        'UC_ChicaneSlit:exposure': [0.01, .9],
        'UC_ChicaneSlit:triggerdelay': [300.0, 600.0],
        'UC_Probe:exposure': [0.01,.5]
    }

    observables = ['UC_ChicaneSlit:meancounts']

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
        self.interface.set_values(variable_inputs,max_retries)

    def get_observables(self, observable_names):

        self._observable_devices = self.interface.initialize_subscribers(observable_names)

        all_vals = {name: [] for name in observable_names}  # Initialize the dictionary

        for i in range(1, self.shots_per_step):
            # result = self.interface.get_values(observable_names)

            result = self.interface.get_fresh_values(observable_names, max_attempts=3, wait_time=0.5)

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

        objective_target = self.process_observables(all_vals)

        return objective_target

    def process_observables(self, acquired_data):

        median_vals = {key: np.median(values) for key, values in acquired_data.items()}

        return median_vals
