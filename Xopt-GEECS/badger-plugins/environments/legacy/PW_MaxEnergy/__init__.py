import torch
import time
import numpy as np
from typing import Dict
from badger import environment
from badger.interface import Interface


class Environment(environment.Environment):

    name = 'PW_MaxEnergy'

    variables = {
        'HEX-PL1-2:ypos': [3.6, 4.3],
        'HEX-PL1-2:zpos': [-3, -2]
    }

    observables = ['EM-HPD-1:data']

    some_parameter: str = 'test'

    # def __init__(self,interface: Interface, params=None):
    #     time.sleep(.1)
    #     super().__init__()
    #     self.interface = interface
    #     assert self.interface, 'Must provide an interface!'

    _control_devices = {}
    _observable_devices = {}
    _variables = {}
    _observations = {}
    _max_retries = 3

    def get_variables(self, variable_names):
        print('self.some_parameter: ',self.some_parameter)
        print('in environment get_variables. variable names:',self.variable_names)
        assert self.interface, 'Must provide an interface!'
        self._control_devices = self.interface.initialize_subscribers(variable_names)
        results = self.interface.get_values(variable_names)
        print('variable_outputs: ',results)
        self.interface.close_subscribers(self._control_devices)
        values = {key: nested_dict['value'] for key, nested_dict in results.items()}
        return values

        # # Keys to exclude
        # exclude_keys = ['fresh', 'shot number']
        #
        # # Parsing the dictionary
        # variable_outputs = {
        #     f'{outer_key}:{inner_key}': inner_value
        #     for outer_key, inner_dict in results.items()
        #     for inner_key, inner_value in inner_dict.items()
        #     if inner_key not in exclude_keys
        # }
        # return variable_outputs

    def set_variables(self, variable_inputs: Dict[str, float],max_retries=_max_retries):
        self.interface.set_values(variable_inputs,max_retries)

    def get_observables(self, observable_names):

        self._observable_devices = self.interface.initialize_subscribers(observable_names)

        all_vals = {name: [] for name in observable_names}  # Initialize the dictionary

        number_of_shots = 10 #this should be a setting somewhere
        for i in range(1, number_of_shots):
            # result = self.interface.get_values(observable_names)

            result = self.get_fresh_values(observable_names, max_attempts=3, wait_time=0.5)

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
        # median_vals = {}
        # for key, values in all_vals.items():
        #     if values:  # Check if the list is not empty
        #         median_vals[key] = np.median(values)
        #     else:
        #         median_vals[key] = None  # Or some other placeholder for empty lists
        median_vals = {key: np.median(values) for key, values in all_vals.items()}


        return median_vals

    def get_fresh_values(self, observable_names, max_attempts=3, wait_time=0.01):

        # a first attempt at making a method that requires the get_values method to return a 'fresh' result

        print('in get fresh values')
        attempts = 0
        result = {}
        temp_observable_names = observable_names
        stale_observables = observable_names

        got_fresh_data = False
        while attempts < max_attempts:
            attempts += 1
            temp_result = self.interface.get_values(stale_observables)
            print('temp result: ',temp_result)

            if all(sub_dict.get('fresh', False) for sub_dict in temp_result.values()):
                got_fresh_data = True
                break  # Exit the loop if all 'fresh' values are True

            time.sleep(wait_time)

        if got_fresh_data:
            print('got fresh data')
        else:
            print('failed to get a new value')

        return temp_result
