import torch
import time
import numpy as np
from typing import Dict
from badger import environment
from badger.interface import Interface


class Environment(environment.Environment):

    name = 'camera_exposure_time_test'

    variables = {
        'UC_ChicaneSlit:exposure': [0.01, .9],
        'UC_ChicaneSlit:triggerdelay': [300.0, 600.0],
        'UC_Probe:exposure': [0.01,.5]
    }

    observables = ['UC_ChicaneSlit:meancounts']
    
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
            result = self.interface.get_values(observable_names)
            
            # result = self.get_fresh_values(observable_names, max_attempts=3, wait_time=0.5)
            
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
        
    def get_fresh_values(self, observable_names, max_attempts=1, wait_time=0.5):
        print('in get fresh values')
        attempts = 0
        result = {}
        temp_observable_names = observable_names
        stale_observables = observable_names

        while attempts < max_attempts:
            attempts += 1
            temp_result = self.interface.get_values(stale_observables)
            print('temp result: ',temp_result)
            
            for key,val in temp_result.items():
                if key == 'fresh':
                    if val:
                        print('got fresh_data for: ',key)
                        result[key] = obs_data  # Update with fresh data

            # for obs_name, obs_data in temp_result.items():
            #     if obs_data.get('fresh', True):
            #         print('got fresh_data for: ',obs_name)
            #         result[obs_name] = obs_data  # Update with fresh data
            #
            #         if obs_name in stale_observables:
            #             stale_observables.remove(obs_name)
            #         print('current result: ', result)
            #         print('stale_observables: ',stale_observables)
            #     else:
            #         print('got stale data for: ',obs_data)
            #         print('stale_observables: ',stale_observables)
                    
                    
                    
                    
                    
            #         result[obs_name] = obs_data  # Update with fresh data

            # if not fresh_observables:  # Break the loop if all observables are fresh
            #     break
            # else:
            #     observable_names = fresh_observables  # Update the list for the next attempt
            #     time.sleep(wait_time)

        return temp_result
    