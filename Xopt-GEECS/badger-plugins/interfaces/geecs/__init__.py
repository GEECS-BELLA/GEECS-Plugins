import numpy as np
import time
import threading
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interface import load_config
from badger import interface

config = load_config()
if config and 'Experiment' in config and 'expt' in config['Experiment']:
    default_experiment = config['Experiment']['expt']
    print(f"default experiment is: {default_experiment}")
else:
    print("Configuration file not found or default experiment not defined. While use Undulator as experiment. Could be a problem for you.")
    default_experiment = 'Undulator'

GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(default_experiment)


class Interface(interface.Interface):

    name = 'geecs'
    devices: dict = {}
    testing: bool = False
    print("initializing interface")

    @staticmethod
    def get_default_params():
        return None

    def get_values(self, variables):
        
        # the 'variables' arg is a list with entries like ['GEECS_Device1:variable1','GEECS_Device1:variable2'
        # 'GEECS_Device1:variable1'...]. To efficiently and synchronously acquire multiple variables on a 
        # single device, TCP is the way. But, only a single TCP sucbscriber should be used that is logging
        # all relevant variables on a given device. The loop/logic below takes the variables list, identifies
        # all variables associated with a unique device  
        
        # The loop below populates an initially empty dict (results) with current geecs device states 
        # retrieved from TCP subscribers. 
        results = {}
        for var in variables:
            device_name, _ = var.split(':')
            if device_name not in results:
                results[device_name] = {}        
                var_list = [v.split(':')[1] for v in variables if v.startswith(device_name)]
                state = self.devices[device_name].state
                print('device state step1: ',state)
                for device_var in var_list:
                    results[device_name][device_var] = state[device_var]
                results[device_name]['fresh'] = state['fresh']
                results[device_name]['shot number'] = state['shot number']
                self.devices[device_name].state['fresh'] = False
        
        parsed_results = self.parse_tcp_states(results)
        
        # print('parsed_results: ',parsed_results)
             
        return parsed_results
        
    def get_fresh_values(self, observable_names, max_attempts=1, wait_time=0.01):
        print('in get fresh values')
        attempts = 0
        result = {}  # This will store the final results
        fresh_observables = set()  # To track observables that have been reported as fresh

        while attempts < max_attempts and len(fresh_observables) < len(observable_names):
            attempts += 1
            stale_observables = [name for name in observable_names if name not in fresh_observables]
            temp_result = self.get_values(stale_observables)
            print('temp result: ', temp_result)

            # Check each sub_dict for 'fresh'
            for name, sub_dict in temp_result.items():
                if sub_dict.get('fresh', False):
                    fresh_observables.add(name)
                    result[name] = sub_dict  # Update result with fresh data

            if len(fresh_observables) == len(observable_names):
                print('got fresh data for all observables')
                break  # Exit the loop if all observables have been reported as fresh
            
            time.sleep(wait_time)

        if len(fresh_observables) < len(observable_names):
            print('failed to get fresh data for all observables')

        return result

    def set_values(self, variable_inputs, max_retries = 1):
        
        for variable, value in variable_inputs.items():
            device_name, attribute = variable.split(':')
            retries = 0
            error_occured = False
            while retries < max_retries:
                device_response = self.devices[device_name].set(attribute, value) #edit by josh
                print('device response: ',device_response)
                time.sleep(0.21)
                device_state = self.devices[device_name].state
                device_error = device_state['GEECS device error']
                if device_error:
                    print("Hardware error from GEECS device, retrying...")
                    retries += 1
                    time.sleep(0.2)  # Sleep for 0.2 seconds before retrying
                    if retries >= max_retries:
                        print("Maximum retries reached. Breaking out of loops.")
                        error_occurred = True
                        break
                else:
                    print('No error in device')
                    break  # Exit retry loop if no error
                if error_occured:
                    print('device was not set, consider aborting')
                
    def initialize_subscribers(self,variables):
        
        # it seems that calling "get_variables" by using the "add current" button
        # reinitializes the whole environment and interface. Thus, self.devices
        # here gets set back to self.devices = {}. Calling "get_observables" does 
        # not reinitialize self.devices. To ensure all threads are dead before creating
        # new ones, the loop below closes out any existing threads. The root of the problem
        # here is that badger seems to "recreate the enivorment" for a variety of actions.
        # In doing so, it doesn't kill threads spawned by sub routines, which causes annoying
        # hangups requiring force quits of python. To circumvent, TCP subscribers are created
        # and destroyed at each "get_variables" and "get_observable" call.
        # UPDATE TO THE ABOVE:  
        
        for device_name, device in self.devices.items():
            try:
                print(f"Attempting to unsubscribe from {device_name}...")
                device.unsubscribe_var_values()
                device.close()
                print(f"Successfully unsubscribed from {device_name}.")
            except Exception as e:
                print(f"Error unsubscribing from {device_name}: {e}")
        self.devices = {}
        
        # this first loop looks for the unique device names, creates GeecsDevice objects
        # then creates a tcp subscriber with associated variables
        for var in variables:
            device_name, _ = var.split(':')
            if device_name not in self.devices:
                device = GeecsDevice(device_name)  
                var_list = [v.split(':')[1] for v in variables if v.startswith(device_name)]
                device.use_alias_in_TCP_subscription = False
                print('var_list for subscription: ',var_list)
                device.subscribe_var_values(var_list)  
                self.devices[device_name] = device
        
        # after all of the subscribers have been created, this loop goes through and waits until the 
        # first valid entry for the variables has been updated in the state.        
        for var in variables:
            device_name, _ = var.split(':')
            timeout = 10
            t0 = time.monotonic()
            t1 = t0
            
            while self.devices[device_name].state.get(var.split(':')[1]) is None and t1-t0 < timeout:
                time.sleep(0.2)
                t1 = time.monotonic()
            if t1-t0 > timeout:
                print('timeout; tcp subscriber didnt update correcly')
            
             
                                
        # List all active threads. This is used for debugging to make sure no unwanted threads are hanging
        active_threads = threading.enumerate()
        for thread in active_threads:
            print(thread.name)
            
        return self.devices
        
    def close_subscribers(self,devices):
        
        print('in close subscrbers, devices: ',devices)
        for device_name, device in self.devices.items():
            try:
                print(f"Attempting to unsubscribe from {device_name}...")
                device.unsubscribe_var_values()
                print(f"Successfully unsubscribed from {device_name}.")
            except Exception as e:
                print(f"Error unsubscribing from {device_name}: {e}")
                
        # List all active threads
        active_threads = threading.enumerate()

        for thread in active_threads:
            print(thread.name)
            
        return 
        
    def parse_tcp_states(self, get_values_result):
        
        # Takes a result from get_values that aggregates variabeles by device in TCP states and 
        # parses them back into a dict that is in line with single_variable:single_value
        data = get_values_result

        parsed_dict = {}
        
        shared_keys = ['fresh', 'shot number']
        
        for top_key, nested_dict in data.items():
            common_data = {k: v for k, v in nested_dict.items() if k in shared_keys}
            for nested_key, value in nested_dict.items():
                if nested_key not in common_data:
                    new_key = f'{top_key}:{nested_key}'
                    parsed_dict[new_key] = {'value': value, **common_data}
        
                    
        return parsed_dict
        

