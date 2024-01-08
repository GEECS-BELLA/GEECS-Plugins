import numpy as np
import time
import threading
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice

GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

from badger import interface

class Interface(interface.Interface):

    name = 'geecs'
    devices: dict = {}
    testing: bool = False
    print("initializing interface")

    @staticmethod
    def get_default_params():
        return None

    def get_values(self, variables):
        values = {}
        # print('in get_values, variables: ',variables)
        for var in variables:
            print(var)
            device_name, variable = var.split(':')
            device = self.devices[device_name]

            # if self.testing:
            #     values[var] = np.random.random()
            #     continue

            try:
                value = device.state[variable]  # Replace with actual method to get device state
                if (value is None) or np.isnan(value):
                    raise ValueError(f"Value for {var} is invalid.")
                values[var] = value
                device.unsubscribe_var_values()
                
            except Exception as e:
                values[var] = None
                print(f"Error getting value for {var}: {e}")
        print(values)
        return values

    def set_values(self, variable_inputs):
        # print('in interface set_values:',variable_inputs)
        # print('in interface self.devices:',self.devices)

        for variable, value in variable_inputs.items():
            device_name, attribute = variable.split(':')
            device_response = self.devices[device_name].set(attribute, value)
            print('device response: ',device_response)
            time.sleep(.1)
                
    def initialize_subscribers(self,variables):
        # print(variables)
        # print(self.devices)
        #
        print('in initialize subscrbers, self.devices: ',self.devices)
        for device_name, device in self.devices.items():
            try:
                print(f"Attempting to unsubscribe from {device_name}...")
                device.unsubscribe_var_values()
                print(f"Successfully unsubscribed from {device_name}.")
            except Exception as e:
                print(f"Error unsubscribing from {device_name}: {e}")

        
        for var in variables:
            device_name, _ = var.split(':')
            if device_name not in self.devices:
                print(device_name)
                device = GeecsDevice(device_name)  # Replace with actual device initialization
                var_list = [v.split(':')[1] for v in variables if v.startswith(device_name)]
                device.use_alias_in_TCP_subscription = False
                device.subscribe_var_values(var_list)  # Replace with actual subscription method
                self.devices[device_name] = device
                time.sleep(1.1)
                print(device.state)
                
        print('in initialize subscrbers (end), self.devices : ',self.devices)
                
        # List all active threads
        active_threads = threading.enumerate()

        for thread in active_threads:
            print(thread.name)
        


