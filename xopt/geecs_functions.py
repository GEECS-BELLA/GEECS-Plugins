
import sys
import time
import numpy as np

sys.path.append('../GEECS-PythonAPI')
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice

import importlib

class GeecsXoptInterface:
    
    _instance = None
    
    def __new__(cls, experiment_name="Undulator", *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GeecsXoptInterface, cls).__new__(cls)
            # Set the exp_info when the class is instantiated
            GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(experiment_name)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance
        
    def _initialize(self, *args, **kwargs):
        # All initialization code goes here
        self.devices = {}
        self.backend_vocs = {}
        self.objectives_dict = {}
        self.controls_dict_for_tcp = {}
        # self.objectives_dict = {'var1':{'device_name':'UC_Amp2_IR_input','device_subscribe_variables':['MeanCounts']}}
        self.obj_instance=None
        self.subscribed_objective_devices={}
        self.objective_function_variables=[]
        self.objective_function_devices=[]
        self.devices_active=False
        
        
        # Any other attributes you want to initialize
        self._initialized = True
        
    def get_objective_function_class(self, class_name='default', module_name="objective_functions"):
        """
        Dynamically imports a module and retrieves a class based on its name.

        Args:
        - class_name (str): The name of the class to retrieve.
        - module_name (str, optional): The name of the module where the class is defined. Default is "opt_functions".

        Returns:
        - class: The retrieved class.
        """
        print('module name: ',module_name)
        print('class name: ',class_name)
        
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls
        
    # def initialize_objective(self, device_name, variable_name):
    #     device = GeecsDevice(device_name)
    #     self.objective_function_devices.append(device)
    #     self.objective_function_variables.append(variable_name)
    #     current_value=device.get(variable_name)
    #     return current_value
        
    def initialize_objective(self):
        self.tcp_subscription_factory()
        if self.subscribed_objective_devices=={}:
            self.devices_active=False
        else:
            self.devices_active=True
        
    def tcp_subscription_factory(self):
        for key in self.objectives_dict:
            device_name = self.objectives_dict[key]['device_name']
            device_variables = self.objectives_dict[key]['device_subscribe_variables']
            
            self.subscribed_objective_devices[key] = GeecsDevice(device_name)  # Assuming GeecsDevice takes the device_name as an argument
            self.subscribed_objective_devices[key].use_alias_in_TCP_subscription = False
            self.subscribed_objective_devices[key].subscribe_var_values(device_variables)
    
    def initialize_control(self, device_name, variable_name, min_value, max_value, use_current_pos, delta):
        # Here, replace this with the actual initialization logic for your control
        device = GeecsDevice(device_name)
        current_value=device.get(variable_name)
        
        if use_current_pos:
            min_bounds = current_value - float(delta)
            max_bounds = current_value + float(delta)
        else:
            min_bounds = float(min_value)
            max_bounds = float(max_value)
        
        # Create a key by concatenating the device name and variable name, separated by an underscore
        key = f"{device_name}::{variable_name}"
        self.devices[key]={"GEECS_Object":device,'variable':variable_name,"bounds":[min_bounds,max_bounds]}
        self.backend_vocs[key]=[min_bounds,max_bounds]
        
        # You might want to return some status or result from the initialization
        return current_value
        
    def initialize_all_controls(self, all_controls_data):
        results={}
        for control_data in all_controls_data:
            # Extract data from the current control data dictionary
            device_name = control_data["device_name"]
            variable_name = control_data["variable_name"]
            min_value = control_data["min_value"]
            max_value = control_data["max_value"]
            use_current_pos=control_data["use_current_pos"]
            delta=control_data['delta']

            # Initialize the control (assuming this method returns a result or status message)
            res=self.initialize_control(device_name, variable_name, min_value, max_value, use_current_pos, delta)
            key = f"{device_name}::{variable_name}"
            results[key]=res
        
        if results=={}:
            self.devices_active=False
        else:
            self.devices_active=True
        return results  # Return the list of results or status messages
        
    def close_all_controls(self):
        for key in self.devices:
            print(self.devices[key])
            geecs_obj = self.devices[key]['GEECS_Object']
            geecs_obj.close()

    def close_all_objectives(self):
        for key in self.subscribed_objective_devices:
            print(self.subscribed_objective_devices[key])
            self.subscribed_objective_devices[key].unsubscribe_var_values()
            time.sleep(0.2)
            # self.subscribed_objective_devices[key].close()

    def get_last_acquired_value(self, device_name, variable_name):
        key = f"{device_name}::{variable_name}"
        value=self.devices[key]['GEECS_Object'].get(variable_name)
        return value
    
    def unnormalize_controls(self, key, val):
        # Correctly access the bounds using the key
        bounds = self.backend_vocs[key]  # bounds is now a list [min, max]
        var_range = bounds[1] - bounds[0]  # Use indices to access elements within the list
        # print(var_range)
        offset = bounds[0]  # Corrected this line as well
        new_val = (val / 2 + 0.5) * var_range + offset
        return new_val
    
    def normalize_controls(self, key, val):
        bounds = self.backend_vocs[key]
        var_range = bounds[1] - bounds[0]  # Calculate the range using the correct indices
        offset = bounds[0]  # Get the correct offset
        # Calculate the new normalized value
        new_val = ((val - offset) / var_range - 0.5) * 2
        return new_val
    
    def tcp_subscription_factory(self):
        for key in self.objectives_dict:
            device_name = self.objectives_dict[key]['device_name']
            device_variables = self.objectives_dict[key]['device_subscribe_variables']
            
            self.subscribed_objective_devices[key] = GeecsDevice(device_name)  # Assuming GeecsDevice takes the device_name as an argument
            self.subscribed_objective_devices[key].use_alias_in_TCP_subscription = False
            self.subscribed_objective_devices[key].subscribe_var_values(device_variables)  
            
    
    def get_obj_var_tcp_state(self, wait_for_new=True):
        devices_needing_refresh = list(self.subscribed_objective_devices.keys())
        fresh_values = {}  # Dictionary to store the fresh values

        while devices_needing_refresh:
            for key in devices_needing_refresh.copy():  # Copy the list to prevent modification issues during iteration
                device = self.subscribed_objective_devices[key]
                state = device.state
                fresh = state.get('fresh', False)
        
                # Log the values
                for var in self.objectives_dict[key]['device_subscribe_variables']:
                    value = state.get(var)
                    # Log or print the values here
                    # print(f"{key} - {var}: {value} (Fresh: {fresh})")
        
                if fresh or not wait_for_new:
                    # Store the values
                    fresh_values[key] = {var: state.get(var) for var in self.objectives_dict[key]['device_subscribe_variables']}
            
                    # Mark the state as not fresh
                    device.state['fresh'] = False
                    # Remove the device from the list needing refresh
                    devices_needing_refresh.remove(key)
                else:
                    # Optionally wait for a short duration before checking again
                    # to prevent "busy-waiting" and consuming unnecessary resources
                    time.sleep(0.1)

        return fresh_values
        
    def set_from_dict(self,input_dict):
        for i in list(input_dict.keys()):
            try:
                set_val = float(input_dict[i])
                self.devices[i]["GEECS_Object"].set(self.devices[i]["variable"], set_val)
                time.sleep(0)

            except Exception as e:
                print(f"An error occurred: {e}")
                break
        

    def calcTransmission(self,input_dict):
        
        # some parameters for setting up the simulation case
        optPosition = np.array([18.45, 0.6])
        numParts = 200000
        
        startDist = np.transpose([
            np.random.normal(optPosition[0], 0.4, numParts),
            np.random.normal(optPosition[1], 0.4, numParts)
        ])
        
        center1 = [input_dict['U_Hexapod::ypos'], input_dict['U_Hexapod::zpos']]
        separation = 15

        center2 = [input_dict['U_Hexapod::ypos'], input_dict['U_Hexapod::zpos']]
        rotw = np.pi / 180 * (input_dict['U_Hexapod::wangle'] + 0.15) * 4
        rotv = np.pi / 180 * (input_dict['U_Hexapod::vangle'] + 0.25) * 4

        yOffset = separation * np.tan(rotw)
        zOffset = separation * np.tan(rotv)

        center2[0] = center2[0] + yOffset
        center2[1] = center2[1] + zOffset

        dist = startDist[
            (np.sqrt((startDist[:, 0] - center1[0])**2 + (startDist[:, 1] - center1[1])**2) < 0.2) &
            (np.sqrt((startDist[:, 0] - center2[0])**2 + (startDist[:, 1] - center2[1])**2) < 0.2)
        ]

        random_number_normal = np.random.normal(0, 0.0015)

        return len(dist) / numParts +random_number_normal
    