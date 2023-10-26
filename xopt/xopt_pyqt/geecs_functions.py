
import sys
import time
import numpy as np

sys.path.append('../../GEECS-PythonAPI')
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice

GeecsDevice.exp_info = GeecsDatabase.collect_exp_info("Undulator")

class GeecsXoptInterface:
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GeecsXoptInterface, cls).__new__(cls)
            # Place your initialization here
            cls._instance.devices = {}
            cls._instance.backend_vocs = {}
            # Initialize other properties
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self):
        # Your existing __init__ code goes here, for example:

        self.devices = {}  # A dict that will store the initialized device controls
        self.backend_vocs={}
        
        # some parameters for setting up the simulation case
        self.optPosition = np.array([18.45, 0.6])
        self.numParts = 200000

        self.startDist = np.transpose([
            np.random.normal(self.optPosition[0], 0.4, self.numParts),
            np.random.normal(self.optPosition[1], 0.4, self.numParts)
        ])
    
    def initialize_control(self, device_name, variable_name, min_value, max_value):
        # Here, replace this with the actual initialization logic for your control
        device = GeecsDevice(device_name)
        
        # Create a key by concatenating the device name and variable name, separated by an underscore
        key = f"{device_name}_{variable_name}"
        self.devices[key]={"GEECS_Object":device,'variable':variable_name,"bounds":[float(min_value),float(max_value)]}
        self.backend_vocs[key]=[float(min_value),float(max_value)]
        
        current_value=device.get(variable_name)
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

            # Initialize the control (assuming this method returns a result or status message)
            res=self.initialize_control(device_name, variable_name, min_value, max_value)
            key = f"{device_name}_{variable_name}"
            results[key]=res
           
        return results  # Return the list of results or status messages

    def get_last_acquired_value(self, device_name, variable_name):
        key = f"{device_name}_{variable_name}"
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
        # Access the bounds using the key, which retrieves a list [min, max]
        bounds = self.backend_vocs[key]
        var_range = bounds[1] - bounds[0]  # Calculate the range using the correct indices
        offset = bounds[0]  # Get the correct offset

        # Calculate the new normalized value
        new_val = ((val - offset) / var_range - 0.5) * 2
        return new_val

        
    def calcTransmission(self,input_dict):
        center1 = [input_dict['U_Hexapod_ypos'], input_dict['U_Hexapod_zpos']]
        separation = 15

        center2 = [input_dict['U_Hexapod_ypos'], input_dict['U_Hexapod_zpos']]
        rotw = np.pi / 180 * (input_dict['U_Hexapod_wangle'] + 0.15) * 4
        rotv = np.pi / 180 * (input_dict['U_Hexapod_vangle'] + 0.25) * 4

        yOffset = separation * np.tan(rotw)
        zOffset = separation * np.tan(rotv)

        center2[0] = center2[0] + yOffset
        center2[1] = center2[1] + zOffset

        dist = self.startDist[
            (np.sqrt((self.startDist[:, 0] - center1[0])**2 + (self.startDist[:, 1] - center1[1])**2) < 0.2) &
            (np.sqrt((self.startDist[:, 0] - center2[0])**2 + (self.startDist[:, 1] - center2[1])**2) < 0.2)
        ]

        return len(dist) / self.numParts
    