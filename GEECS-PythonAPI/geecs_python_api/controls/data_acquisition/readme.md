# Data Acquisition Module

This module provides functionality for controlling and logging data from multiple devices in a synchronized manner. It includes classes to manage device communication, configure data paths, and handle logging operations, including triggering systems for precise data acquisition. The main components are the `DeviceManager`, `DataInterface`, and `DataLogger` classes.

## Key Components

### 1. `DeviceManager`
This class is responsible for:
- Managing the initialization and subscription to devices.
- Handling TCP subscriptions and receiving data from various devices.
- Parsing and organizing data received from devices into structured formats.
- Managing event-driven observables and asynchronous observables.

### 2. `DataInterface`
This class is responsible for:
- Managing data storage paths based on the system's domain and current date.
- Constructing paths for raw data and analysis data.
- Providing utility methods to manage directories and determine scan folders.

### 3. `DataLogger`
This class is responsible for:
- Managing the actual logging process, including:
  - Synchronizing the data collection process.
  - Polling data from asynchronous devices.
  - Triggering system control to turn data acquisition on and off.
  - Handling and processing scan configurations for complex data collection workflows.
- Converting logged data into structured formats such as DataFrames, TDMS files, and text/HDF5 formats.

### 4. Logging Setup Function (`setup_logging`)
Note, this is a function to log information on the execution of the python code, not on the data acquistion. The usage of the term 'logging' should be better disambiguated. The `setup_logging` function is used to configure the logging system. It allows logging to a file and optionally to the console for debugging or monitoring.

## To do, known issues and key differences from Master Control:
### 1. Time based data acquisition: 
In contrast to MasterControl which uses ShotNumber and number of shots to define how data is acquired, this approach uses a different approach. Synchronization is determined using a timestamp generated during the acquisition loop of the GEECS labview driver. Note: this means the "timestamp" variable must be added to a devicetype for which you want synchronization. Currently support devices are Point Grey Camera and PicoscopeV2. It's trivial to extend this additional device types.

Using this approach, rather than specify a number of required shots, the user specifies an amount of time to acquire data after each step. This does mean that occasionaly you might get 1 more or less shot than "expected" at each interval. 

It's likely possible to implement a more "shotnumber" based approach, but the time based approach is likely sufficient for most cases. 

### 2. Generalize beyond HTU: 
There are a few parts of the code that are still hardcoded to use HTU specific information. This should be better generalized

### 3. Extend use of config file: 
There are entries for scan info and scan parameters in the config file, but they aren't actually used yet. This information should be parsed and passed along to be use to start scans etc. 

### 4. Add methods to create ScanInfo and Scan.ini files like MC: 
Info from the config file should be used to create the MC style .ini and info files. 

### 5. Develop 'complex actions' that can be used in pre/post-scan-actions: 
Maybe there should an Actions class that could read a config file to perform a sequence of actions in a specific order and ensuring everything is performed correctly. For example, for HTU, 'insert_pmq' would need to first check the laser is shuttered (close Gaia internal shutters, insert gaia stops, read gaia stop status, insert amp4 dump, read amp4 dump status, insert pmq triplet, etc.) With basic actions like this defined, pre and post scan actions could be stacked into the config file to be executed. 


## Usage Example

As this is a module that relies on other components in the geecs_python_api, it is important to load the necessary dependencies. Note, this developed under the Xopt-GEECS conda environment

```python
from geecs_python_api.controls.interface import load_config
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interface.geecs_errors import ErrorAPI
from geecs_python_api.controls.data_acquisition import DeviceManager, DataLogger, DataInterface, setup_logging

config = load_config()
if config and 'Experiment' in config and 'expt' in config['Experiment']:
    default_experiment = config['Experiment']['expt']
    print(f"default experiment is: {default_experiment}")
else:
    print("Configuration file not found or default experiment not defined. While use Undulator as experiment. Could be a problem for you.")
    default_experiment = 'Undulator'

GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(default_experiment)

```

Next, the dataquisition module can be imported and used. Here's an example of how to use the `DeviceManager`, `DataInterface`, and `DataLogger` classes for controlling devices and logging data.

```python
from geecs_python_api.controls.data_acquisition import DeviceManager, DataLogger, DataInterface, setup_logging

# Step 1: Initialize DeviceManager and load the configuration
device_manager = DeviceManager()
device_manager.load_from_config('config.yaml')

# Step 2: Set up data storage paths with DataInterface
data_interface = DataInterface()

# Step 3: Initialize the DataLogger for managing data acquisition and logging
data_logger = DataLogger(device_manager, data_interface)

# Step 4: Configure logging (log to both a file and the console)
setup_logging(log_file="my_log.log", console=True)

# Step 5: Define the scan configuration for device movements and data acquisition
scan_config = [
    {'device_var': 'U_ESP_JetXYZ:Position.Axis 1', 'start': 4, 'end': 6, 'step': 0.5, 'wait_time': 5.5}
]

# Step 6: Start logging in a separate thread
res = data_logger.start_logging_thread(scan_config=scan_config)

```

### Explanation of Example:

- **Step 1**: The `DeviceManager` loads device configurations from a YAML file (e.g., `config.yaml`) and subscribes to the devices.
- **Step 2**: The `DataInterface` is responsible for managing the paths where the data will be saved.
- **Step 3**: The `DataLogger` is initialized to control data logging operations.
- **Step 4**: Logging is configured to write logs to a file and optionally to the console.
- **Step 5**: A scan configuration is defined, specifying how devices should move and when data should be logged.
- **Step 6**: Logging starts in a separate thread, where devices are moved, and data is logged in real-time.

### Config File (`config.yaml`)

The `config.yaml` file defines the devices to be used, their variables, and other configurations like saving data settings. Here is a sample structure of the config file:

```yaml
scan_info:
  experiment: "Example Experiment"
  description: "A sample configuration for data acquisition"

scan_parameters:
  scan_mode: "Statistics"
  scan_range: [0, 10]

Devices:
  U_ESP_JetXYZ:
    variable_list: ['Position.Axis 1', 'Position.Axis 2']
    synchronous: true
    save_nonscalar_data: false

  U_LaserEnergyMLinkPicoscope:
    variable_list: ['Python Results.ChA', 'Python Results.ChB']
    synchronous: false
    save_nonscalar_data: true
```