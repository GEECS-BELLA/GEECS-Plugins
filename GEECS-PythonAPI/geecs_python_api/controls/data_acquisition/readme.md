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
The `setup_logging` function is used to configure the logging system. It allows logging to a file and optionally to the console for debugging or monitoring.

## Usage Example

Here's an example of how to use the `DeviceManager`, `DataInterface`, and `DataLogger` classes for controlling devices and logging data.

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

## How to Extend or Modify:
### 1. Adding new devices: Update the config.yaml file with new device names and their respective variables.
### 2. Custom scan configurations: Modify the scan_config structure in the script to define custom movements or actions for different devices.
### 3. Logging formats: Extend the DataLogger class to add support for different data output formats like CSV, JSON, or custom formats.