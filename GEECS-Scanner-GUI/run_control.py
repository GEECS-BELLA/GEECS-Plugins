from geecs_python_api.controls.interface import load_config
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interface.geecs_errors import ErrorAPI
from geecs_python_api.controls.data_acquisition import DeviceManager, DataLogger, DataInterface, setup_console_logging


def submit_run(config_dictionary, scan_config):
    # Step 4: Configure logging (log to both a file and the console)
    setup_console_logging(log_file="my_log.log", console=True)

    # Step 1: Initialize DeviceManager and load the configuration
    device_manager = DeviceManager(experiment_dir='HTU')
    device_manager.load_from_dictionary(config_dictionary)

    # Step 2: Set up data storage paths with DataInterface
    data_interface = DataInterface()

    # Step 3: Initialize the DataLogger for managing data acquisition and logging
    data_logger = DataLogger(device_manager, data_interface, experiment_dir='HTU')

    # Step 5: Define the scan configuration for device movements and data acquisition

    # Step 6: Start logging in a separate thread
    res = data_logger.start_logging_thread(scan_config=scan_config)

