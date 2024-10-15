from geecs_python_api.controls.data_acquisition import DeviceManager, DataLogger, DataInterface


class RunControl:
    def __init__(self, experiment_name=""):
        self.experiment_name = experiment_name
        self.device_manager = None
        self.data_interface = None
        self.data_logger = None
        self.is_in_setup = False
        self.is_in_stopping = False

    def submit_run(self, config_dictionary, scan_config):
        self.is_in_setup = True

        # Step 1: Initialize DeviceManager and load the configuration
        self.device_manager = DeviceManager(experiment_dir='HTU')
        self.device_manager.load_from_dictionary(config_dictionary)

        # Step 2: Set up data storage paths with DataInterface
        self.data_interface = DataInterface()

        # Step 3: Initialize the DataLogger for managing data acquisition and logging
        self.data_logger = DataLogger(experiment_dir='HTU', device_manager=self.device_manager, data_interface=self.data_interface)

        # Step 5: Define the scan configuration for device movements and data acquisition

        # Step 6: Start logging in a separate thread
        res = self.data_logger.start_logging_thread(scan_config=scan_config)
        self.is_in_setup = False

    def is_busy(self):
        return self.is_in_setup

    def is_initialized(self):
        return not (self.device_manager is None and self.data_interface is None and self.data_logger is None)

    def is_active(self):
        if not self.is_initialized():
            return False
        return self.data_logger.is_logging_active()

    def stop_scan(self):
        if self.is_initialized() and not self.is_stopping():
            self.is_in_stopping = True
            self.data_logger.stop_logging_thread_event.set()
            self.device_manager = None
            self.data_interface = None
            self.data_logger = None

    def is_stopping(self):
        return self.is_in_stopping

    def clear_stop_state(self):
        self.is_in_stopping = False
