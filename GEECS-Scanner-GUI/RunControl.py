from geecs_python_api.controls.data_acquisition.scan_manager import ScanManager


class RunControl:
    def __init__(self, experiment_name=None):

        self.scan_manager = ScanManager(experiment_name)

        self.is_in_setup = False
        self.is_in_stopping = False

    def submit_run(self, config_dictionary, scan_config):
        self.is_in_setup = True

        self.scan_manager.reinitialize(config_path=None, config_dictionary=config_dictionary)
        self.scan_manager.start_scan_thread(scan_config=scan_config)

        self.is_in_setup = False

    def is_busy(self):
        return self.is_in_setup

    def is_active(self):
        return self.scan_manager.is_scanning_active()

    def stop_scan(self):
        if not self.is_stopping():
            self.is_in_stopping = True
            self.scan_manager.stop_scanning_thread()

    def is_stopping(self):
        return self.is_in_stopping

    def clear_stop_state(self):
        self.is_in_stopping = False
