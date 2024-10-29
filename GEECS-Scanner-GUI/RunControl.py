from geecs_python_api.controls.data_acquisition.scan_manager import ScanManager


class RunControl:
    def __init__(self, experiment_name="", shot_control=""):
        if experiment_name == "" or shot_control == "":
            print("Specify experiment name and shot control device")
            self.scan_manager = None
        else:
            self.scan_manager = ScanManager(experiment_dir=experiment_name, shot_control_device=shot_control)

        self.is_in_setup = False
        self.is_in_stopping = False

    def get_database_dict(self):
        if self.scan_manager is None:
            return None
        else:
            return self.scan_manager.get_database_dict()

    def submit_run(self, config_dictionary, scan_config):
        if self.scan_manager is not None:
            self.is_in_setup = True

            self.scan_manager.reinitialize(config_path=None, config_dictionary=config_dictionary)
            self.scan_manager.start_scan_thread(scan_config=scan_config)

            self.is_in_setup = False

    def get_progress(self):
        if self.scan_manager is not None:
            return self.scan_manager.estimate_current_completion()*100
        else:
            return 0

    def is_busy(self):
        return self.is_in_setup

    def is_active(self):
        if self.scan_manager is not None:
            return self.scan_manager.is_scanning_active()
        else:
            return False

    def stop_scan(self):
        if self.scan_manager is not None:
            if not self.is_stopping():
                self.is_in_stopping = True
                self.scan_manager.stop_scanning_thread()

    def is_stopping(self):
        return self.is_in_stopping

    def clear_stop_state(self):
        self.is_in_stopping = False
