"""
Watches the scan folders for a given day and reports any new scans
"""

import os
import time
import yaml
import threading
from devices_to_analysis_mapping import check_for_analysis_match
from image_analysis.analyzers.online_analysis_modules.directory_functions import compile_daily_path


class NewScanLookout:
    def __init__(self, exp, year, month, day, overwrite_previous=False, ignore_list=None, check_interval=2, do_print=False):
        self.experiment = exp
        self.year = year
        self.month = month
        self.day = day
        self.root_folder = compile_daily_path(self.day, self.month, self.year, experiment=self.experiment)
        self.processed_list_filename = f"processed_scans_{self.experiment}.yaml"

        self.processed_list = []
        if ignore_list is not None:
            self.processed_list = ignore_list
        if not overwrite_previous:
            self.read_processed_list()

        self.check_interval = check_interval

        self.stop_event = threading.Event()
        self.lookout_thread = threading.Thread(target=self.scan_folder_check_loop)

        self.do_print = do_print

    def start_lookout(self):
        self.lookout_thread.start()
        return self.lookout_thread

    def scan_folder_check_loop(self):
        while not self.stop_event.is_set():
            self.check_for_new_files()
            time.sleep(self.check_interval)

    def stop_lookout(self):
        self.stop_event.set()
        self.lookout_thread.join()

    def check_for_new_files(self):
        do_yaml_update = False

        if not os.path.isdir(self.root_folder):
            print(f"'{self.root_folder}' is not a valid path")
            return

        # For each folder in the root_folder
        for folder_name in os.listdir(self.root_folder):
            if folder_name.startswith("Scan") and folder_name[4:].isdigit():
                scan_number = int(folder_name[4:])

                # Are they currently in the processed_list?
                if scan_number not in self.processed_list:

                    # Else, do they have a .tdms file?
                    scan_folder = os.path.join(self.root_folder, folder_name)
                    tdms_file = os.path.join(scan_folder, f"{folder_name}.tdms")
                    if os.path.exists(tdms_file):

                        # If so, add it to the processed list and figure out what analyses can be run
                        self.processed_list.append(scan_number)
                        do_yaml_update = True

                        valid_analyzers = check_for_analysis_match(scan_folder)

                        # Send those analysis commands to the appropriate analyzers.
                        # TODO implement command
                        if self.do_print:
                            print(f"{self.month}/{self.day}/{self.year}; Scan{scan_number}: {valid_analyzers}")

        # If there was a new scan, update the yaml file
        if do_yaml_update:
            self.write_processed_list()

        if self.do_print:
            print("-Check Complete")

    def read_processed_list(self):
        contents = self.read_yaml_file()

        year_data = contents.get(str(self.year)) if contents is not None else None
        month_data = year_data.get(str(self.month)) if year_data is not None else None
        day_data = month_data.get(str(self.day)) if month_data is not None else None

        if day_data is not None:
            for scan in day_data:
                self.processed_list.append(scan)

    def read_yaml_file(self):
        contents = None
        if os.path.exists(self.processed_list_filename):
            with open(self.processed_list_filename, 'r') as file:
                contents = yaml.safe_load(file) or []
        return contents

    def write_processed_list(self):
        data = self.read_yaml_file()
        new_contents = {str(self.year): {str(self.month): {str(self.day): self.processed_list}}}
        if data is None:
            data = new_contents
        else:
            data = recursive_update(data, new_contents)

        with open(self.processed_list_filename, 'w') as file:
            yaml.safe_dump(data, file)


def recursive_update(base, new):
    for key, value in new.items():
        if isinstance(value, dict):
            base[key] = recursive_update(base.get(key, {}), value)
        else:
            base[key] = value
    return base


if __name__ == "__main__":
    print("--Starting Lookout")
    lookout = NewScanLookout(exp='Undulator', year=2024, month=10, day=31)
    lookout.start_lookout()

    print("--Sleeping for 10 seconds")
    time.sleep(10)

    print("--Ending lookout")
    lookout.stop_lookout()
