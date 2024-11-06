"""
Watches the scan folders for a given day and reports any new scans
"""

import os
import time
import threading
from devices_to_analysis_mapping import check_for_analysis_match
from image_analysis.analyzers.online_analysis_modules.directory_functions import  compile_daily_path


class NewScanLookout:
    def __init__(self, exp, year, month, day, overwrite_previous=False, ignore_list=None, check_interval=2):
        self.experiment = exp
        self.year = year
        self.month = month
        self.day = day
        self.root_folder = compile_daily_path(self.day, self.month, self.year, experiment=self.experiment)

        self.overwrite_previous = overwrite_previous

        self.processed_list = []
        if ignore_list is not None:
            self.processed_list = ignore_list
        # TODO load previously-processed scans to the processed_list

        self.check_interval = check_interval

        self.stop_event = threading.Event()
        self.lookout_thread = threading.Thread(target=self.scan_folder_check_loop)

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
                        valid_analyzers = check_for_analysis_match(scan_folder)

                        # Send those analysis commands to the appropriate analyzers.
                        # TODO implement command
                        print(f"{self.day}/{self.month}/{self.year}; Scan{scan_number}: {valid_analyzers}")

        print("-Check Complete")
        return


if __name__ == "__main__":
    print("--Starting Lookout")
    lookout = NewScanLookout(exp='Undulator', year=2024, month=11, day=5)
    lookout.start_lookout()

    print("--Sleeping for 10 seconds")
    time.sleep(10)

    print("--Ending lookout")
    lookout.stop_lookout()
