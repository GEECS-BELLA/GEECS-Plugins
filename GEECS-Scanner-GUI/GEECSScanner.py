"""
Script to contain the logic for the GEECSScanner GUI

-Chris
"""

import sys
import os
import yaml
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit, QCompleter
from PyQt5.QtCore import Qt, QEvent
from GEECSScanner_ui import Ui_MainWindow
try:
    from geecs_python_api.controls.interface import load_config
except TypeError:
    print("No configuration file found!  This is required for GEECSScanner!")
    #sys.exit()

MAXIMUM_SCAN_SIZE = 1e6


class GEECSScannerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.experiment = "<None Selected>"
        self.repetition_rate = ""
        self.load_config_settings()

        self.noscan_num = 100
        self.scan_start = 0
        self.scan_stop = 0
        self.scan_step_size = 0
        self.scan_shot_per_step = 0

        self.ui.repititionRateDisplay.setText(self.repetition_rate)
        self.ui.repititionRateDisplay.textChanged.connect(self.update_repetition_rate)

        self.ui.experimentDisplay.setText(self.experiment)
        self.ui.experimentDisplay.installEventFilter(self)
        self.ui.experimentDisplay.editingFinished.connect(self.experiment_selected)

        self.populate_found_list()

        # Connect buttons to their respective functions
        self.ui.addDeviceButton.clicked.connect(self.add_files)
        self.ui.foundDevices.itemDoubleClicked.connect(self.add_files)
        self.ui.removeDeviceButton.clicked.connect(self.remove_files)
        self.ui.selectedDevices.itemDoubleClicked.connect(self.remove_files)

        self.ui.noscanRadioButton.setChecked(True)
        self.ui.noscanRadioButton.toggled.connect(self.update_scan_edit_state)
        self.ui.scanRadioButton.toggled.connect(self.update_scan_edit_state)
        self.update_scan_edit_state()

        self.ui.lineStartValue.editingFinished.connect(self.calculate_num_shots)
        self.ui.lineStopValue.editingFinished.connect(self.calculate_num_shots)
        self.ui.lineStepSize.editingFinished.connect(self.calculate_num_shots)
        self.ui.lineShotStep.editingFinished.connect(self.calculate_num_shots)
        self.ui.lineNumShots.editingFinished.connect(self.update_noscan_num_shots)

        self.scan_variable = ""
        self.scan_device_list = []
        self.populate_scan_devices()
        self.ui.lineScanVariable.editingFinished.connect(self.check_scan_device)
        self.ui.lineScanVariable.installEventFilter(self)

    def eventFilter(self, source, event):
        # Creates a custom event for the text boxes so that the completion suggestions are shown when mouse is clicked
        if event.type() == QEvent.MouseButtonPress and source == self.ui.experimentDisplay:
            self.show_experiment_list()
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineScanVariable:
            self.show_scan_device_list()
            return True
        return super().eventFilter(source, event)

    def load_config_settings(self):
        # Loads in the experiment name and repetition rate from the configuration file located in ~/.config/
        try:
            config = load_config()
            try:
                default_experiment = config['Experiment']['expt']
            except KeyError:
                print("Could not find 'expt' in config")
                default_experiment = "<None Selected>"
            if os.path.isdir("./experiments/" + default_experiment):
                self.experiment = default_experiment

            try:
                self.repetition_rate = config['Experiment']['rep_rate_hz']
            except KeyError:
                print("Could not find 'rep_rate_hz' in config")
        except NameError:
            print("Could not read from config file")
        return

    def show_experiment_list(self):
        # Displays the found experiments in the ./experiments/ subfolder for selecting experiment
        folders = [f for f in os.listdir("./experiments/") if os.path.isdir(os.path.join("./experiments", f))]
        completer = QCompleter(folders, self)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.ui.experimentDisplay.setCompleter(completer)
        self.ui.experimentDisplay.setFocus()
        completer.complete()

    def experiment_selected(self):
        # Upon selecting the experiment, reset the list of save devices and scan devices.
        selected_experiment = self.ui.experimentDisplay.text()
        if not (selected_experiment in self.experiment):
            self.clear_lists()
            new_folder_path = os.path.join("./experiments/", selected_experiment)
            if os.path.isdir(new_folder_path):
                self.experiment = selected_experiment
                self.ui.experimentDisplay.setText(self.experiment)
                self.populate_found_list()

            self.ui.lineScanVariable.setText("")
            self.scan_variable = ""
            self.populate_scan_devices()

    def clear_lists(self):
        # Clears the list of found and selected save devices
        self.ui.selectedDevices.clear()
        self.ui.foundDevices.clear()

    def update_repetition_rate(self, text):
        # Updates the repetition rate when it is changed in the text box
        # TODO need to check that this is a number
        self.repetition_rate = text

    def populate_found_list(self):
        # List all files in the save_devices folder under chosen experiment:
        try:
            experiment_folder = "./experiments/" + self.experiment + "/save_devices/"
            for file_name in os.listdir(experiment_folder):
                full_path = os.path.join(experiment_folder, file_name)
                if os.path.isfile(full_path):
                    root, ext = os.path.splitext(file_name)
                    self.ui.foundDevices.addItem(root)
        except OSError:
            self.clear_lists()

    def add_files(self):
        # Move selected files from the "Found" list to the "Selected" list
        selected_items = self.ui.foundDevices.selectedItems()
        for item in selected_items:
            self.ui.foundDevices.takeItem(self.ui.foundDevices.row(item))
            self.ui.selectedDevices.addItem(item)

    def remove_files(self):
        # Move selected files from the "Selected" list back to the "Found" list
        selected_items = self.ui.selectedDevices.selectedItems()
        for item in selected_items:
            self.ui.selectedDevices.takeItem(self.ui.selectedDevices.row(item))
            self.ui.foundDevices.addItem(item)

    def update_scan_edit_state(self):
        # Depending on which radio button is selected, enable/disable text boxes for if this scan is a noscan or a
        #  variable scan.  Previous values are saved so the user can switch between the two scan modes easily.
        if self.ui.noscanRadioButton.isChecked():
            self.ui.lineScanVariable.setEnabled(False)
            self.ui.lineScanVariable.setText("")
            self.ui.lineStartValue.setEnabled(False)
            self.ui.lineStartValue.setText("")
            self.ui.lineStopValue.setEnabled(False)
            self.ui.lineStopValue.setText("")
            self.ui.lineStepSize.setEnabled(False)
            self.ui.lineStepSize.setText("")
            self.ui.lineShotStep.setEnabled(False)
            self.ui.lineShotStep.setText("")
            self.ui.lineNumShots.setEnabled(True)
            self.ui.lineNumShots.setText(str(self.noscan_num))

        else:
            self.ui.lineScanVariable.setEnabled(True)
            self.ui.lineScanVariable.setText(self.scan_variable)
            self.ui.lineStartValue.setEnabled(True)
            self.ui.lineStartValue.setText(str(self.scan_start))
            self.ui.lineStopValue.setEnabled(True)
            self.ui.lineStopValue.setText(str(self.scan_stop))
            self.ui.lineStepSize.setEnabled(True)
            self.ui.lineStepSize.setText(str(self.scan_step_size))
            self.ui.lineShotStep.setEnabled(True)
            self.ui.lineShotStep.setText(str(self.scan_shot_per_step))
            self.ui.lineNumShots.setEnabled(False)
            self.calculate_num_shots()

    def populate_scan_devices(self):
        # Generates a list of found scan devices from the scan_devices.yaml file
        self.scan_device_list = []
        try:
            experiment_folder = "./experiments/" + self.experiment + "/scan_devices/"
            with open(experiment_folder+"scan_devices.yaml", 'r') as file:
                data = yaml.safe_load(file)
                devices = data['single_scan_devices']
                self.scan_device_list = devices

        except Exception as e:
            print(f"Error loading scan_devices.yaml file: {e}")

        completer = QCompleter(self.scan_device_list, self.ui.lineScanVariable)
        self.ui.lineScanVariable.setCompleter(completer)

    def show_scan_device_list(self):
        # Displays the list of scan devices when the user interacts with the scan variable selection text box
        completer = QCompleter(self.scan_device_list, self)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.ui.lineScanVariable.setCompleter(completer)
        self.ui.lineScanVariable.setFocus()
        completer.complete()

    def check_scan_device(self):
        # Checks what is inputted into the scan variable selection box against the list of scan variables
        scan_device = self.ui.lineScanVariable.text()
        if scan_device in self.scan_device_list:
            self.scan_variable = scan_device
        else:
            self.scan_variable = ""
            self.ui.lineScanVariable.setText("")

    def calculate_num_shots(self):
        # Given the parameters for a 1D scan, calculate the total number of shots
        try:
            start = float(self.ui.lineStartValue.text())
            self.scan_start = start
            stop = float(self.ui.lineStopValue.text())
            self.scan_stop = stop
            step_size = float(self.ui.lineStepSize.text())
            self.scan_step_size = step_size
            shot_per_step = int(self.ui.lineShotStep.text())
            self.scan_shot_per_step = shot_per_step

            if step_size == 0:
                self.ui.lineNumShots.setText("N/A")
            else:
                shot_array = self.build_shot_array()
                self.ui.lineNumShots.setText(str(len(shot_array)))

        except ValueError:
            self.ui.lineNumShots.setText("N/A")

    def build_shot_array(self):
        # Given the parameters for a 1D scan, generate an array with the value of the scan variable for each shot
        if (self.scan_stop-self.scan_start)/self.scan_step_size*self.scan_shot_per_step > MAXIMUM_SCAN_SIZE:
            return []
        else:
            array = []
            current = self.scan_start
            while ((self.scan_step_size > 0 and current <= self.scan_stop)
                   or (self.scan_step_size < 0 and current >= self.scan_stop)):
                array.extend([current]*self.scan_shot_per_step)
                current = round(current + self.scan_step_size, 10)
            return array

    def update_noscan_num_shots(self):
        # Updates the value of the number of shots in noscan mode, but only if it is a positive integer.
        if self.ui.noscanRadioButton.isChecked():
            try:
                num_shots = int(self.ui.lineNumShots.text())
                if num_shots > 0:
                    self.noscan_num = num_shots
                else:
                    self.ui.lineNumShots.setText("N/A")
            except ValueError:
                self.ui.lineNumShots.setText("N/A")


if __name__ == '__main__':
    # When this python file is run, open and display the GEECSScanner main window
    app = QApplication(sys.argv)

    perfect_application = GEECSScannerWindow()
    perfect_application.show()

    sys.exit(app.exec_())
