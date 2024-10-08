"""
Script to contain the logic for the GEECSScanner GUI

-Chris
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit, QCompleter
from PyQt5.QtCore import Qt, QEvent
from GEECSScanner_ui import Ui_MainWindow
try:
    from geecs_python_api.controls.interface import load_config
except TypeError:
    print("No configuration file found!  Recommended to add one.")


class GEECSScannerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.experiment = "<None Selected>"
        self.repetition_rate = ""
        self.load_config_settings()

        self.ui.repititionRateDisplay.setText(self.repetition_rate)
        self.ui.repititionRateDisplay.textChanged.connect(self.update_repetition_rate)

        self.ui.experimentDisplay.setText(self.experiment)
        self.ui.experimentDisplay.setReadOnly(True)
        self.ui.experimentDisplay.installEventFilter(self)

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

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress and source == self.ui.experimentDisplay:
            self.show_experiment_list()
            return True
        return super().eventFilter(source, event)

    def load_config_settings(self):
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
        folders = [f for f in os.listdir("./experiments/") if os.path.isdir(os.path.join("./experiments", f))]
        completer = QCompleter(folders, self)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.ui.experimentDisplay.setCompleter(completer)
        self.ui.experimentDisplay.setReadOnly(False)
        self.ui.experimentDisplay.clear()
        self.ui.experimentDisplay.setFocus()
        completer.complete()
        self.ui.experimentDisplay.returnPressed.connect(self.experiment_selected)

    def experiment_selected(self):
        self.clear_lists()
        selected_experiment = self.ui.experimentDisplay.text()
        new_folder_path = os.path.join("./experiments/", selected_experiment)
        if os.path.isdir(new_folder_path):
            self.experiment = selected_experiment
            self.ui.experimentDisplay.setText(self.experiment)
            self.populate_found_list()

        self.ui.experimentDisplay.setReadOnly(True)

    def clear_lists(self):
        self.ui.selectedDevices.clear()
        self.ui.foundDevices.clear()

    def update_repetition_rate(self, text):
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
        if self.ui.noscanRadioButton.isChecked():
            self.ui.lineScanVariable.setEnabled(False)
            self.ui.lineStartValue.setEnabled(False)
            self.ui.lineStopValue.setEnabled(False)
            self.ui.lineStepSize.setEnabled(False)
            self.ui.lineShotStep.setEnabled(False)
            self.ui.lineNumShots.setEnabled(True)

        else:
            self.ui.lineScanVariable.setEnabled(True)
            self.ui.lineStartValue.setEnabled(True)
            self.ui.lineStopValue.setEnabled(True)
            self.ui.lineStepSize.setEnabled(True)
            self.ui.lineShotStep.setEnabled(True)
            self.ui.lineNumShots.setEnabled(False)

if __name__ == ('__main__'):
    app = QApplication(sys.argv)

    file_selector = GEECSScannerWindow()
    file_selector.show()

    sys.exit(app.exec_())
