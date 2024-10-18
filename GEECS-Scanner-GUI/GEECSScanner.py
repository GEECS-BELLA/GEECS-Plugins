"""
Script to contain the logic for the GEECSScanner GUI

-Chris
"""

import sys
import os
import traceback

import yaml
from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog, QCompleter
from PyQt5.QtCore import Qt, QEvent, QTimer
from GEECSScanner_ui import Ui_MainWindow
from ScanElementEditor import ScanElementEditor
from LogStream import EmittingStream, MultiStream
from RunControl import RunControl

try:
    from geecs_python_api.controls.interface import load_config
except TypeError:
    print("No configuration file found!  This is required for GEECSScanner!")
    # sys.exit()

MAXIMUM_SCAN_SIZE = 1e6
RELATIVE_PATH = "../GEECS-PythonAPI/geecs_python_api/controls/data_acquisition/configs/"
PRESET_LOCATIONS = "./scan_presets/"

class GEECSScannerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.logDisplay.setReadOnly(True)
        #sys.stdout = MultiStream(sys.stdout, EmittingStream(self.ui.logDisplay))
        #sys.stderr = MultiStream(sys.stderr, EmittingStream(self.ui.logDisplay))

        self.experiment = ""
        self.repetition_rate = 0
        self.shot_control_device = ""
        self.load_config_settings()

        self.RunControl = RunControl(experiment_name=self.experiment, shot_control=self.shot_control_device)

        self.noscan_num = 100
        self.scan_start = 0
        self.scan_stop = 0
        self.scan_step_size = 0
        self.scan_shot_per_step = 0

        self.ui.repititionRateDisplay.setText(str(self.repetition_rate))
        self.ui.repititionRateDisplay.editingFinished.connect(self.update_repetition_rate)
        self.ui.lineTimingDevice.setText(self.shot_control_device)
        self.ui.lineTimingDevice.editingFinished.connect(self.update_shot_control_device)

        self.ui.experimentDisplay.setText(self.experiment)
        self.ui.experimentDisplay.installEventFilter(self)
        self.ui.experimentDisplay.editingFinished.connect(self.experiment_selected)

        self.populate_found_list()

        # Connect buttons to their respective functions
        self.ui.addDeviceButton.clicked.connect(self.add_files)
        self.ui.foundDevices.itemDoubleClicked.connect(self.add_files)
        self.ui.removeDeviceButton.clicked.connect(self.remove_files)
        self.ui.selectedDevices.itemDoubleClicked.connect(self.remove_files)

        self.ui.newDeviceButton.clicked.connect(self.open_element_editor)

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

        self.populate_preset_list()
        self.ui.listScanPresets.itemDoubleClicked.connect(self.apply_preset)
        self.ui.presetSaveButton.clicked.connect(self.save_current_preset)
        self.ui.presetDeleteButton.clicked.connect(self.delete_selected_preset)

        self.ui.startScanButton.clicked.connect(self.initialize_scan)
        self.ui.stopScanButton.clicked.connect(self.stop_scan)

        self.ui.progressBar.setValue(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_indicator)
        self.ui.scanStatusIndicator.setText("")
        self.timer.start(200)
        self.update_indicator()

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
                default_experiment = ""
            if os.path.isdir(RELATIVE_PATH + "experiments/" + default_experiment):
                self.experiment = default_experiment

            try:
                self.repetition_rate = float(config['Experiment']['rep_rate_hz'])
            except KeyError:
                print("Could not find 'rep_rate_hz' in config")

            try:
                self.shot_control_device = config['Experiment']['shot_control']
            except KeyError:
                print("Could not find 'rep_rate_hz' in config")

        except NameError:
            print("Could not read from config file")
        return

    def show_experiment_list(self):
        # Displays the found experiments in the ./experiments/ subfolder for selecting experiment
        folders = [f for f in os.listdir(RELATIVE_PATH + "experiments/")
                   if os.path.isdir(os.path.join(RELATIVE_PATH + "experiments", f))]
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
            new_folder_path = os.path.join(RELATIVE_PATH + "experiments/", selected_experiment)
            if os.path.isdir(new_folder_path):
                self.experiment = selected_experiment
                self.ui.experimentDisplay.setText(self.experiment)
                self.RunControl = RunControl(experiment_name=self.experiment, shot_control=self.shot_control_device)
                self.populate_found_list()

            self.ui.lineScanVariable.setText("")
            self.scan_variable = ""
            self.populate_scan_devices()

    def clear_lists(self):
        # Clears the list of found and selected save devices
        self.ui.selectedDevices.clear()
        self.ui.foundDevices.clear()
        self.ui.listScanPresets.clear()

    def update_repetition_rate(self):
        # Updates the repetition rate when it is changed in the text box
        try:
            rep_rate = float(self.ui.repititionRateDisplay.text())
            if rep_rate > 0:
                self.repetition_rate = rep_rate
            else:
                self.ui.repititionRateDisplay.setText("N/A")
                self.repetition_rate = 0
        except ValueError:
            self.ui.repititionRateDisplay.setText("N/A")
            self.repetition_rate = 0

    def update_shot_control_device(self):
        # Updates the shot control device when it is changed in the text box
        self.shot_control_device = self.ui.lineTimingDevice.text()
        self.RunControl = RunControl(experiment_name=self.experiment, shot_control=self.shot_control_device)

    def populate_found_list(self):
        # List all files in the save_devices folder under chosen experiment:
        try:
            experiment_preset_folder = RELATIVE_PATH + "experiments/" + self.experiment + "/save_devices/"
            for file_name in os.listdir(experiment_preset_folder):
                full_path = os.path.join(experiment_preset_folder, file_name)
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

    def open_element_editor(self):
        self.element_editor = ScanElementEditor()
        self.element_editor.exec_()

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
            experiment_folder = RELATIVE_PATH + "experiments/" + self.experiment + "/scan_devices/"
            with open(experiment_folder + "scan_devices.yaml", 'r') as file:
                data = yaml.safe_load(file)
                devices = data['single_scan_devices']
                self.scan_device_list = list(devices.keys())

            composite_variables_location = RELATIVE_PATH + "experiments/" + self.experiment + "/aux_configs/"
            with open(composite_variables_location + "composite_variables.yaml", 'r') as file:
                data = yaml.safe_load(file)
                composite_vars = data['composite_variables']
                self.scan_device_list.extend(list(composite_vars.keys()))

        except Exception as e:
            print(f"Error loading file: {e}")

        completer = QCompleter(self.scan_device_list, self.ui.lineScanVariable)
        self.ui.lineScanVariable.setCompleter(completer)

    def read_device_tag_from_nickname(self, name):
        print(name)
        try:
            experiment_folder = RELATIVE_PATH + "experiments/" + self.experiment + "/scan_devices/"
            with open(experiment_folder + "scan_devices.yaml", 'r') as file:
                data = yaml.safe_load(file)
                if name in data['single_scan_devices']:
                    return data['single_scan_devices'][name]
                else:
                    return name

        except Exception as e:
            print(f"Error loading scan_devices.yaml file: {e}")

    def show_scan_device_list(self):
        # Displays the list of scan devices when the user interacts with the scan variable selection text box
        completer = QCompleter(self.scan_device_list, self)
        completer.setMaxVisibleItems(30)
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
        if (self.scan_stop - self.scan_start) / self.scan_step_size * self.scan_shot_per_step > MAXIMUM_SCAN_SIZE:
            return []
        else:
            array = []
            current = self.scan_start
            while ((self.scan_step_size > 0 and current <= self.scan_stop)
                   or (self.scan_step_size < 0 and current >= self.scan_stop)):
                array.extend([current] * self.scan_shot_per_step)
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

    def populate_preset_list(self):
        self.ui.listScanPresets.clear()
        try:
            experiment_folder = PRESET_LOCATIONS + self.experiment + "/"
            for file_name in os.listdir(experiment_folder):
                full_path = os.path.join(experiment_folder, file_name)
                if os.path.isfile(full_path):
                    root, ext = os.path.splitext(file_name)
                    self.ui.listScanPresets.addItem(root)
        except OSError:
            print("Could not locate pre-existing scan presets.")
            self.ui.listScanPresets.clear()

    def save_current_preset(self):
        text, ok = QInputDialog.getText(self, 'Save Configuration', 'Enter filename:')
        if ok and text:
            save_device_list = []
            for i in range(self.ui.selectedDevices.count()):
                device = self.ui.selectedDevices.item(i).text()
                save_device_list.append(device)

            settings = {'Devices': save_device_list, 'Info': self.ui.textEditScanInfo.toPlainText()}
            if self.ui.noscanRadioButton.isChecked():
                settings['Scan Mode'] = 'No Scan'
                settings['Num Shots'] = self.noscan_num
            elif self.ui.scanRadioButton.isChecked():
                settings['Scan Mode'] = '1D Scan'
                settings['Variable'] = self.scan_variable
                settings['Start'] = self.scan_start
                settings['Stop'] = self.scan_stop
                settings['Step Size'] = self.scan_step_size
                settings['Shot per Step'] = self.scan_shot_per_step

            folder = f"{PRESET_LOCATIONS}{self.experiment}/"
            os.makedirs(folder, exist_ok=True)

            with open(f"{folder}/{text}.yaml", 'w') as file:
                yaml.dump(settings, file, default_flow_style=False)

            self.populate_preset_list()

    def apply_preset(self):
        selected_element = self.ui.listScanPresets.selectedItems()
        settings = {}
        for preset in selected_element:
            preset_filename = f"{PRESET_LOCATIONS}{self.experiment}/{preset.text()}.yaml"
            with open(preset_filename, 'r') as file:
                settings = yaml.safe_load(file)

        self.clear_lists()
        self.populate_found_list()
        self.populate_preset_list()

        devices_to_select = []
        for index in range(self.ui.foundDevices.count()):
            item = self.ui.foundDevices.item(index)
            if item.text() in settings['Devices']:
                devices_to_select.append(item)
        for device in devices_to_select:
            self.ui.foundDevices.takeItem(self.ui.foundDevices.row(device))
            self.ui.selectedDevices.addItem(device)

        self.ui.textEditScanInfo.setText(str(settings['Info']))
        if settings['Scan Mode'] in "No Scan":
            self.ui.noscanRadioButton.setChecked(True)
            self.update_scan_edit_state()
            self.ui.lineNumShots.setText(str(settings['Num Shots']))
            self.update_noscan_num_shots()
        elif settings['Scan Mode'] in "1D Scan":
            self.ui.scanRadioButton.setChecked(True)
            self.update_scan_edit_state()
            self.ui.lineScanVariable.setText(str(settings['Variable']))
            self.check_scan_device()
            self.ui.lineStartValue.setText(str(settings['Start']))
            self.ui.lineStopValue.setText(str(settings['Stop']))
            self.ui.lineStepSize.setText(str(settings['Step Size']))
            self.ui.lineShotStep.setText(str(settings['Shot per Step']))
            self.calculate_num_shots()

    def delete_selected_preset(self):
        selected_element = self.ui.listScanPresets.selectedItems()
        for preset in selected_element:
            preset_filename = f"{PRESET_LOCATIONS}{self.experiment}/{preset.text()}.yaml"
            try:
                os.remove(preset_filename)
                print(f"{preset_filename} has been deleted :(")
            except FileNotFoundError:
                print(f"{preset_filename} not found.")
            except PermissionError:
                print(f"Permission denied: {preset_filename}")
            except Exception as e:
                print(f"Error occurred: {e}")

        self.ui.listScanPresets.clear()
        self.populate_preset_list()

    def check_for_errors(self):
        if not self.repetition_rate > 0:
            print("Error: Need nonzero repetition rate")
            return True
        return False

    def initialize_scan(self):
        if not self.check_for_errors():
            # From the information provided in the GUI, create a scan configuration file and submit to GEECS for
            #  data logging.
            self.ui.startScanButton.setEnabled(False)
            self.ui.experimentDisplay.setEnabled(False)
            self.ui.repititionRateDisplay.setEnabled(False)
            self.ui.lineTimingDevice.setEnabled(False)
            self.ui.startScanButton.setText("Starting...")
            QApplication.processEvents()

            save_device_list = {}
            list_of_steps = []
            for i in range(self.ui.selectedDevices.count()):
                filename = self.ui.selectedDevices.item(i).text()
                fullpath = RELATIVE_PATH + f"experiments/{self.experiment}/save_devices/{filename}.yaml"
                with open(fullpath, 'r') as file:
                    try:
                        data = yaml.safe_load(file)
                        save_device_list.update(data['Devices'])

                        if 'setup_action' in data:
                            setup_action = data['setup_action']
                            list_of_steps.extend(setup_action['steps'])
                    except yaml.YAMLError as exc:
                        print(f"Error reading YAML file: {exc}")

            scan_information = {
                'experiment': self.experiment,
                'description': self.ui.textEditScanInfo.toPlainText()
            }

            if self.ui.scanRadioButton.isChecked():
                scan_variable_tag = self.read_device_tag_from_nickname(self.scan_variable)
                scan_config = {
                    'device_var': scan_variable_tag,
                    'start': self.scan_start,
                    'end': self.scan_stop,
                    'step': self.scan_step_size,
                    'wait_time': (self.scan_shot_per_step + 0.5)/self.repetition_rate
                }
                scan_mode = scan_variable_tag
                scan_array_initial = self.scan_start
                scan_array_final = self.scan_stop
            elif self.ui.noscanRadioButton.isChecked():
                scan_config = {
                    'device_var': 'noscan',
                    'wait_time': (self.noscan_num + 0.5)/self.repetition_rate
                }
                scan_mode = "noscan"
                scan_array_initial = 0
                scan_array_final = self.noscan_num
            else:
                scan_config = None
                scan_mode = ""
                scan_array_initial = 0
                scan_array_final = 0

            scan_parameters = {  # TODO What does this even do???
                'scan_mode': scan_mode,
                'scan_range': [scan_array_initial, scan_array_final]
            }

            run_config = {
                'Devices': save_device_list,
                'scan_info': scan_information,
                'scan_parameters': scan_parameters,
            }
            if list_of_steps:
                steps = {'steps': list_of_steps}
                run_config['setup_action'] = steps

            self.RunControl.submit_run(config_dictionary=run_config, scan_config=scan_config)
            self.ui.startScanButton.setText("Start Scan")

    def update_indicator(self):
        if self.RunControl.is_active():
            self.ui.scanStatusIndicator.setStyleSheet("background-color: red;")
            self.ui.startScanButton.setEnabled(False)
            self.ui.stopScanButton.setEnabled(not self.RunControl.is_stopping())
            self.ui.progressBar.setValue(int(self.RunControl.get_progress()))
        else:
            self.ui.scanStatusIndicator.setStyleSheet("background-color: green;")
            self.ui.startScanButton.setEnabled(not self.RunControl.is_busy())
            self.ui.stopScanButton.setEnabled(False)
            self.RunControl.clear_stop_state()

        self.ui.experimentDisplay.setEnabled(self.ui.startScanButton.isEnabled())
        self.ui.repititionRateDisplay.setEnabled(self.ui.startScanButton.isEnabled())
        self.ui.lineTimingDevice.setEnabled(self.ui.startScanButton.isEnabled())

    def stop_scan(self):
        self.ui.stopScanButton.setEnabled(False)
        QApplication.processEvents()
        self.RunControl.stop_scan()


def exception_hook(exctype, value, tb):
    # This is a global wrapper to print out tracebacks of python errors.  PyCharm wasn't doing this with PyQT windows
    print("An error occurred:")
    traceback.print_exception(exctype, value, tb)
    sys.__excepthook__(exctype, value, tb)
    sys.exit(1)


if __name__ == '__main__':
    # When this python file is run, open and display the GEECSScanner main window
    sys.excepthook = exception_hook
    app = QApplication(sys.argv)

    perfect_application = GEECSScannerWindow()
    perfect_application.show()

    sys.exit(app.exec_())
