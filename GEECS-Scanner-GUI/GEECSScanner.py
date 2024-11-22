"""
Script to contain the logic for the GEECSScanner GUI.  Can be launched by running this script in python.

-Chris
"""

import sys
from pathlib import Path
import threading
import traceback
import importlib
import yaml
import configparser

from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog, QCompleter, QMessageBox
from PyQt5.QtCore import Qt, QEvent, QTimer
from GEECSScanner_ui import Ui_MainWindow
from ScanElementEditor import ScanElementEditor
from MultiScanner import MultiScanner
from LogStream import EmittingStream, MultiStream

CURRENT_VERSION = 'v0.2'  # Try to keep this up-to-date, increase the version # with significant changes :)

MAXIMUM_SCAN_SIZE = 1e6
RELATIVE_PATH = Path("../GEECS-PythonAPI/geecs_python_api/controls/data_acquisition/configs/")
PRESET_LOCATIONS = Path("./scan_presets/")
MULTISCAN_CONFIGS = Path("./multiscan_presets/")
CONFIG_PATH = Path('~/.config/geecs_python_api/config.ini').expanduser()


class GEECSScannerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle(f"GEECS Scanner - {CURRENT_VERSION}")

        # Sets up the log at the bottom of the GUI to display errors.  TODO need to fix this, was crashing
        self.ui.logDisplay.setReadOnly(True)
        #sys.stdout = MultiStream(sys.stdout, EmittingStream(self.ui.logDisplay))
        #sys.stderr = MultiStream(sys.stderr, EmittingStream(self.ui.logDisplay))

        # Load experiment, repetition rate, and shot control device from the .config file
        self.experiment = ""
        self.repetition_rate = 0
        self.shot_control_device = ""
        self.MC_ip = None
        self.load_config_settings()

        # Initializes run control if possible, this serves as the interface to scan_manager and data_acquisition
        self.RunControl = None
        self.reinitialize_run_control()

        # Default values for the line edits
        self.noscan_num = 100
        self.scan_start = 0
        self.scan_stop = 0
        self.scan_step_size = 0
        self.scan_shot_per_step = 0

        # Line edits for the repetition rate and timing device
        self.ui.repititionRateDisplay.setText(str(self.repetition_rate))
        self.ui.repititionRateDisplay.editingFinished.connect(self.update_repetition_rate)
        self.ui.lineTimingDevice.setText(self.shot_control_device)
        self.ui.lineTimingDevice.editingFinished.connect(self.update_shot_control_device)

        # Button to launch dialog for changing the config file contents
        self.ui.buttonUpdateConfig.clicked.connect(self.reset_config_file)

        # Button to launch the multiscan window
        self.is_in_multiscan = False
        self.ui.buttonLaunchMultiScan.clicked.connect(self.open_multiscanner)

        # Line edit for the experiment display
        self.ui.experimentDisplay.setText(self.experiment)
        self.ui.experimentDisplay.installEventFilter(self)
        self.ui.experimentDisplay.editingFinished.connect(self.experiment_selected)

        # Populates the list of available save elements
        self.populate_found_list()

        # Functionality to move elements back and forth between the available and selected elements list
        self.ui.addDeviceButton.clicked.connect(self.add_files)
        self.ui.foundDevices.itemDoubleClicked.connect(self.add_files)
        self.ui.removeDeviceButton.clicked.connect(self.remove_files)
        self.ui.selectedDevices.itemDoubleClicked.connect(self.remove_files)

        self.updating_found_list = False
        self.updating_selected_list = False
        self.ui.foundDevices.itemSelectionChanged.connect(self.clear_selected_list_selection)
        self.ui.selectedDevices.itemSelectionChanged.connect(self.clear_found_list_selection)

        # Buttons to launch the element editor and refresh the list of available elements
        self.ui.newDeviceButton.clicked.connect(self.open_element_editor_new)
        self.ui.editDeviceButton.clicked.connect(self.open_element_editor_load)
        self.ui.buttonRefreshLists.clicked.connect(self.refresh_element_list)

        # Buttons to launch the side guis for the action library and scan variables
        self.ui.buttonActionLibrary.setEnabled(False)
        self.ui.buttonScanVariables.setEnabled(False)

        # Radio buttons that select if the next scan is to be a noscan or 1dscan
        self.ui.noscanRadioButton.setChecked(True)
        self.ui.noscanRadioButton.toggled.connect(self.update_scan_edit_state)
        self.ui.scanRadioButton.toggled.connect(self.update_scan_edit_state)
        self.update_scan_edit_state()

        # Upon changing a scan parameter, recalculate the total number of shots
        self.ui.lineStartValue.editingFinished.connect(self.calculate_num_shots)
        self.ui.lineStopValue.editingFinished.connect(self.calculate_num_shots)
        self.ui.lineStepSize.editingFinished.connect(self.calculate_num_shots)
        self.ui.lineShotStep.editingFinished.connect(self.calculate_num_shots)
        self.ui.lineNumShots.editingFinished.connect(self.update_noscan_num_shots)

        # Connect the line edit for the 1d scan variable to the list of available scan variables
        self.scan_variable = ""
        self.scan_variable_list = []
        self.scan_composite_list = []
        self.populate_scan_devices()
        self.ui.lineScanVariable.editingFinished.connect(self.check_scan_device)
        self.ui.lineScanVariable.installEventFilter(self)

        # Buttons to save the current scan as a preset, delete selected preset, and double-clicking loads the preset
        self.populate_preset_list()
        self.ui.listScanPresets.itemDoubleClicked.connect(self.apply_preset)
        self.ui.presetSaveButton.clicked.connect(self.save_current_preset)
        self.ui.presetDeleteButton.clicked.connect(self.delete_selected_preset)

        # Buttons to start and stop the current scan
        self.is_starting = False
        self.ui.startScanButton.clicked.connect(self.initialize_scan)
        self.ui.stopScanButton.clicked.connect(self.stop_scan)

        # Every 200 ms, check status of any ongoing scan and update the GUI accordingly
        self.ui.progressBar.setValue(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_indicator)
        self.ui.scanStatusIndicator.setText("")
        self.timer.start(200)
        self.update_indicator()

        self.element_editor = None
        self.multiscanner_window = None

    def eventFilter(self, source, event):
        # Creates a custom event for the text boxes so that the completion suggestions are shown when mouse is clicked
        if event.type() == QEvent.MouseButtonPress and source == self.ui.experimentDisplay:
            self.show_experiment_list()
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineScanVariable:
            self.show_scan_device_list()
            return True
        return super().eventFilter(source, event)

    def reinitialize_run_control(self):
        """
        Attempts to reinitialize the instance of RunControl, typically done when the experiment or shot control device
        is changed.  To do so, the experiment name in the config file is updated because geecs-python-api is heavily
        dependent on this for loading the correct database.
        """
        print("Reinitialization of Run Control")
        try:
            # The experiment name in the config is very embedded in geecs-python-api, so we have to rewrite it here...
            config = configparser.ConfigParser()
            config.read(CONFIG_PATH)
            if config['Experiment']['expt'] != self.experiment:
                print("Experiment name changed, rewriting config file")
                config.set('Experiment', 'expt', self.experiment)
                with open(CONFIG_PATH, 'w') as file:
                    config.write(file)

            RunControl = getattr(importlib.import_module('RunControl'), 'RunControl')
            self.RunControl = RunControl(experiment_name=self.experiment, shot_control=self.shot_control_device, master_control_ip=self.MC_ip)
        except AttributeError:
            print("ERROR: presumably because the entered experiment is not in the GEECS database")
            self.RunControl = None

    def load_config_settings(self):
        """
        Loads the experiment name, repetition rate, and shot control device from the config file on the current computer
        If an error occurs during this process, the user is prompted to fix the config file.
        """
        try:
            module = importlib.import_module('geecs_python_api.controls.interface')
            load_config = getattr(module, 'load_config')
            config = load_config()

            try:
                self.experiment = config['Experiment']['expt']
            except KeyError:
                self.prompt_config_reset("Could not find 'expt' in config")

            try:
                self.repetition_rate = float(config['Experiment']['rep_rate_hz'])
            except KeyError:
                self.prompt_config_reset("Could not find 'rep_rate_hz' in config")

            try:
                self.shot_control_device = config['Experiment']['shot_control']
            except KeyError:
                self.prompt_config_reset("Could not find 'shot_control' in config")
            try:
                self.MC_ip = config['Experiment']['MC_ip']
            except:
                pass

        except TypeError:
            self.prompt_config_reset("No configuration file found")
        except NameError:
            self.prompt_config_reset("No configuration file found")
        return

    def prompt_config_reset(self, notice_str: str):
        """
        Asks the user if they would like to repair the config file.  If not, close the GUI

        :param notice_str: Text displayed at the title of the question box
        """
        reply = QMessageBox.question(self, notice_str, 'Generate and/or repair .config file?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.reset_config_file()
            self.load_config_settings()
        else:
            print("Shutting Down")
            sys.exit()

    def reset_config_file(self):
        """
        If the config file does not exist, create one and give it default entries.  Then, ask the user if they would
        like to update the four crucial elements of this config file.  Afterwards, the new dictionary is written as the
        current config file and run control is reinitialized.
        """
        if not CONFIG_PATH.exists():
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            default_content = configparser.ConfigParser()
            default_content['Paths'] = {
                'geecs_data': 'C:\GEECS\\user data\\'
            }
            default_content['Experiment'] = {
                'expt': 'none',
                'rep_rate_hz': 'none',
                'shot_control': 'none'
            }
            with open(CONFIG_PATH, 'w') as config_file:
                default_content.write(config_file)
            print(f"Wrote new config file to {CONFIG_PATH}")

        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)

        config = self.prompt_config_update(config, 'Paths', 'geecs_data',
                                           'Enter GEECS data path: (ex: C:\\GEECS\\user data\\)')
        config = self.prompt_config_update(config, 'Experiment', 'expt',
                                           'Enter Experiment Name: (ex: Undulator)')
        config = self.prompt_config_update(config, 'Experiment', 'rep_rate_hz',
                                           'Enter repetition rate in Hz: (ex: 1)')
        config = self.prompt_config_update(config, 'Experiment', 'shot_control',
                                           'Enter shot control device: (ex: U_DG645_ShotControl)')

        print(f"Writing config file to {CONFIG_PATH}")
        with open(CONFIG_PATH, 'w') as file:
            config.write(file)
        self.load_config_settings()
        self.reinitialize_run_control()

    def prompt_config_update(self, config, section, option, information):
        """
        Prompts the user for a string to replace what is currently in the config file.

        :param config: the current configuration dictionary
        :param section: section in the config
        :param option: option in the specified section
        :param information: text to be displayed in the dialog box
        :return: the updated configuration dictionary
        """
        if config.has_section(section) and config.has_option(section, option):
            current = config[section][option]
        else:
            current = None
        text, ok = QInputDialog.getText(self, 'Config File Edit', information, text=current)
        if ok:
            config.set(section, option, text)
        return config

    def show_experiment_list(self):
        """
        Displays the found experiments in the ./experiments/ subfolder for selecting experiment
        """
        folders = [f.stem for f in (RELATIVE_PATH / 'experiments').iterdir() if f.is_dir()]
        completer = QCompleter(folders, self)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.ui.experimentDisplay.setCompleter(completer)
        self.ui.experimentDisplay.setFocus()
        completer.complete()

    def experiment_selected(self):
        """
        Upon selecting the experiment, reset the list of save devices and scan devices and reinitialize Run Control
        """
        selected_experiment = self.ui.experimentDisplay.text()
        if not (selected_experiment in self.experiment):
            self.clear_lists()
            new_folder_path = RELATIVE_PATH / 'experiments' / selected_experiment
            if new_folder_path.is_dir():
                self.experiment = selected_experiment
                self.ui.experimentDisplay.setText(self.experiment)
                self.reinitialize_run_control()
                self.populate_found_list()

            self.ui.lineScanVariable.setText("")
            self.scan_variable = ""
            self.populate_scan_devices()

    def clear_lists(self):
        """
        Clear all the lists in the GUI.  Used when a fresh start is needed, such as new experiment name or using preset
        """
        self.ui.selectedDevices.clear()
        self.ui.foundDevices.clear()
        self.ui.listScanPresets.clear()

    def update_repetition_rate(self):
        """Updates the repetition rate when it is changed in the text box, making sure it is a number
        """
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
        """Updates the shot control device when it is changed in the text box, then reinitializes Run Control
        """
        self.shot_control_device = self.ui.lineTimingDevice.text()
        self.reinitialize_run_control()

    def populate_found_list(self):
        """Gets all files in the save_devices folder under chosen experiment and adds it to the available elements list
        """
        try:
            experiment_preset_folder = RELATIVE_PATH / "experiments" / self.experiment / "save_devices"
            for f in experiment_preset_folder.iterdir():
                if f.is_file():
                    self.ui.foundDevices.addItem(f.stem)
        except OSError:
            self.clear_lists()

    def add_files(self):
        """Move selected files from the "Found" list to the "Selected" list"""
        selected_items = self.ui.foundDevices.selectedItems()
        for item in selected_items:
            self.ui.foundDevices.takeItem(self.ui.foundDevices.row(item))
            self.ui.selectedDevices.addItem(item)

    def remove_files(self):
        """Move selected files from the "Selected" list back to the "Found" list"""
        selected_items = self.ui.selectedDevices.selectedItems()
        for item in selected_items:
            self.ui.selectedDevices.takeItem(self.ui.selectedDevices.row(item))
            self.ui.foundDevices.addItem(item)

    def clear_found_list_selection(self):
        """When the selected list is changed, clear the found list selection (without recursively clearing this list)"""
        if not self.updating_selected_list:
            self.updating_found_list = True
            self.ui.foundDevices.clearSelection()
            self.updating_found_list = False

    def clear_selected_list_selection(self):
        """When the found list is changed, clear the selected list selection (without recursively clearing this list)"""
        if not self.updating_found_list:
            self.updating_selected_list = True
            self.ui.selectedDevices.clearSelection()
            self.updating_selected_list = False

    def open_element_editor_new(self):
        """Opens the ScanElementEditor GUI with a blank template.  If Run Control is initialized then the database
        dictionary is passed for device/variable hints.  Afterwards, refresh the lists of the available and selected
        elements."""
        if self.RunControl is not None:
            database_dict = self.RunControl.get_database_dict()
        else:
            database_dict = None
        config_folder = RELATIVE_PATH / "experiments" / self.experiment / "save_devices"
        self.element_editor = ScanElementEditor(database_dict=database_dict, config_folder=config_folder)
        self.element_editor.exec_()
        self.refresh_element_list()

    def open_element_editor_load(self):
        """Opens the ScanElementEditor GUI with using the selected element as a template.  If Run Control is initialized
        then the database dictionary is passed for device/variable hints.  Afterwards, refresh the lists of the
        available and selected elements."""
        selected_element = self.ui.foundDevices.selectedItems()
        if not selected_element:
            selected_element = self.ui.selectedDevices.selectedItems()
            if not selected_element:
                self.open_element_editor_new()
                return

        element_name = None
        for selection in selected_element:
            element_name = selection.text().strip() + ".yaml"

        if self.RunControl is not None:
            database_dict = self.RunControl.get_database_dict()
        else:
            database_dict = None
        config_folder = RELATIVE_PATH / "experiments" / self.experiment / "save_devices"
        self.element_editor = ScanElementEditor(database_dict=database_dict, config_folder=config_folder, load_config=element_name)
        self.element_editor.exec_()
        self.refresh_element_list()

    def open_multiscanner(self):
        """Opens the multiscanner window, and in the process sets a flag that disables starting scans on the main gui"""
        self.multiscanner_window = MultiScanner(self, MULTISCAN_CONFIGS / self.experiment)
        self.multiscanner_window.show()

        self.is_in_multiscan = True
        self.update_indicator()

    def exit_multiscan_mode(self):
        """Cleans up the multiscanner window and resets the enable-ability for buttons on the main window"""
        self.is_in_multiscan = False
        self.multiscanner_window = None

    def refresh_element_list(self):
        """Refreshes the list of available and selected elements, but does not clear them.  Instead, all previously
        selected elements remain in the selected list, new files are added to the available list, and deleted files are
        removed from either list."""
        print("Refreshing element list...")
        try:
            experiment_preset_folder = RELATIVE_PATH / "experiments" / self.experiment / "save_devices"

            selected_elements_list = {self.ui.selectedDevices.item(i).text()
                                      for i in range(self.ui.selectedDevices.count())}

            self.ui.foundDevices.clear()
            self.ui.selectedDevices.clear()

            for f in experiment_preset_folder.iterdir():
                if f.is_file():
                    if f.stem in selected_elements_list:
                        self.ui.selectedDevices.addItem(f.stem)
                    else:
                        self.ui.foundDevices.addItem(f.stem)
        except OSError:
            print("OSError occurred!")
            self.clear_lists()

        print("Refreshing scan variable list...")
        self.populate_scan_devices()

        print(" ...Done!")

    def update_scan_edit_state(self):
        """Depending on which radio button is selected, enable/disable text boxes for if this scan is a noscan or a
        variable scan.  Previous values are saved so the user can switch between the two scan modes easily."""
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
        """Generates a list of found scan devices from the scan_devices.yaml file"""
        self.scan_variable_list = []
        self.scan_composite_list = []
        try:
            experiment_folder = RELATIVE_PATH / "experiments" / self.experiment / "scan_devices"
            with open(experiment_folder / "scan_devices.yaml", 'r') as file:
                data = yaml.safe_load(file)
                devices = data['single_scan_devices']
                self.scan_variable_list = list(devices.keys())

            composite_variables_location = RELATIVE_PATH / "experiments" / self.experiment / "aux_configs"
            with open(composite_variables_location / "composite_variables.yaml", 'r') as file:
                data = yaml.safe_load(file)
                composite_vars = data['composite_variables']
                self.scan_composite_list = list(composite_vars.keys())

        except Exception as e:
            print(f"Error loading file: {e}")

        completer = QCompleter(self.scan_variable_list + self.scan_composite_list, self.ui.lineScanVariable)
        self.ui.lineScanVariable.setCompleter(completer)

    def read_device_tag_from_nickname(self, name):
        """
        Given a string, use it as a key in the scan_devices.yaml file and return the associated GEECS variable name.
        If the key is not found, assume it is a composite variable and just return the key itself.

        :param name: Selected scan variable to be converted to GEECS variable and/or composite variable
        """
        try:
            experiment_folder = RELATIVE_PATH / "experiments" / self.experiment / "scan_devices"
            with open(experiment_folder / "scan_devices.yaml", 'r') as file:
                data = yaml.safe_load(file)
                if name in data['single_scan_devices']:
                    return data['single_scan_devices'][name]
                else:
                    return name

        except Exception as e:
            print(f"Error loading scan_devices.yaml file: {e}")

    def show_scan_device_list(self):
        """Displays the list of scan devices when the user interacts with the scan variable selection text box"""
        completer = QCompleter(self.scan_variable_list + self.scan_composite_list, self)
        completer.setMaxVisibleItems(30)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.ui.lineScanVariable.setCompleter(completer)
        self.ui.lineScanVariable.setFocus()
        completer.complete()

    def check_scan_device(self):
        """Checks what is inputted into the scan variable selection box against the list of scan variables.  Otherwise,
        reset the line edit."""
        scan_device = self.ui.lineScanVariable.text()
        if scan_device in self.scan_variable_list:
            self.scan_variable = scan_device
            self.ui.labelStartValue.setText("Start Value: (abs)")
            self.ui.labelStopValue.setText("Stop Value: (abs)")
        elif scan_device in self.scan_composite_list:
            self.scan_variable = scan_device
            self.ui.labelStartValue.setText("Start Value: (rel)")
            self.ui.labelStopValue.setText("Stop Value: (rel)")
        else:
            self.scan_variable = ""
            self.ui.lineScanVariable.setText("")
            self.ui.labelStartValue.setText("Start Value:")
            self.ui.labelStopValue.setText("Stop Value:")

    def calculate_num_shots(self):
        """Given the parameters for a 1D scan, calculate the total number of shots and display it on the GUI"""
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
        """Given the parameters for a 1D scan, generate an array with the value of the scan variable for each shot."""
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
        """Updates the value of the number of shots in noscan mode, but only if it is a positive integer."""
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
        """Searches for existing presets in the designated folder and populates each preset to the list on the GUI"""
        self.ui.listScanPresets.clear()
        for preset in self.load_preset_list():
            self.ui.listScanPresets.addItem(preset)

    def load_preset_list(self):
        """
        :return: List containing the names of all presets in the designated folder
        """
        preset_list = []
        try:
            experiment_folder = PRESET_LOCATIONS / self.experiment
            for f in experiment_folder.iterdir():
                if f.is_file():
                    preset_list.append(f.stem)

        except OSError:
            print("Could not locate pre-existing scan presets.")
        return preset_list

    def save_current_preset(self):
        """Takes the current scan configuration and prompts the user if they would like to save it as a preset.  If so,
        the user give a filename to save under and the information is compiled into a yaml that "apply_preset" uses"""
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

            preset_file = PRESET_LOCATIONS / self.experiment / (text + ".yaml")
            preset_file.parent.mkdir(parents=True, exist_ok=True)

            with open(preset_file, 'w') as file:
                yaml.dump(settings, file, default_flow_style=False)

            self.populate_preset_list()

    def apply_preset(self):
        """Loads the yaml file selected in the preset list, clears all current information on the GUI, then
        systematically populate everything so that the GUI is equivalent to when it was saved as a preset"""
        selected_element = self.ui.listScanPresets.selectedItems()
        for preset in selected_element:
            preset_name = f"{preset.text()}"

        self.apply_preset_from_name(preset_name)

    def apply_preset_from_name(self, preset_name, load_save_elements=True, load_scan_params=True):
        """
        :param preset_name: Name of the preset
        :param load_save_elements: Defaults to True, flag to load the save elements from a preset file
        :param load_scan_params: Defaults to True, flag to load the scan parameters from a preset file
        """
        preset_filename = PRESET_LOCATIONS / self.experiment / (preset_name + ".yaml")
        with open(preset_filename, 'r') as file:
            settings = yaml.safe_load(file)

        self.ui.textEditScanInfo.setText(str(settings['Info']))

        if load_save_elements:
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

        if load_scan_params:
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
        """Deletes the preset that is currently selected in the list.  Afterwards, refreshes the preset list"""
        selected_element = self.ui.listScanPresets.selectedItems()
        for preset in selected_element:
            preset_filename = PRESET_LOCATIONS / self.experiment / (preset.text() + ".yaml")
            try:
                preset_filename.unlink()
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
        """Checks the full GUI for any blatant errors.  To be used before submitting a scan to be run"""
        # TODO Need to add more logic in here.  IE, at least 1 shot, at least 1 save device, etc etc
        if not self.repetition_rate > 0:
            print("Error: Need nonzero repetition rate")
            return True
        return False

    def initialize_scan(self):
        """Compiles the information from the GUI into a dictionary that can be used by scan_manager.  This dictionary is
        then sent to RunControl to be submitted for a scan."""
        if not self.check_for_errors():
            # From the information provided in the GUI, create a scan configuration file and submit to GEECS for
            #  data logging.
            self.is_starting = True
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
                fullpath = RELATIVE_PATH / "experiments" / self.experiment / "save_devices" / (filename + ".yaml")
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
                'description': self.ui.textEditScanInfo.toPlainText().replace('\n', ' ')
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

            scan_parameters = {  # TODO What does this even do?  Perhaps remove from dictionary?
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
            self.is_starting = False

    def update_indicator(self):
        """Checks the current status of any ongoing run and updates the GUI accordingly.  Several configurations are
        represented here, but the overall goal is to not allow RunControl to be edited or Start Scan to be clicked when
        a scan is currently running.  If a scan is running, the Stop Scan button is enabled and the progress bar shows
        the current progress.  When multiscan window is open, you can't launch a single scan or change run control"""

        # TODO Could be useful to clean up the logic here.  It has become quite a mess with all the combinations
        if self.RunControl is None:
            self.ui.scanStatusIndicator.setStyleSheet("background-color: grey;")
            self.ui.startScanButton.setEnabled(False)
            self.ui.stopScanButton.setEnabled(False)
        elif self.RunControl.is_active():
            self.ui.scanStatusIndicator.setStyleSheet("background-color: red;")
            self.ui.startScanButton.setEnabled(False)
            self.ui.stopScanButton.setEnabled(not self.RunControl.is_stopping())
            self.ui.progressBar.setValue(self.RunControl.get_progress())
        else:
            self.ui.scanStatusIndicator.setStyleSheet("background-color: green;")
            self.ui.stopScanButton.setEnabled(False)
            self.ui.startScanButton.setEnabled(not self.RunControl.is_busy())
            self.RunControl.clear_stop_state()

        if self.RunControl is not None:
            if self.is_in_multiscan:
                self.ui.scanStatusIndicator.setStyleSheet("background-color: blue;")
                self.ui.startScanButton.setEnabled(False)
            self.ui.experimentDisplay.setEnabled(self.ui.startScanButton.isEnabled())
            self.ui.repititionRateDisplay.setEnabled(self.ui.startScanButton.isEnabled())
            self.ui.lineTimingDevice.setEnabled(self.ui.startScanButton.isEnabled())
            self.ui.buttonUpdateConfig.setEnabled(self.ui.startScanButton.isEnabled())
            self.ui.buttonLaunchMultiScan.setEnabled(self.ui.startScanButton.isEnabled())

    def is_ready_for_scan(self):
        """
        :return: True if Run control is initialized, and not currently starting up or scanning
        """
        if self.RunControl is None:
            return False
        else:
            return not (self.RunControl.is_active() or self.is_starting)

    def stop_scan(self):
        """Submits a request to RunControl to stop the current scan.  In the meantime, disable the Stop Scan button so
        that the user can't click this button multiple times."""
        self.ui.stopScanButton.setEnabled(False)
        QApplication.processEvents()
        self.RunControl.stop_scan()

    def closeEvent(self, event):
        """Upon the GUI closing, also attempts to close child windows and stop any currently-running scans"""
        if self.multiscanner_window:
            self.multiscanner_window.stop_multiscan()
            self.multiscanner_window.close()
        if self.element_editor:
            self.element_editor.close()

        self.stop_scan()

        for thread in threading.enumerate():
            print(thread.name)


def exception_hook(exctype, value, tb):
    """
    Global wrapper to print out tracebacks of python errors during the execution of a PyQT window.

    :param exctype: Exception Type
    :param value: Value of the exception
    :param tb: Traceback
    """
    print("An error occurred:")
    traceback.print_exception(exctype, value, tb)
    sys.__excepthook__(exctype, value, tb)
    sys.exit(1)


if __name__ == '__main__':
    """Launches the GEECS Scanner GUI"""
    sys.excepthook = exception_hook
    app = QApplication(sys.argv)

    perfect_application = GEECSScannerWindow()
    perfect_application.show()

    sys.exit(app.exec_())
