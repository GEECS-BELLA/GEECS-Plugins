"""
Script to contain the logic for the GEECSScanner GUI.  Can be launched by running this script in python.

-Chris
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from run_control import RunControl
    from PyQt5.QtWidgets import QWidget

import sys
from pathlib import Path
import threading
import importlib
import yaml
import configparser
import logging
from importlib.metadata import version

from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog, QCompleter, QMessageBox
from PyQt5.QtCore import Qt, QEvent, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices, QIcon
from .gui.GEECSScanner_ui import Ui_MainWindow
from .lib import MenuBarOption, MenuBarOptionBool, MenuBarOptionStr
from . import SaveElementEditor, MultiScanner, ShotControlEditor, ScanVariableEditor, ActionLibrary
from ..utils import ApplicationPaths as AppPaths, module_open_folder as of
from ..utils.exceptions import ConflictingScanElements

from geecs_scanner.data_acquisition import DatabaseDictLookup

CURRENT_VERSION = "v" + version("geecs-scanner-gui")  # Pulled from `pyproject.toml` for GEECS-Scanner-GUI sub-repo

MAXIMUM_SCAN_SIZE = 1e6  # A simple check to not start a scan if it exceeds this number of shots.

# Lists of options to appear in the menu bar
BOOLEAN_OPTIONS = ["On-Shot TDMS"]
STRING_OPTIONS = ["Master Control IP", "Save Hiatus Period (s)"]


class GEECSScannerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle(f"GEECS Scanner - {CURRENT_VERSION}")
        self.setWindowIcon(QIcon(":/application_icon.ico"))

        # Sets up the log at the bottom of the GUI to display errors.  TODO need to fix this, was crashing
        self.ui.logDisplay.setReadOnly(True)
        # sys.stdout = MultiStream(sys.stdout, EmittingStream(self.ui.logDisplay))
        # sys.stderr = MultiStream(sys.stderr, EmittingStream(self.ui.logDisplay))

        # Load experiment, repetition rate, and shot control device from the .config file
        self.experiment = ""
        self.repetition_rate = 0
        self.timing_configuration_name = ""
        self.load_config_settings()

        # Initializes run control if possible, this serves as the interface to scan_manager and data_acquisition
        self.RunControl: Optional[RunControl] = None
        self.database_lookup = DatabaseDictLookup()
        self.app_paths: Optional[AppPaths] = None
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

        self.ui.lineTimingDevice.setReadOnly(True)
        self.ui.lineTimingDevice.setText(self.timing_configuration_name)
        self.ui.lineTimingDevice.installEventFilter(self)
        self.ui.lineTimingDevice.textChanged.connect(self.update_shot_control_device)

        # Button to launch dialog for changing the config file contents
        self.ui.buttonUpdateConfig.clicked.connect(self.reset_config_file)

        # Button to launch the multiscanner and action library widgets
        self.is_in_multiscan = False
        self.ui.buttonLaunchMultiScan.clicked.connect(self.open_multiscanner)
        self.is_in_action_library = False
        self.ui.buttonActionLibrary.clicked.connect(self.open_action_library)

        # Line edit for the experiment display
        self.ui.experimentDisplay.setText(self.experiment)
        self.ui.experimentDisplay.installEventFilter(self)
        self.ui.experimentDisplay.textChanged.connect(self.experiment_selected)

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
        self.load_element_name = None
        self.ui.newDeviceButton.clicked.connect(self.open_element_editor_new)
        self.ui.editDeviceButton.clicked.connect(self.open_element_editor_load)
        self.ui.buttonRefreshLists.clicked.connect(self.refresh_element_list)

        self.ui.buttonDeleteElement.setIcon(QIcon(":/trashcan_icon.ico"))
        self.ui.buttonDeleteElement.setIconSize(self.ui.buttonDeleteElement.size()*0.8)
        self.ui.buttonDeleteElement.clicked.connect(self.delete_selected_element)

        # Buttons to launch the side guis for the timing device setup and scan variables
        self.ui.buttonScanVariables.clicked.connect(self.open_scan_variable_editor)
        self.ui.buttonOpenTimingSetup.clicked.connect(self.open_timing_setup)

        # Radio buttons that select if the next scan is to be a noscan or 1dscan
        self.ui.noscanRadioButton.setChecked(True)
        self.ui.noscanRadioButton.toggled.connect(self.update_scan_edit_state)
        self.ui.scanRadioButton.toggled.connect(self.update_scan_edit_state)
        self.ui.backgroundRadioButton.toggled.connect(self.update_scan_edit_state)
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
        self.scan_composite_data = {}
        self.populate_scan_devices()
        self.ui.lineScanVariable.textChanged.connect(self.check_scan_device)
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

        # Variables to store the current scan number, flag to search for the latest, and a 10s timer to turn off flag
        self.current_scan_number = 0
        self.known_scan_number = False
        self.scan_number_timer = QTimer(self)
        self.scan_number_timer.timeout.connect(self.forget_scan_number)
        self.scan_number_timer.start(10000)

        # Every 200 ms, check status of any ongoing scan and update the GUI accordingly
        self.ui.progressBar.setValue(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_indicator)
        self.ui.scanStatusIndicator.setText("")
        self.timer.start(200)
        self.update_indicator()

        # Menu Bar: Open
        self.ui.action_open_Daily_Scan_Folder.triggered.connect(self.open_daily_scan_folder)
        self.ui.action_open_Experiment_Config_Folder.triggered.connect(self.open_experiment_config_folder)
        self.ui.action_open_User_Config_File.triggered.connect(self.open_user_config_file)

        self.ui.action_open_Github_Page.triggered.connect(self.open_github_page)
        self.ui.actionExperiment_Scanlog.triggered.connect(self.open_experiment_scanlog)

        # Menu Bar: Options
        self.all_options: list[MenuBarOption] = []
        for opt in BOOLEAN_OPTIONS:
            menu_opt = MenuBarOptionBool(self, opt)
            self.ui.menuOptions.addAction(menu_opt)
            self.all_options.append(menu_opt)
        self.ui.menuOptions.addSeparator()
        for opt in STRING_OPTIONS:
            menu_opt = MenuBarOptionStr(self, opt)
            self.ui.menuOptions.addAction(menu_opt)
            self.all_options.append(menu_opt)

        # Menu Bar: Preferences
        self.ui.actionDarkMode.toggled.connect(self.toggle_light_dark)

        # Initial state of side-gui's
        self.element_editor = None
        self.multiscanner_window = None
        self.action_library_window = None
        self.timing_editor = None
        self.variable_editor = None

        # Set current GUI mode
        self.toggle_light_dark()

    def eventFilter(self, source, event):
        # Creates a custom event for the text boxes so that the completion suggestions are shown when mouse is clicked
        if event.type() == QEvent.MouseButtonPress and source == self.ui.experimentDisplay and self.ui.experimentDisplay.isEnabled():
            self.show_experiment_list()
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineScanVariable and self.ui.lineScanVariable.isEnabled():
            self.show_scan_device_list()
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineTimingDevice and self.ui.lineTimingDevice.isEnabled():
            self.show_timing_configuration_list()
            return True
        return super().eventFilter(source, event)

    def reinitialize_run_control(self):
        """
        Attempts to reinitialize the instance of RunControl, typically done when the experiment or shot control device
        is changed.  To do so, the experiment name in the config file is updated because geecs-python-api is heavily
        dependent on this for loading the correct database.
        """

        if self.experiment is None or self.experiment == "":
            logging.warning("No experiment selected")
            self.RunControl = None
            self.app_paths = None
            return

        logging.info("Reinitialization of Run Control")
        self.app_paths = AppPaths(experiment=self.experiment)

        # Before initializing, rewrite config file if experiment name or timing configuration name has changed
        config = configparser.ConfigParser()
        config.read(AppPaths.config_file())

        do_write = False
        if config['Experiment']['expt'] != self.experiment:
            logging.info("Experiment name changed, rewriting config file")
            config.set('Experiment', 'expt', self.experiment)
            do_write = True

        if ((not config.has_option('Experiment', 'timing_configuration')) or
                config['Experiment']['timing_configuration'] != self.timing_configuration_name):
            logging.info("Timing configuration changed, rewriting config file")
            config.set('Experiment', 'timing_configuration', self.timing_configuration_name)
            do_write = True

        if do_write:
            with open(AppPaths.config_file(), 'w') as file:
                config.write(file)

        shot_control_path = self.app_paths.shot_control() / (self.timing_configuration_name + ".yaml")
        if not shot_control_path.exists():
            shot_control_path = None

        try:
            module_path = Path(__file__).parent / 'run_control.py'
            sys.path.insert(0, str(module_path.parent))
            run_control_class = getattr(importlib.import_module('run_control'), 'RunControl')
            self.RunControl = run_control_class(experiment_name=self.experiment,
                                                shot_control_configuration=shot_control_path)

        except AttributeError:
            logging.error("AttributeError at RunControl: presumably because the entered experiment is not in the GEECS database")
            self.RunControl = None
        except KeyError:
            logging.error("KeyError at RunControl: presumably because no GEECS Database is connected to located devices")
            self.RunControl = None
        except ValueError:
            logging.error("ValueError at RunControl: presumably because no experiment name or shot control given")
            self.RunControl = None
        except ConnectionRefusedError as e:
            logging.error(f"ConnectionRefusedError at RunControl: {e}")
            self.RunControl = None

        sys.path.pop(0)

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
            except ValueError:
                self.prompt_config_reset("`rep_rate_hz` needs to be an int or float")

            try:
                self.timing_configuration_name = config['Experiment']['timing_configuration']
            except KeyError:
                logging.warning("No prior 'timing_configuration' set in config file")
                pass
            try:
                ip_address = config['Options']['master control ip']
                logging.info(f"Will attempt to use IP '{ip_address}' for ECS dumps.")
            except KeyError:
                logging.warning("Not including master control ip, no ECS dumps.")
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
            logging.info("Shutting Down")
            sys.exit()

    def reset_config_file(self):
        """
        If the config file does not exist, create one and give it default entries.  Then, ask the user if they would
        like to update the four crucial elements of this config file.  Afterwards, the new dictionary is written as the
        current config file and run control is reinitialized.
        """
        if AppPaths.create_config_if_missing():
            logging.info(f"Wrote new config file to {AppPaths.config_file()}")

        config = configparser.ConfigParser()
        config.read(AppPaths.config_file())

        config = self.prompt_config_update(config, 'Paths', 'geecs_data',
                                           'Enter GEECS data path: (ex: C:\\GEECS\\user data\\)')
        config = self.prompt_config_update(config, 'Experiment', 'expt',
                                           'Enter Experiment Name: (ex: Undulator)')
        config = self.prompt_config_update(config, 'Experiment', 'rep_rate_hz',
                                           'Enter repetition rate in Hz: (ex: 1)')

        logging.info(f"Writing config file to {AppPaths.config_file()}")
        with open(AppPaths.config_file(), 'w') as file:
            config.write(file)
        self.load_config_settings()
        self.ui.experimentDisplay.setText(self.experiment)
        self.experiment_selected(force_update=True)
        # self.reinitialize_run_control()

    def prompt_config_update(self, config: configparser.ConfigParser,
                             section: str, option: str, information: str) -> configparser.ConfigParser:
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
        folders = [f.stem for f in AppPaths.BASE_PATH.iterdir() if f.is_dir()]
        completer = QCompleter(folders, self)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.ui.experimentDisplay.setCompleter(completer)
        self.ui.experimentDisplay.setFocus()
        completer.complete()

    def experiment_selected(self, force_update=False):
        """
        Upon selecting the experiment, reset the list of save devices and scan devices and reinitialize Run Control
        """
        selected_experiment = self.ui.experimentDisplay.text()
        if not (selected_experiment in self.experiment) or force_update:
            self.clear_lists()

            self.experiment = selected_experiment
            self.ui.experimentDisplay.setText(self.experiment)
            self.timing_configuration_name = ""
            self.ui.lineTimingDevice.setText(self.timing_configuration_name)
            self.reinitialize_run_control()
            of.reload_scan_data_paths()

            self.populate_found_list()
            self.ui.lineScanVariable.setText("")
            self.scan_variable = ""
            self.populate_scan_devices()
            self.populate_preset_list()

    def find_database_dict(self) -> dict:
        """
        First will retrieve database through RunControl if initialized, otherwise will use the experiment name

        :return: Database dictionary representing all devices and associated variables for the given experiment
        """
        if self.RunControl is not None:
            return self.RunControl.get_database_dict()
        else:
            try:
                self.database_lookup.reload(experiment_name=self.experiment)
                return self.database_lookup.get_database()
            except Exception as e:  # TODO could pursue a less broad exception catching here...
                logging.warning("Error occurred when retrieving database dictionary: {e}")
                return {}

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

    def show_timing_configuration_list(self):
        """ Shows list of available timing device configurations """
        if self.app_paths is None:
            logging.error("No defined paths for timing configurations")
            return

        config_folder_path = self.app_paths.shot_control()
        if config_folder_path.exists():
            files = [f.stem for f in config_folder_path.iterdir() if f.suffix == ".yaml"]
            completer = QCompleter(files, self)
            completer.setCompletionMode(QCompleter.PopupCompletion)
            completer.setCaseSensitivity(Qt.CaseInsensitive)

            self.ui.lineTimingDevice.setCompleter(completer)
            self.ui.lineTimingDevice.setFocus()
            completer.complete()

    def update_shot_control_device(self):
        """ Updates the shot control device when it is changed in the text box, then reinitializes Run Control """
        self.timing_configuration_name = self.ui.lineTimingDevice.text()
        self.reinitialize_run_control()

    def populate_found_list(self):
        """Gets all files in the save_devices folder under chosen experiment and adds it to the available elements list
        """
        try:
            experiment_preset_folder = self.app_paths.save_devices()
            for f in experiment_preset_folder.iterdir():
                if f.is_file():
                    self.ui.foundDevices.addItem(f.stem)
        except OSError:
            self.clear_lists()
        except AttributeError:
            logging.error("No defined path for save devices")
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

    def delete_selected_element(self):
        """ Prompts the user if the selected save element should be deleted from the experiment """
        selected_element = self.ui.foundDevices.selectedItems()
        if not selected_element:
            selected_element = self.ui.selectedDevices.selectedItems()
            if not selected_element:
                return

        name = max((item.text() for item in selected_element), default="")
        reply = QMessageBox.question(self, "Delete Save Element",
                                     f"Delete element '{name}' and remove from experiment?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            element_filename = self.app_paths.save_devices() / (name + ".yaml")
            try:
                element_filename.unlink()
                logging.info(f"{element_filename} has been deleted")
            except FileNotFoundError:
                logging.error(f"{element_filename} not found.")
            except PermissionError:
                logging.error(f"Permission denied: {element_filename}")
            except Exception as e:
                logging.error(f"Error occurred: {e}")

            self.refresh_element_list()

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
        database_dict = self.find_database_dict()

        config_folder = None if self.app_paths is None else self.app_paths.save_devices()
        self.element_editor = SaveElementEditor(main_window=self, database_dict=database_dict,
                                                config_folder=config_folder, load_config=self.load_element_name)
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
        self.load_element_name = element_name
        self.open_element_editor_new()
        self.load_element_name = None

    def open_multiscanner(self):
        """Opens the multiscanner window, and in the process sets a flag that disables starting scans on the main gui"""
        preset_folder = None if self.app_paths is None else self.app_paths.multiscan_presets()
        self.multiscanner_window = MultiScanner(self, preset_folder)
        self.multiscanner_window.show()

        self.is_in_multiscan = True
        self.update_indicator()

    def exit_multiscan_mode(self):
        """Cleans up the multiscanner window and resets the enable-ability for buttons on the main window"""
        self.is_in_multiscan = False
        self.multiscanner_window = None

    def open_action_library(self):
        """Opens the multiscanner window, and in the process sets a flag that disables starting scans on the main gui"""
        actions_folder = None if self.app_paths is None else self.app_paths.action_library()
        database_dict = self.find_database_dict()
        self.action_library_window = ActionLibrary(self, database_dict, actions_folder)
        self.action_library_window.show()

        self.is_in_action_library = True
        self.update_indicator()

    def exit_action_library(self):
        """Cleans up the multiscanner window and resets the enable-ability for buttons on the main window"""
        self.is_in_action_library = False
        self.action_library_window = None

    def open_scan_variable_editor(self):
        database_dict = self.find_database_dict()
        config_folder_path = None if self.app_paths is None else self.app_paths.scan_devices()

        self.variable_editor = ScanVariableEditor(main_window=self, database_dict=database_dict,
                                                  config_folder=config_folder_path)
        self.variable_editor.exec_()
        self.populate_scan_devices()

    def open_timing_setup(self):
        """ Opens the timing setup window, using the current contents of the line edit to populate the dialog gui """
        database_dict = self.find_database_dict()
        config_folder_path = None if self.app_paths is None else self.app_paths.shot_control()

        self.timing_editor = ShotControlEditor(main_window=self, config_folder_path=config_folder_path,
                                               current_config=self.ui.lineTimingDevice.text(),
                                               database_dict=database_dict)
        self.timing_editor.selected_configuration.connect(self.handle_returned_timing_configuration)
        self.timing_editor.exec_()

    def handle_returned_timing_configuration(self, specified_configuration):
        """Handler for the return string of the timing setup dialog, sets the new configuration and reload RunControl"""
        force_update = bool(self.timing_configuration_name == specified_configuration)
        self.ui.lineTimingDevice.setText(specified_configuration)
        if force_update:
            self.update_shot_control_device()

    def refresh_element_list(self):
        """Refreshes the list of available and selected elements, but does not clear them.  Instead, all previously
        selected elements remain in the selected list, new files are added to the available list, and deleted files are
        removed from either list."""
        logging.info("Refreshing element list...")
        try:
            selected_elements_list = {self.ui.selectedDevices.item(i).text()
                                      for i in range(self.ui.selectedDevices.count())}

            self.ui.foundDevices.clear()
            self.ui.selectedDevices.clear()

            for f in self.app_paths.save_devices().iterdir():
                if f.is_file():
                    if f.stem in selected_elements_list:
                        self.ui.selectedDevices.addItem(f.stem)
                    else:
                        self.ui.foundDevices.addItem(f.stem)
        except OSError:
            logging.error("OSError occurred!")
            self.clear_lists()
        except AttributeError:
            logging.error("No defined path for save devices")
            self.clear_lists()

        logging.info("Refreshing scan variable list...")
        self.populate_scan_devices()

        logging.info(" ...Done!")

    def update_scan_edit_state(self):
        """Depending on which radio button is selected, enable/disable text boxes for if this scan is a noscan or a
        variable scan.  Previous values are saved so the user can switch between the two scan modes easily."""
        if self.ui.noscanRadioButton.isChecked() or self.ui.backgroundRadioButton.isChecked():
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
        self.scan_composite_data = {}

        try:
            if self.app_paths is None:
                raise FileNotFoundError("No defined paths for scan devices")

            with open(self.app_paths.scan_devices() / "scan_devices.yaml", 'r') as file:
                data = yaml.safe_load(file)
                devices = data['single_scan_devices']
                self.scan_variable_list = list(devices.keys())

            with open(self.app_paths.scan_devices() / "composite_variables.yaml", 'r') as file:
                self.scan_composite_data = yaml.safe_load(file)
                composite_vars = self.scan_composite_data['composite_variables']
                self.scan_composite_list = list(composite_vars.keys())

        except FileNotFoundError as e:
            logging.error(f"Error loading file: {e}")

        completer = QCompleter(self.scan_variable_list + self.scan_composite_list, self.ui.lineScanVariable)
        self.ui.lineScanVariable.setCompleter(completer)

    def read_device_tag_from_nickname(self, name: str):
        """
        Given a string, use it as a key in the scan_devices.yaml file and return the associated GEECS variable name.
        If the key is not found, assume it is a composite variable and just return the key itself.

        :param name: Selected scan variable to be converted to GEECS variable and/or composite variable
        """
        try:
            with open(self.app_paths.scan_devices() / "scan_devices.yaml", 'r') as file:
                data = yaml.safe_load(file)
                if name in data['single_scan_devices']:
                    return data['single_scan_devices'][name]
                else:
                    return name

        except Exception as e:
            logging.error(f"Error loading scan_devices.yaml file: {e}")

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
            mode = self.scan_composite_data['composite_variables'][scan_device]['mode'][:3]
            self.ui.labelStartValue.setText(f"Start Value: ({mode})")
            self.ui.labelStopValue.setText(f"Stop Value: ({mode})")
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

    def build_shot_array(self) -> list[int]:
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
        if self.ui.noscanRadioButton.isChecked() or self.ui.backgroundRadioButton.isChecked():
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

    def load_preset_list(self) -> list[str]:
        """
        :return: List containing the names of all presets in the designated folder
        """
        preset_list = []
        try:
            for f in self.app_paths.presets().iterdir():
                if f.is_file():
                    preset_list.append(f.stem)

        except OSError:
            logging.error("Could not locate pre-existing scan presets.")
        except AttributeError:
            logging.error("No defined path for scan presets")
        return preset_list

    def save_current_preset(self):
        """Takes the current scan configuration and prompts the user if they would like to save it as a preset.  If so,
        the user give a filename to save under and the information is compiled into a yaml that "apply_preset" uses"""
        if self.app_paths is None:
            logging.error("No defined paths for scan presets")
            return

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
            elif self.ui.backgroundRadioButton.isChecked():
                settings['Scan Mode'] = 'Background'
                settings['Num Shots'] = self.noscan_num
            elif self.ui.scanRadioButton.isChecked():
                settings['Scan Mode'] = '1D Scan'
                settings['Variable'] = self.scan_variable
                settings['Start'] = self.scan_start
                settings['Stop'] = self.scan_stop
                settings['Step Size'] = self.scan_step_size
                settings['Shot per Step'] = self.scan_shot_per_step

            preset_file = self.app_paths.presets() / (text + ".yaml")
            preset_file.parent.mkdir(parents=True, exist_ok=True)

            with open(preset_file, 'w') as file:
                yaml.dump(settings, file, default_flow_style=False)

            self.populate_preset_list()

    def apply_preset(self):
        """Loads the yaml file selected in the preset list, clears all current information on the GUI, then
        systematically populate everything so that the GUI is equivalent to when it was saved as a preset"""
        selected_element = self.ui.listScanPresets.selectedItems()
        preset_name = None
        for preset in selected_element:
            preset_name = f"{preset.text()}"

        if preset_name is not None:
            self.apply_preset_from_name(preset_name)

    def apply_preset_from_name(self, preset_name: str, load_save_elements: bool = True, load_scan_params: bool = True):
        """
        :param preset_name: Name of the preset
        :param load_save_elements: Defaults to True, flag to load the save elements from a preset file
        :param load_scan_params: Defaults to True, flag to load the scan parameters from a preset file
        """
        preset_filename = self.app_paths.presets() / (preset_name + ".yaml")
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
            if settings['Scan Mode'] in ["No Scan", "Background"]:
                if settings['Scan Mode'] in "No Scan":
                    self.ui.noscanRadioButton.setChecked(True)
                else:
                    self.ui.backgroundRadioButton.setChecked(True)
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
            preset_filename = self.app_paths.presets() / (preset.text() + ".yaml")
            try:
                preset_filename.unlink()
                logging.info(f"{preset_filename} has been deleted :(")
            except FileNotFoundError:
                logging.error(f"{preset_filename} not found.")
            except PermissionError:
                logging.error(f"Permission denied: {preset_filename}")
            except Exception as e:
                logging.error(f"Error occurred: {e}")

        self.ui.listScanPresets.clear()
        self.populate_preset_list()

    def check_for_errors(self) -> bool:
        """Checks the full GUI for any blatant errors.  To be used before submitting a scan to be run"""
        # TODO Need to add more logic in here.  IE, at least 1 shot, at least 1 save device, etc etc
        if not self.repetition_rate > 0:
            logging.error("Need nonzero repetition rate")
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
            list_of_setup_steps = []
            list_of_closeout_steps = []
            for i in range(self.ui.selectedDevices.count()):
                filename = self.ui.selectedDevices.item(i).text()
                with open(self.app_paths.save_devices() / (filename + ".yaml"), 'r') as file:
                    try:
                        data = yaml.safe_load(file)
                        self.combine_elements(save_device_list, data['Devices'])

                        if 'setup_action' in data:
                            setup_action = data['setup_action']
                            list_of_setup_steps.extend(setup_action['steps'])

                        if 'closeout_action' in data:
                            setup_action = data['closeout_action']
                            list_of_closeout_steps.extend(setup_action['steps'])

                    except yaml.YAMLError as exc:
                        logging.error(f"Error reading YAML file: {exc}")
                        QMessageBox.warning(self, 'YAML Error', f"Could not read '{filename}.yaml'", QMessageBox.Ok)
                        self.is_starting = False
                        return

                    except ConflictingScanElements as e:
                        logging.error(e.message)
                        QMessageBox.warning(self, 'Conflicting Save Elements', e.message, QMessageBox.Ok)
                        self.is_starting = False
                        return

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
            elif self.ui.noscanRadioButton.isChecked() or self.ui.backgroundRadioButton.isChecked():
                scan_config = {
                    'device_var': 'noscan',
                    'wait_time': (self.noscan_num + 0.5)/self.repetition_rate
                }
            else:
                scan_config = None
            scan_config['background'] = str(self.ui.backgroundRadioButton.isChecked())

            option_dict = {
                "rep_rate_hz": self.repetition_rate,
                "randomized_beeps": self.ui.actionRandomizedBeeps.isChecked()
            }
            for opt in self.all_options:
                option_dict[opt.get_name()] = opt.get_value()

            run_config = {
                'Devices': save_device_list,
                'scan_info': scan_information,
                'options': option_dict,
            }

            if list_of_setup_steps:
                setup_action_steps = {'steps': list_of_setup_steps}
                run_config['setup_action'] = setup_action_steps
            if list_of_closeout_steps:
                closeout_action_steps = {'steps': list_of_closeout_steps}
                run_config['closeout_action'] = closeout_action_steps

            success = self.RunControl.submit_run(config_dictionary=run_config, scan_config=scan_config)
            if not success:
                QMessageBox.critical(self, "Device Error",
                                     f"Device reinitialization failed.  Check log for problem device(s)")
            self.is_starting = False
            self.current_scan_number += 1

    @staticmethod
    def combine_elements(dict_element1, dict_element2):
        """
        Combines two dictionaries representing save elements.  Duplicate variables are not added, and conflicting
        boolean flags throw a ConflictingScanElements error

        :param dict_element1: base dictionary element
        :param dict_element2: dictionary to check and append
        """
        for device, details2 in dict_element2.items():
            if device in dict_element1:
                details1 = dict_element1[device]
                # Add any new variables for overlapping devices
                if 'variable_list' in details2:
                    if 'variable_list' in details1:
                        existing_vars = set(details1['variable_list'])
                        new_vars = set(details2['variable_list'])
                        combined_vars = existing_vars.union(new_vars)
                        details1['variable_list'] = list(combined_vars)
                    else:
                        details1['variable_list'] = details2['variable_list']

                # If boolean flags are conflicting, throw an error
                for flag in ['add_all_variables', 'save_nonscalar_data', 'synchronous']:
                    if details2.get(flag, False) != details1.get(flag, False):
                        raise ConflictingScanElements(f"Conflict involving {device}: '{flag}', please resolve.")
            else:
                dict_element1[device] = details2

    def forget_scan_number(self):
        """ Every 10 seconds (defined in __init__) the current scan number is forgotten and must be recalculated """
        self.known_scan_number = False

    def _set_scan_number_info(self, label: str = "", turn_off: bool = False):
        """
        Updates the visible information tracking the current scan number

        :param label: Message to display next to scan number
        :param turn_off: optional flag to set text for both elements to empty
        """
        if turn_off or not self.experiment:
            self.ui.labelScanNumber.setText("")
            self.ui.lineLastScan.setText("")
            return

        if not self.known_scan_number:
            self.current_scan_number = of.get_latest_scan_number(experiment=self.experiment)
            self.known_scan_number = True
        if self.current_scan_number == 0:
            self.ui.labelScanNumber.setText("No Scans Today:")
            self.ui.lineLastScan.setText("")
        else:
            self.ui.labelScanNumber.setText(label)
            self.ui.lineLastScan.setText(str(self.current_scan_number))

    def update_indicator(self):
        """Checks the current status of any ongoing run and updates the GUI accordingly.  Several configurations are
        represented here, but the overall goal is to not allow RunControl to be edited or Start Scan to be clicked when
        a scan is currently running.  If a scan is running, the Stop Scan button is enabled and the progress bar shows
        the current progress.  When multiscan window is open, you can't launch a single scan or change run control"""

        # TODO Could be useful to clean up the logic here.  It has become quite a mess with all the combinations
        if self.RunControl is None:
            self._set_scan_number_info(turn_off=True)
            self.ui.scanStatusIndicator.setStyleSheet("background-color: grey;")
            self.ui.startScanButton.setEnabled(False)
            self.ui.stopScanButton.setEnabled(False)
        elif self.RunControl.is_active():
            self._set_scan_number_info(label="Current Scan:")
            self.ui.scanStatusIndicator.setStyleSheet("background-color: red;")
            self.ui.startScanButton.setEnabled(False)
            self.ui.stopScanButton.setEnabled(not self.RunControl.is_stopping())
            self.ui.progressBar.setValue(self.RunControl.get_progress())
        else:
            self._set_scan_number_info(label="Previous Scan:")
            self.ui.scanStatusIndicator.setStyleSheet("background-color: green;")
            self.ui.stopScanButton.setEnabled(False)
            self.ui.startScanButton.setEnabled(not self.RunControl.is_busy())
            self.RunControl.clear_stop_state()

        if self.RunControl is not None:
            if self.is_starting:
                self.ui.startScanButton.setText("Starting...")
            else:
                self.ui.startScanButton.setText("Start Scan")

            enable_buttons = True
            if self.is_in_multiscan:
                self.ui.scanStatusIndicator.setStyleSheet("background-color: blue;")
                self.ui.startScanButton.setEnabled(False)
                enable_buttons = False
            self.ui.buttonLaunchMultiScan.setEnabled(self.ui.startScanButton.isEnabled())

            if self.is_in_action_library:
                enable_buttons = False
            self.ui.buttonActionLibrary.setEnabled(not self.is_in_action_library)

            self.ui.experimentDisplay.setEnabled(enable_buttons)
            self.ui.repititionRateDisplay.setEnabled(enable_buttons)
            self.ui.lineTimingDevice.setEnabled(enable_buttons)
            self.ui.buttonUpdateConfig.setEnabled(enable_buttons)
            self.ui.buttonOpenTimingSetup.setEnabled(enable_buttons)

    def toggle_light_dark(self):
        mode = 'dark' if self.ui.actionDarkMode.isChecked() else 'light'
        base_path = Path(__file__).parent / "gui"

        # List of Widgets that also need updating (only widgets that can be active at the same time as the main window)
        gui_list: list[Optional[QWidget]] = [self, self.multiscanner_window, self.action_library_window]
        for widget in gui_list:
            if widget is not None:
                if mode == 'light':
                    widget.setStyleSheet(self._read_stylesheet(base_path / "light_mode.qss"))
                elif mode == 'dark':
                    widget.setStyleSheet(self._read_stylesheet(base_path / "dark_mode.qss"))

    @staticmethod
    def _read_stylesheet(filename):
        with open(filename, "r") as file:
            return file.read()

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

    def open_daily_scan_folder(self):
        """ Opens file explorer at the location of server's data save folder for the day """
        if self.app_paths is not None:
            try:
                of.open_daily_data_folder(experiment=self.experiment)
            except AttributeError:
                logging.error(f"Server location not defined for '{self.experiment}', see 'geecs_paths_config.py'")

    def open_experiment_config_folder(self):
        """
        Opens file explorer at the location of the experiment config files, or the list of experiments if none selected
        """
        if self.app_paths is not None:
            of.open_folder(self.app_paths.experiment())
        else:
            of.open_folder(AppPaths.base_path())

    @staticmethod
    def open_user_config_file():
        """ Opens file explorer at the location of the user's config .ini file """
        of.open_folder(AppPaths.config_file().parent)

    def open_github_page(self):
        """ In the default browser, opens the GitHub url for the GEECS-Scanner-GUI sub-repo under the master branch """
        url_string = "https://github.com/GEECS-BELLA/GEECS-Plugins/tree/master/GEECS-Scanner-GUI"
        url = QUrl(url_string)
        if not QDesktopServices.openUrl(url):
            QMessageBox.warning(self, "Open URL", f"Failed to open URL: {url_string}")

    def open_experiment_scanlog(self):
        """ In the default browser, opens the scanlog for the day for the current experiment.  Must be set up in
         `module_open_folder.py` for each experiment. """
        if self.experiment is None:
            return
        url_string = of.get_experiment_scanlog_url(experiment=self.experiment)
        if not url_string:
            QMessageBox.warning(self, "Open URL", f"Scanlog retrieval method not defined for experiment '{self.experiment}'")
            return

        url = QUrl(url_string)
        if not QDesktopServices.openUrl(url):
            QMessageBox.warning(self, "Open URL", f"Failed to open URL: {url_string}")

    def closeEvent(self, event):
        """Upon the GUI closing, also attempts to close child windows and stop any currently-running scans"""
        if self.multiscanner_window:
            self.multiscanner_window.stop_multiscan()
            self.multiscanner_window.close()
        if self.action_library_window:
            self.action_library_window.close()
        if self.element_editor:
            self.element_editor.close()

        if self.RunControl is not None:
            self.stop_scan()

        # TODO find out where the logging.basicConfig is configured... (this doesnt print)
        logging.debug("List of active threads upon closing:")
        for thread in threading.enumerate():
            logging.debug(thread.name)
