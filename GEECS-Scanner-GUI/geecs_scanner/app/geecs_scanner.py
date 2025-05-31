"""
Script to contain the logic for the GEECSScanner GUI.  Can be launched by running this script in python.

-Chris
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np
import numexpr as ne

if TYPE_CHECKING:
    from run_control import RunControl
    from PyQt5.QtWidgets import QWidget
    from geecs_scanner.app.lib.action_control import ActionControl

import sys
import os
from pathlib import Path
import threading
import importlib
import configparser
import logging
from importlib.metadata import version

from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog, QMessageBox
from PyQt5.QtCore import QEvent, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices, QIcon
from .gui.GEECSScanner_ui import Ui_MainWindow
from .lib import MenuBarOption, MenuBarOptionBool, MenuBarOptionStr
from .lib.gui_utilities import display_completer_list, read_yaml_file_to_dict, write_dict_to_yaml_file
from . import SaveElementEditor, MultiScanner, ShotControlEditor, ScanVariableEditor, ActionLibrary
from ..utils import ApplicationPaths as AppPaths, module_open_folder as of
from ..utils.exceptions import ConflictingScanElements, ActionError
from geecs_scanner.data_acquisition.types import ScanConfig, ScanMode

from geecs_scanner.data_acquisition import DatabaseDictLookup

CURRENT_VERSION = "v" + version("geecs-scanner-gui")  # Pulled from `pyproject.toml` for GEECS-Scanner-GUI sub-repo

MAXIMUM_SCAN_SIZE = 1e6  # A simple check to not start a scan if it exceeds this number of shots.

# Lists of options to appear in the menu bar.  Automatically connects these to options in the user's .ini file
BOOLEAN_OPTIONS = ["On-Shot TDMS", "Save Direct on Network"]
STRING_OPTIONS = ["Master Control IP", "Save Hiatus Period (s)"]


class GEECSScannerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.unit_test_mode = 'PYTEST_CURRENT_TEST' in os.environ

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
        if not self.unit_test_mode:
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
        self.populate_available_element_list()

        # Functionality to move elements back and forth between the available and selected elements list
        self.ui.addDeviceButton.clicked.connect(self.add_element_to_selected)
        self.ui.foundDevices.itemDoubleClicked.connect(self.add_element_to_selected)
        self.ui.removeDeviceButton.clicked.connect(self.remove_element_from_selected)
        self.ui.selectedDevices.itemDoubleClicked.connect(self.remove_element_from_selected)

        self.updating_found_list = False
        self.updating_selected_list = False
        self.ui.foundDevices.itemSelectionChanged.connect(self.clear_selected_element_list_selection)
        self.ui.selectedDevices.itemSelectionChanged.connect(self.clear_available_element_list_selection)

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

        # Connect the line edit for the 1d scan variable to the list of available scan variables
        self.scan_variable = ""
        self.scan_variable_list = []
        self.scan_composite_list = []
        self.scan_composite_data = {}
        self.populate_scan_variable_lists()
        self.ui.lineScanVariable.textChanged.connect(self.check_scan_device)
        self.ui.lineScanVariable.installEventFilter(self)

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

        # Tool tip button to display list of scan steps
        self.ui.toolbuttonStepList.setToolTip("")
        self.ui.toolbuttonStepList.clicked.connect(self.show_list_of_steps)

        # Buttons to save the current scan as a preset, delete selected preset, and double-clicking loads the preset
        self.populate_preset_list()
        self.ui.listScanPresets.itemDoubleClicked.connect(self.apply_preset)
        self.ui.presetSaveButton.clicked.connect(lambda: self.save_current_preset())
        self.ui.presetDeleteButton.clicked.connect(self.delete_selected_preset)

        # Buttons to start and stop the current scan
        self.is_starting = False
        self.ui.startScanButton.clicked.connect(self.initialize_and_start_scan)
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
        self.timer.timeout.connect(self.update_gui_status)
        self.ui.scanStatusIndicator.setText("")
        self.timer.start(200)
        self.update_gui_status()

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
        self.element_editor: Optional[SaveElementEditor] = None
        self.multiscanner_window: Optional[MultiScanner] = None
        self.action_library_window: Optional[ActionLibrary] = None
        self.timing_editor: Optional[ShotControlEditor] = None
        self.variable_editor: Optional[ScanVariableEditor] = None

        # Set current GUI mode
        self.toggle_light_dark()

    # # # # # Generic functions used throughout the GUI or use broad sections of the GUI # # # # #

    def eventFilter(self, source, event):
        """ Creates a custom event for text boxes so that the completion suggestions are shown when mouse is clicked """
        if event.type() == QEvent.MouseButtonPress and source == self.ui.experimentDisplay and self.ui.experimentDisplay.isEnabled():
            experiment_names = [f.stem for f in AppPaths.BASE_PATH.iterdir() if f.is_dir()]
            display_completer_list(self, self.ui.experimentDisplay, experiment_names)
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineScanVariable and self.ui.lineScanVariable.isEnabled():
            full_list = (sorted(self.scan_variable_list, key=lambda s: s.lower()) +
                         sorted(self.scan_composite_list, key=lambda s: s.lower()))
            display_completer_list(self, self.ui.lineScanVariable, full_list,
                                   max_visible_lines=30, alphabetical_sorting=False)
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineTimingDevice and self.ui.lineTimingDevice.isEnabled():
            if configuration_names := self.get_list_timing_configurations():
                display_completer_list(self, self.ui.lineTimingDevice, configuration_names)
            return True
        return super().eventFilter(source, event)

    def clear_lists(self):
        """
        Clear all the lists in the GUI.  Used when a fresh start is needed, such as new experiment name or using preset
        """
        self.ui.selectedDevices.clear()
        self.ui.foundDevices.clear()
        self.ui.listScanPresets.clear()

    def update_gui_status(self):
        """ Checks the current status of any ongoing run and updates the GUI accordingly.  Several configurations are
        represented here, but the overall goal is to not allow RunControl to be edited or Start Scan to be clicked when
        a scan is currently running.  If a scan is running, the Stop Scan button is enabled and the progress bar shows
        the current progress.  When multiscan window is open, you can't launch a single scan or change run control """
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

    # # # # # Functions that deal with setting/configuring the experiment name, RunControl, or config settings # # # # #

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

        # Do not change application paths or write to config if window is in unit test mode
        if not self.unit_test_mode:
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
        except (ConnectionError, ConnectionRefusedError) as e:
            logging.error(f"{type(e)} at RunControl: {e}")
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

            self.populate_available_element_list()
            self.ui.lineScanVariable.setText("")
            self.scan_variable = ""
            self.populate_scan_variable_lists()
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
                logging.warning(f"Error occurred when retrieving database dictionary: {e}")
                return {}

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

    def get_list_timing_configurations(self) -> Optional[list[str]]:
        """
        Gets list of available timing device configurations

        :return: None if path not defined, otherwise returns list of yaml files in timing configuration folder
        """
        if self.app_paths is None:
            logging.error("No defined paths for timing configurations")
            return None

        config_folder_path = self.app_paths.shot_control()
        if config_folder_path.exists():
            return [f.stem for f in config_folder_path.iterdir() if f.suffix == ".yaml"]

    def update_shot_control_device(self):
        """ Updates the shot control device when it is changed in the text box, then reinitializes Run Control """
        self.timing_configuration_name = self.ui.lineTimingDevice.text()
        self.reinitialize_run_control()

    # # # # # Functions that update and use the two lists of available and selected save elements # # # # #

    def populate_available_element_list(self):
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

    def add_element_to_selected(self):
        """Move selected files from the "Found" list to the "Selected" list"""
        selected_items = self.ui.foundDevices.selectedItems()
        for item in selected_items:
            self.ui.foundDevices.takeItem(self.ui.foundDevices.row(item))
            self.ui.selectedDevices.addItem(item)

    def remove_element_from_selected(self):
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

    def clear_available_element_list_selection(self):
        """When the selected list is changed, clear the found list selection (without recursively clearing this list)"""
        if not self.updating_selected_list:
            self.updating_found_list = True
            self.ui.foundDevices.clearSelection()
            self.updating_found_list = False

    def clear_selected_element_list_selection(self):
        """When the found list is changed, clear the selected list selection (without recursively clearing this list)"""
        if not self.updating_found_list:
            self.updating_selected_list = True
            self.ui.selectedDevices.clearSelection()
            self.updating_selected_list = False

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
        self.populate_scan_variable_lists()

        logging.info(" ...Done!")

    # # # # #  Functions that open side-gui's when clicked on  # # # # #

    def open_element_editor_new(self):
        """Opens the ScanElementEditor GUI with a blank template.  If Run Control is initialized then the database
        dictionary is passed for device/variable hints.  Afterwards, refresh the lists of the available and selected
        elements."""
        database_dict = self.find_database_dict()

        config_folder = None if self.app_paths is None else self.app_paths.save_devices()
        action_control = None if self.RunControl is None else self.RunControl.get_action_control()

        self.element_editor = SaveElementEditor(main_window=self, database_dict=database_dict,
                                                action_control=action_control, config_folder=config_folder,
                                                load_config=self.load_element_name)
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
        self.update_gui_status()

    def exit_multiscan_mode(self):
        """Cleans up the multiscanner window and resets the enable-ability for buttons on the main window"""
        self.is_in_multiscan = False
        self.multiscanner_window = None

    def open_action_library(self):
        """Opens the multiscanner window, and in the process sets a flag that disables starting scans on the main gui"""
        database_dict = self.find_database_dict()
        action_control = None if self.RunControl is None else self.RunControl.get_action_control()
        self.action_library_window = ActionLibrary(self, database_dict, action_control=action_control)
        self.action_library_window.show()

        self.is_in_action_library = True
        self.update_gui_status()

    def refresh_action_control(self) -> Optional[ActionControl]:
        if self.RunControl is None:
            return None
        else:
            return self.RunControl.get_action_control(experiment_name_refresh=self.experiment)

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
        self.populate_scan_variable_lists()

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

    # # # # #  Functions that work with the scan parameter section of the GUI  # # # # #

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
            self.ui.toolbuttonStepList.setEnabled(False)
            self.ui.toolbuttonStepList.setVisible(False)

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
            self.ui.toolbuttonStepList.setEnabled(True)
            self.ui.toolbuttonStepList.setVisible(True)

    def populate_scan_variable_lists(self):
        """Generates a list of found scan devices from the scan_devices.yaml file"""
        self.scan_variable_list = []
        self.scan_composite_list = []
        self.scan_composite_data = {}

        try:
            if self.app_paths is None:
                raise FileNotFoundError("No defined paths for scan devices")

            scan_device_filename = self.app_paths.scan_devices() / "scan_devices.yaml"
            devices = read_yaml_file_to_dict(scan_device_filename)['single_scan_devices']
            self.scan_variable_list = list(devices.keys())

            composite_device_filename = self.app_paths.scan_devices() / "composite_variables.yaml"
            self.scan_composite_data = read_yaml_file_to_dict(composite_device_filename)['composite_variables']
            self.scan_composite_list = list(self.scan_composite_data.keys())

        except FileNotFoundError as e:
            logging.error(f"Error loading file: {e}")

    def read_device_tag_from_nickname(self, name: str):
        """
        Given a string, use it as a key in the scan_devices.yaml file and return the associated GEECS variable name.
        If the key is not found, assume it is a composite variable and just return the key itself.

        :param name: Selected scan variable to be converted to GEECS variable and/or composite variable
        """
        try:
            scan_variable_filenames = self.app_paths.scan_devices() / "scan_devices.yaml"
            scan_devices = read_yaml_file_to_dict(scan_variable_filenames)['single_scan_devices']
            if name in scan_devices:
                return scan_devices[name]
            else:
                return name

        except Exception as e:
            logging.error(f"Error loading scan_devices.yaml file: {e}")

    def check_scan_device(self):
        """Checks what is inputted into the scan variable selection box against the list of scan variables.  Otherwise,
        reset the line edit."""
        scan_device = self.ui.lineScanVariable.text()
        if not scan_device:
            return
        elif scan_device in self.scan_variable_list:
            self.scan_variable = scan_device
            self.ui.labelStartValue.setText("Start Value: (abs)")
            self.ui.labelStopValue.setText("Stop Value: (abs)")
        elif scan_device in self.scan_composite_list:
            self.scan_variable = scan_device
            mode = self.scan_composite_data[scan_device]['mode'][:3]
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

            shot_array = self.build_shot_array()
            self.ui.lineNumShots.setText(str(len(shot_array)))
            self._update_list_of_steps(shot_array=shot_array, skip_tcp_request=True)

        except ValueError:
            self.ui.lineNumShots.setText("N/A")
        except SyntaxError:
            self.ui.lineNumShots.setText("Error")

    def build_shot_array(self) -> list[float]:
        """Given the parameters for a 1D scan, generate an array with the value of the scan variable for each shot.

        :return: list of scan device values for each shot

        :raises:
            ValueError: Scan parameters are not set up correctly for a variety of reasons
        """
        if self.scan_step_size == 0:
            raise ValueError("Step size must be nonzero")
        elif self.scan_shot_per_step <= 0:
            raise ValueError("Shots per step must be greater than zero")
        elif abs((self.scan_stop - self.scan_start) / self.scan_step_size) * self.scan_shot_per_step > MAXIMUM_SCAN_SIZE:
            raise ValueError("Number of shots exceeds maximum scan size")
        elif self.scan_shot_per_step > MAXIMUM_SCAN_SIZE:
            raise ValueError("Number of shots exceeds maximum scan size")
        else:
            array = []
            current = self.scan_start
            positive = self.scan_start < self.scan_stop
            while ((positive and current <= self.scan_stop)
                   or (not positive and current >= self.scan_stop)):
                array.extend([current] * self.scan_shot_per_step)
                if positive:
                    current = round(current + abs(self.scan_step_size), 10)
                else:
                    current = round(current - abs(self.scan_step_size), 10)
            return array

    def _update_list_of_steps(self, shot_array: Optional[list[float]] = None, skip_tcp_request: bool = False):
        """
        Updates the tool tip button with the current steps for the scan, which is visible upon hovering over the button

        :param shot_array: the value of the scan parameter on each shot
        :param skip_tcp_request: if True, will not evaluate a relative composite variable as this takes longer

        :raises:
            SyntaxError: Numerical expression for composite variable is bad
            ValueError: Device variable returned for relative composite variables was not a float
        """
        if shot_array is None:
            self.ui.toolbuttonStepList.setToolTip("")
            return

        # Gather all the bins
        bins = np.unique(shot_array)

        # Set up lists to hold strings of devices and associated values
        scan_device = self.ui.lineScanVariable.text()
        devices: list[str] = [scan_device]
        device_values: list[np.ndarray] = [bins]

        # If the variable is a composite variable, get each step value for each device
        if scan_device in self.scan_composite_list:
            for component in self.scan_composite_data[scan_device]['components']:
                devices.append(f"{component['device']}:{component['variable']}")

                expression_results = np.zeros(len(bins))
                for i in range(len(bins)):
                    try:
                        expression_results[i] = ne.evaluate(component['relation'], local_dict={"composite_var": bins[i]})
                    except SyntaxError:
                        self.ui.toolbuttonStepList.setToolTip("")
                        raise SyntaxError(f"Bad relation syntax in composite variable '{scan_device}' at "
                                          f"'{component['device']}:{component['variable']}'")

                # If the composite variable is relative and not in scan, use a get command to calculate the actual value
                if self.scan_composite_data[scan_device]['mode'] in ['relative'] and self.RunControl is not None:
                    # Can't perform get command if scan is ongoing
                    if self.RunControl.is_active() or self.is_starting:
                        self.ui.toolbuttonStepList.setToolTip("Check back after scan")
                        return

                    # Don't execute get command unless button was explicitly clicked on
                    if skip_tcp_request:
                        self.ui.toolbuttonStepList.setToolTip("Click to show")
                        return

                    action_control = self.RunControl.get_action_control()
                    current_value = action_control.return_device_value(device_name=component['device'],
                                                                       variable=component['variable'])
                    try:
                        expression_results += float(current_value)
                    except (ValueError, TypeError):
                        raise ValueError(f"{component['device']}:{component['variable']} must return a float, "
                                         f"instead returned '{current_value}'")

                device_values.append(expression_results)

        # Transpose the values into a numpy array for easier indexing
        transposed_device_values = np.vstack(device_values).T

        # Compile the tool tip string
        message_string = "Bin"
        for device in devices:
            message_string += f",   {device}"
        for i in range(len(bins)):
            message_string += f"\n{i}:"
            for j in range(len(devices)):
                message_string += f"   {transposed_device_values[i, j]:.4f}"

        self.ui.toolbuttonStepList.setToolTip(message_string)

    def show_list_of_steps(self):
        """ Displays a pop-up with the list of steps for each device variable.  Errors are handled in the case that
         either a scan parameter is incorrectly configured or there was trouble getting a composite variable value """
        try:
            self.calculate_num_shots()
            self._update_list_of_steps(self.build_shot_array())
            QMessageBox.about(self, "Scan Steps", f"{self.ui.toolbuttonStepList.toolTip()}")
        except (ValueError, SyntaxError) as e:
            QMessageBox.about(self, "Scan Steps", f"{type(e)}:\n{e}")
        except ActionError as e:
            QMessageBox.about(self, "Scan Steps", f"{type(e)}:\n{e.message}")

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

    # # # # #  Functions for the saving, deletion, and usage of scan presets  # # # # #

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

    def save_current_preset(self, filename: Optional[str] = None):
        """Takes the current scan configuration and prompts the user if they would like to save it as a preset.  If so,
        the user give a filename to save under and the information is compiled into a yaml that "apply_preset" uses

        :param filename: Can optionally provide the filename as an argument, primarily used for unit tests
        """
        if self.app_paths is None:
            logging.error("No defined paths for scan presets")
            return

        if filename is not None:
            text = filename
            ok = True
        else:
            if self.unit_test_mode:
                return  # In unit test mode a filename must be given
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
            write_dict_to_yaml_file(preset_file, settings)

            self.populate_preset_list()

    def apply_preset(self):
        """Loads the yaml file selected in the preset list, clears all current information on the GUI, then
        systematically populate everything so that the GUI is equivalent to when it was saved as a preset"""
        selected_element = self.ui.listScanPresets.selectedItems()
        if selected_element:
            self.apply_preset_from_name(selected_element[0].text())

    def apply_preset_from_name(self, preset_name: str, load_save_elements: bool = True, load_scan_params: bool = True):
        """
        :param preset_name: Name of the preset
        :param load_save_elements: Defaults to True, flag to load the save elements from a preset file
        :param load_scan_params: Defaults to True, flag to load the scan parameters from a preset file
        """
        preset_filename = self.app_paths.presets() / (preset_name + ".yaml")
        settings = read_yaml_file_to_dict(preset_filename)

        self.ui.textEditScanInfo.setText(str(settings['Info']))

        if load_save_elements:
            self.clear_lists()
            self.populate_available_element_list()
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
        if selected_element:
            preset_filename = self.app_paths.presets() / (selected_element[0].text() + ".yaml")
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

    # # # # #  Functions that contain the logic for starting and stopping a scan  # # # # #

    def check_for_errors(self) -> bool:
        """Checks the full GUI for any blatant errors.  To be used before submitting a scan to be run"""
        # TODO Need to add more logic in here.  IE, at least 1 shot, at least 1 save device, etc etc
        if not self.repetition_rate > 0:
            logging.error("Need nonzero repetition rate")
            return True

        if self.ui.scanRadioButton.isChecked():
            try:
                self._update_list_of_steps(self.build_shot_array())
            except (ValueError, SyntaxError) as e:
                logging.error(f"{type(e)}: {e}")
                return True
            except ActionError as e:
                logging.error(f"{type(e)}: {e.message}")
                return True

        return False

    def initialize_and_start_scan(self):
        """Compiles the information from the GUI into a dictionary that can be used by scan_manager.  This dictionary is
        then sent to RunControl to be submitted for a scan."""
        if not self.check_for_errors():
            # From the information provided in the GUI, create a scan configuration file and submit `scan_manager.py`
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
                filename = self.app_paths.save_devices() / (self.ui.selectedDevices.item(i).text() + ".yaml")
                try:
                    new_element = read_yaml_file_to_dict(filename)
                    self.combine_elements(save_device_list, new_element['Devices'])

                    if 'setup_action' in new_element:
                        setup_action = new_element['setup_action']
                        list_of_setup_steps.extend(setup_action['steps'])

                    if 'closeout_action' in new_element:
                        setup_action = new_element['closeout_action']
                        list_of_closeout_steps.extend(setup_action['steps'])

                except FileNotFoundError:
                    logging.error(f"FileNotFound Error: {filename}")
                    QMessageBox.warning(self, 'Conflicting Save Elements', f"FileNotFound Error: {filename}", QMessageBox.Ok)
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
                # scan_config = {
                #     'device_var': scan_variable_tag,
                #     'start': self.scan_start,
                #     'end': self.scan_stop,
                #     'step': self.scan_step_size,
                #     'wait_time': (self.scan_shot_per_step + 0.5)/self.repetition_rate
                # }
                scan_config = ScanConfig(
                    device_var = scan_variable_tag,
                    start = self.scan_start,
                    end = self.scan_stop,
                    step = self.scan_step_size,
                    wait_time = (self.scan_shot_per_step + 0.5) / self.repetition_rate,
                    scan_mode = 'STANDARD'
                )
            elif self.ui.noscanRadioButton.isChecked() or self.ui.backgroundRadioButton.isChecked():
                # scan_config = {
                #     'device_var': 'noscan',
                #     'wait_time': (self.noscan_num + 0.5)/self.repetition_rate
                # }
                scan_config = ScanConfig(
                    # device_var = 'noscan',
                    wait_time = (self.noscan_num + 0.5)/self.repetition_rate,
                    scan_mode='NOSCAN'

                )
            else:
                scan_config = None
            scan_config.background = str(self.ui.backgroundRadioButton.isChecked())

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

    # # # # #  Functions for interacting with the toolbar buttons on the top  # # # # #

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

    def toggle_light_dark(self):
        """ Toggles between light and dark mode when the toolbar option is clicked """
        mode = 'dark' if self.ui.actionDarkMode.isChecked() else 'light'
        base_path = Path(__file__).parent / "gui"

        def read_stylesheet(filename):
            with open(filename, "r") as file:
                return file.read()

        # List of Widgets that also need updating (only widgets that can be active at the same time as the main window)
        gui_list: list[Optional[QWidget]] = [self, self.multiscanner_window, self.action_library_window]
        for widget in gui_list:
            if widget is not None:
                if mode == 'light':
                    widget.setStyleSheet(read_stylesheet(base_path / "light_mode.qss"))
                elif mode == 'dark':
                    widget.setStyleSheet(read_stylesheet(base_path / "dark_mode.qss"))

    # # # # #  Handling the close event  # # # # #

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
