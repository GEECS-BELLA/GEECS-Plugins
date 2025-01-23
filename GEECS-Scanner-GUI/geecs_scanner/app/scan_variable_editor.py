from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from . import GEECSScannerWindow
    from PyQt5.QtWidgets import QLineEdit

import yaml
import logging
from pathlib import Path

from PyQt5.QtWidgets import QDialog, QCompleter
from PyQt5.QtCore import Qt, QEvent

from .gui.ScanDeviceEditor_ui import Ui_Dialog


class ScanVariableEditor(QDialog):
    """
    GUI for viewing/editing scan save elements.  To be opened from the GEECSScanner GUI.  The code here is organized by
    having master dictionaries on the backend with all the information, then the GUI adds/subtracts/changes this
    dictionary.  Upon the dictionary changing or a different selection made on the GUI, the visible information on the
    GUI changes to reflect what the user is currently looking at.
    """

    def __init__(self, main_window: GEECSScannerWindow,
                 database_dict: Optional[dict] = None,
                 config_folder: Optional[Path] = None):
        """
        Initializes the GUI

        :param main_window: the main gui window, used only to set the visual stylesheet
        :param database_dict: dictionary that contains all devices and variables in the selected experiment
        :param config_folder: folder that contains scan variables and composite variables for this experiment
        """
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Assign database dictionary to class variable
        self.database_dict = database_dict or {}

        # Paths to the two yaml files
        # TODO if config_folder is None, don't set file paths and put dialog in a limited operation mode
        self.file_variables = config_folder / "scan_devices.yaml"
        self.file_composite = config_folder / "composite_variables.yaml"
        # TODO If the two yaml files do not exist, go ahead and make new, blank files

        # Initialize lists for the scan variables and composite variables
        self.scan_variable_list = []
        self.scan_composite_list = []
        self.scan_variable_data = {}
        self.scan_composite_data = {}
        self.update_variable_information_from_files()

        # Functionality to Scan Variables section
        self.ui.lineVariableNickname.installEventFilter(self)
        self.ui.lineVariableDevice.installEventFilter(self)
        self.ui.lineVariableVariable.installEventFilter(self)

        # Functionality to Composite Variables section
        self.ui.lineCompositeNickname.installEventFilter(self)
        self.ui.lineCompositeDevice.installEventFilter(self)
        self.ui.lineCompositeVariable.installEventFilter(self)

        # Apply the stylesheet of the main window
        self.setStyleSheet(main_window.styleSheet())

    def eventFilter(self, source, event):
        """ Custom event for the text boxes so that the completion suggestions are shown when mouse is clicked """
        # Nickname completer prompts
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineVariableNickname:
            self.display_completer_list(location=self.ui.lineVariableNickname, completer_list=self.scan_variable_list)
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineCompositeNickname:
            self.display_completer_list(location=self.ui.lineCompositeNickname, completer_list=self.scan_composite_list)
            return True

        # Device name completer prompts
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineVariableDevice:
            self.display_completer_list(location=self.ui.lineVariableDevice,
                                        completer_list=sorted(self.database_dict.keys()))
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineCompositeDevice:
            self.display_completer_list(location=self.ui.lineCompositeDevice,
                                        completer_list=sorted(self.database_dict.keys()))
            return True

        # Variable name completer prompts
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineVariableVariable:
            self.display_completer_variable_list(list_location=self.ui.lineVariableVariable,
                                                 device_location=self.ui.lineVariableDevice)
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineCompositeVariable:
            self.display_completer_variable_list(list_location=self.ui.lineCompositeVariable,
                                                 device_location=self.ui.lineCompositeDevice)
            return True

        return super().eventFilter(source, event)

    def update_variable_information_from_files(self):
        """ Loads the data from the two yaml files and populates the lists of nicknames """
        self.scan_variable_list = []
        self.scan_composite_list = []
        self.scan_variable_data = {}
        self.scan_composite_data = {}

        try:
            with open(self.file_variables, 'r') as file:
                self.scan_variable_data = yaml.safe_load(file)
                devices = self.scan_variable_data['single_scan_devices']
                self.scan_variable_list = list(devices.keys())

            with open(self.file_composite, 'r') as file:
                self.scan_composite_data = yaml.safe_load(file)
                composite_vars = self.scan_composite_data['composite_variables']
                self.scan_composite_list = list(composite_vars.keys())
        except FileNotFoundError as e:
            logging.error(f"Error loading file: {e}")

    def display_completer_list(self, location: QLineEdit, completer_list: list[str]):
        """ Displays a completer list at a given location

        :param location: GUI element at which to show the completer list
        :param completer_list: strings to show in the completer pop-up
        """
        location.selectAll()
        completer = QCompleter(completer_list, self)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseSensitive)

        location.setCompleter(completer)
        location.setFocus()
        completer.complete()

    def display_completer_variable_list(self, list_location: QLineEdit, device_location: QLineEdit):
        """ Displays list of variables at one location using the device name at another location

        :param list_location: GUI element at which to show the completer list
        :param device_location: GUI element where the device name is given
        """
        device_name = device_location.text().strip()
        if device_name in self.database_dict:
            variable_list = sorted(self.database_dict[device_name].keys())
            self.display_completer_list(location=list_location, completer_list=variable_list)
