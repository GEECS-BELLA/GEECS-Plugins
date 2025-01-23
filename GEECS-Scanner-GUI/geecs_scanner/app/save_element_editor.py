"""
GUI that displays the organization of individual scan elements and allows the user to edit these scan elements.  Also
allows for a user to load and save the information displayed into this GUI to/from .yaml files for scan elements.

# TODO Currently the 'run' action is not implemented, but potentially could be a feature to run a python script

-Chris
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Any

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QLineEdit
    from . import GEECSScannerWindow

import yaml
import logging
from pathlib import Path
from PyQt5.QtWidgets import QDialog, QCompleter, QPushButton, QFileDialog
from PyQt5.QtCore import Qt, QEvent
from .gui.ScanElementEditor_ui import Ui_Dialog
from .lib import ActionControl


def get_default_device_dictionary() -> dict[str, bool | list[Any]]:
    """
    :return: Default dictionary for devices when they are added to the element's list of devices
    """
    return {
        'variable_list': [],
        'synchronous': True,
        'save_nonscalar_data': True
    }


def get_new_action(action) -> Union[None, dict[str, str]]:
    """
    Translates a given action keyword to a default dictionary that is populated into the action list.

    :param action: action keyword
    :return: default dictionary for the associated action
    """
    # TODO Can probably convert this from an if-else block to a dictionary...
    default = None
    if action == 'set':
        default = {
            'action': 'set',
            'device': '',
            'variable': '',
            'value': ''
        }
    elif action == 'get':
        default = {
            'action': 'get',
            'device': '',
            'variable': '',
            'expected_value': ''
        }
    elif action == 'wait':
        default = {
            'wait': ''
        }
    elif action == 'execute':
        default = {
            'action': 'execute',
            'action_name': ''
        }
    elif action == 'run':
        default = {
            'action': 'run',
            'file_name': '',
            'class_name': ''
        }
    return default


# List of available actions, to be used by the completer for the add action line edit
list_of_actions = [
    'set',
    'get',
    'wait',
    'execute',
    # 'run'
]


def parse_variable_text(text) -> Union[int, float, str]:
    """ Attempts to convert a string first to an int, then a float, and finally just returns the string if unsuccessful

    :param text: string
    :return: either an int, float, or string of the input, in that order
    """
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


class SaveElementEditor(QDialog):
    """
    GUI for viewing/editing scan save elements.  To be opened from the GEECSScanner GUI.  The code here is organized by
    having master dictionaries on the backend with all the information, then the GUI adds/subtracts/changes this
    dictionary.  Upon the dictionary changing or a different selection made on the GUI, the visible information on the
    GUI changes to reflect what the user is currently looking at.
    """

    def __init__(self, main_window: GEECSScannerWindow, database_dict: Optional[dict] = None,
                 config_folder: Path = Path('.'), load_config: Optional[Path] = None):
        """
        Initializes the GUI

        :param main_window: the main gui window, used only to set the visual stylesheet
        :param database_dict: dictionary that contains all devices and variables in the selected experiment
        :param config_folder: folder that contains other element config files for this experiment
        :param load_config: optional; filename to populate the initial state of the GUI's backend dictionary
        """
        super().__init__()

        # Dummy button to try and contain where pressing "Enter" goes, but TODO find a better solution
        self.dummyButton = QPushButton("", self)
        self.dummyButton.setDefault(True)
        self.dummyButton.setVisible(False)

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Dictionary containing all the devices and variables in the experiment
        self.database_dict = database_dict or {}

        # Backend dictionaries for the element devices and actions.  Each action dictionary is an ordered list
        self.devices_dict = {}
        self.actions_dict = {
            'setup': [],
            'closeout': []
        }

        # Functionality to add and remove devices, show device list when selecting line edit, and update the visible
        #  device information when the selection in the device list is changed
        self.ui.lineDeviceName.installEventFilter(self)
        self.ui.buttonAddDevice.clicked.connect(self.add_device)
        self.ui.buttonRemoveDevice.clicked.connect(self.remove_device)
        self.ui.listDevices.itemSelectionChanged.connect(self.update_variable_list)

        # Functionality to add and remove variables from the selected device, as well as show the list of variables
        #  when the variable line edit is selected
        self.ui.lineVariableName.installEventFilter(self)
        self.ui.buttonAddVariable.clicked.connect(self.add_variable)
        self.ui.buttonRemoveVariable.clicked.connect(self.remove_variable)

        # Update the device flags when either of the checkboxes are clicked
        self.ui.checkboxSynchronous.clicked.connect(self.update_device_checkboxes)
        self.ui.checkboxSaveNonscalar.clicked.connect(self.update_device_checkboxes)

        # Make the action line edit only editable from the dropdown completer list of available actions
        self.ui.lineActionName.setReadOnly(True)
        self.ui.lineActionName.installEventFilter(self)

        # Functionality to add and remove actions to and from the visible list
        self.ui.listActions.itemSelectionChanged.connect(self.focus_action)
        self.ui.buttonAddAction.clicked.connect(self.add_action)
        self.ui.buttonRemoveAction.clicked.connect(self.remove_action)

        # Functionality for the options associated with each of the types of actions.  A variable stores what action
        #  type is currently selected, and options 1 and 2 have dropdowns to show available devices and variables when
        #  the 'set' and 'get' types are visible.  When any of the 3 line edits are changed, update the dictionary
        self.action_mode = None
        self.ui.lineActionOption1.installEventFilter(self)
        self.ui.lineActionOption2.installEventFilter(self)
        self.ui.lineActionOption1.editingFinished.connect(self.update_action_info)
        self.ui.lineActionOption2.editingFinished.connect(self.update_action_info)
        self.ui.lineActionOption3.editingFinished.connect(self.update_action_info)

        # Functionality to change the ordering of the actions in the lists
        self.ui.buttonMoveSooner.clicked.connect(self.move_action_sooner)
        self.ui.buttonMoveLater.clicked.connect(self.move_action_later)

        # Functionality to move actions to/from the setup and closeout lists
        self.ui.radioIsPost.clicked.connect(self.set_as_closeout)
        self.ui.radioIsSetup.clicked.connect(self.set_as_setup)

        # If a .yaml file is given, load the dictionary into the backend dictionaries to start with
        self.config_folder = None if config_folder is None else Path(config_folder)
        if load_config is not None:
            self.load_settings_from_file(config_folder / load_config)

        self.action_control: Optional[ActionControl] = None
        self.ui.buttonPerformSetupActions.setEnabled(False)
        self.ui.buttonPerformPostscanActions.setEnabled(False)
        self.ui.buttonEnableActions.clicked.connect(self.initialize_action_control)
        self.ui.buttonPerformSetupActions.clicked.connect(self.perform_setup_actions)
        self.ui.buttonPerformPostscanActions.clicked.connect(self.perform_postscan_actions)

        # Buttons at the bottom to save, open, and close
        self.ui.buttonWindowSave.clicked.connect(self.save_element)
        self.ui.buttonWindowClose.clicked.connect(self.close_window)
        self.ui.buttonWindowLoad.clicked.connect(self.open_element)

        # At the end of initialization, update all the visible information on the GUI
        self.update_device_list()
        self.update_action_list()
        self.update_action_display()
        self.setStyleSheet(main_window.styleSheet())

    def eventFilter(self, source, event):
        """Creates a custom event for the text boxes so that the completion suggestions are shown when mouse is clicked
        """
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionName:
            self.show_action_list()
            self.ui.buttonAddAction.setDefault(True)
            return True
        elif event.type() == QEvent.MouseButtonPress and source == self.ui.lineDeviceName:
            self.show_device_list(self.ui.lineDeviceName)
            self.ui.buttonAddDevice.setDefault(True)
            return True
        elif event.type() == QEvent.MouseButtonPress and source == self.ui.lineVariableName:
            self.show_variable_list(self.ui.lineVariableName)
            self.ui.buttonAddVariable.setDefault(True)
            return True
        elif (event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionOption1
              and self.action_mode in ['set', 'get']):
            self.show_device_list(self.ui.lineActionOption1)
            self.dummyButton.setDefault(True)
            return True
        elif (event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionOption2
              and self.action_mode in ['set', 'get']):
            self.show_variable_list(self.ui.lineActionOption2, source='action')
            self.dummyButton.setDefault(True)
            return True
        return super().eventFilter(source, event)

    def show_device_list(self, location: QLineEdit):
        """
        Shows the list of available experimental devices as a hint completer

        :param location: gui element where the list should be shown
        """
        if location.isEnabled():
            location.selectAll()
            completer = QCompleter(sorted(self.database_dict.keys()), self)
            completer.setCompletionMode(QCompleter.PopupCompletion)
            completer.setCaseSensitivity(Qt.CaseSensitive)

            location.setCompleter(completer)
            location.setFocus()
            completer.complete()

    def show_variable_list(self, location: QLineEdit, source: str = 'device'):
        """
        Shows the list of variables for the currently selected device

        :param location: gui element where the list should be shown
        :param source: if the variable list is in the device or action section, defaults to device section
        """
        if location.isEnabled():
            if source == 'device':
                device_name = self.get_selected_device_name()
            elif source == 'action':
                device_name = self.ui.lineActionOption1.text().strip()
            else:
                device_name = None

            if device_name in self.database_dict:
                location.selectAll()
                completer = QCompleter(sorted(self.database_dict[device_name].keys()), self)
                completer.setCompletionMode(QCompleter.PopupCompletion)
                completer.setCaseSensitivity(Qt.CaseSensitive)

                location.setCompleter(completer)
                location.setFocus()
                completer.complete()

    def update_device_list(self):
        """Update the visible list of devices from the dictionary on the backend
        """
        self.ui.listDevices.clear()

        for device in self.devices_dict:
            self.ui.listDevices.addItem(device)

        self.update_variable_list()

    def add_device(self):
        """Adds a device to the element device list, based on what is entered in the device line edit"""
        text = self.ui.lineDeviceName.text().strip()
        if text and text not in self.devices_dict:
            self.devices_dict[text] = get_default_device_dictionary()
            if text in self.database_dict and 'timestamp' in self.database_dict[text]:
                self.devices_dict[text]['variable_list'].append('timestamp')
            self.update_device_list()

    def remove_device(self):
        """Removes a device from the element device list, based on what is currently selected"""
        selected_device = self.ui.listDevices.selectedItems()
        if not selected_device:
            return
        for selection in selected_device:
            text = selection.text()
            if text in self.devices_dict:
                del self.devices_dict[text]
        self.update_device_list()

    def get_selected_device_name(self) -> Optional[str]:
        """Returns the name of the currently-selected device on the GUI"""
        selected_device = self.ui.listDevices.selectedItems()
        no_selection = not selected_device
        if no_selection:
            return None

        device_name = None
        for selection in selected_device:
            text = selection.text()
            if text in self.devices_dict:
                device_name = text
        return device_name

    def get_selected_device(self):  # TODO improve type hinting for complex dictionaries
        """Returns the device information on the currently-selected device"""
        device_name = self.get_selected_device_name()
        if device_name is None:
            return None
        else:
            return self.devices_dict[device_name]

    def update_variable_list(self):
        """Updates the visible variable information based on the currently-selected device"""
        self.ui.listVariables.clear()

        device = self.get_selected_device()
        enable_variables = device is not None

        self.ui.checkboxSynchronous.setEnabled(enable_variables)
        self.ui.checkboxSaveNonscalar.setEnabled(enable_variables)
        self.ui.buttonAddVariable.setEnabled(enable_variables)
        self.ui.buttonRemoveVariable.setEnabled(enable_variables)
        self.ui.lineVariableName.setEnabled(enable_variables)

        if device is None:
            return

        for variable in device['variable_list']:
            self.ui.listVariables.addItem(variable)
        self.ui.checkboxSynchronous.setChecked(device['synchronous'])
        self.ui.checkboxSaveNonscalar.setChecked(device['save_nonscalar_data'])

    def add_variable(self):
        """Adds variable to the element's device variable list, based on the variable line edit"""
        device = self.get_selected_device()
        if device is not None:
            text = self.ui.lineVariableName.text().strip()
            if text and text not in device['variable_list']:
                if "variable_list" in device:
                    device["variable_list"].append(text)
                else:
                    device["variable_list"] = [text]
                self.update_variable_list()

    def remove_variable(self):
        """Removes variable from the element's device variable list, based on what is currently selected"""
        device = self.get_selected_device()
        if device is not None:
            selected_variable = self.ui.listVariables.selectedItems()
            if not selected_variable:
                return
            for selection in selected_variable:
                text = selection.text()
                if text in device['variable_list']:
                    device["variable_list"].remove(text)
            self.update_variable_list()

    def update_device_checkboxes(self):
        """Updates the checkboxes for the device flags"""
        device = self.get_selected_device()
        if device is not None:
            device['synchronous'] = self.ui.checkboxSynchronous.isChecked()
            device['save_nonscalar_data'] = self.ui.checkboxSaveNonscalar.isChecked()
            self.update_variable_list()

    def show_action_list(self):
        """Shows the list of available actions at the add action line edit"""
        self.ui.listActions.clearSelection()
        completer = QCompleter(list_of_actions, self)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.ui.lineActionName.setCompleter(completer)
        self.ui.lineActionName.setFocus()
        completer.complete()

    @staticmethod
    def generate_action_description(action: dict[str, list]) -> str:
        """For each action in the list, generate a string that displays all the information for that action step"""
        description = "???"
        if action.get("wait") is not None:
            description = f"wait {action['wait']}"
        elif action['action'] == 'execute':
            description = f"execute {action['action_name']}"
        elif action['action'] == 'run':
            description = "run"
        elif action['action'] == 'set':
            description = f"{action['action']} {action['device']}:{action['variable']} {action.get('value')}"
        elif action['action'] == 'get':
            description = f"{action['action']} {action['device']}:{action['variable']} {action.get('expected_value')}"
        return description

    def update_action_list(self, index: Optional[int] = None):
        """Updates the visible list of actions on the GUI according to the current state of the action dictionary"""
        self.ui.listActions.clear()
        self.dummyButton.setDefault(True)
        for item in self.actions_dict['setup']:
            self.ui.listActions.addItem(self.generate_action_description(item))
        self.ui.listActions.addItem("---Scan---")
        for item in self.actions_dict['closeout']:
            self.ui.listActions.addItem(self.generate_action_description(item))
        if index is not None and 0 <= index < self.ui.listActions.count():
            self.ui.listActions.setCurrentRow(index)

    def add_action(self):
        """Appends action to the end of either the setup or closeout action list based on the currently-selected
        radio button.  The action is specified by the action line edit."""
        text = self.ui.lineActionName.text().strip()
        if text:
            if self.ui.radioIsSetup.isChecked():
                self.actions_dict['setup'].append(get_new_action(text))
            else:
                self.actions_dict['closeout'].append(get_new_action(text))
        self.update_action_list()

    def remove_action(self):
        """Removes the currently-selected action from the list of actions"""
        current_selection = self.get_action_list_and_index()
        if current_selection is not None and current_selection[0] is not None:
            action_list, i, absolute_index = current_selection
            del action_list[i]
        self.update_action_list()

    def focus_action(self):
        """Refreshes the list of actions, and in the process removes any currently-selected action"""
        self.ui.lineActionName.clear()
        self.update_action_display()

    def update_action_display(self):
        """Updates the visible list of information based on the currently-selected action"""
        current_selection = self.get_action_list_and_index()
        action = None
        i = None
        absolute_index = None
        if current_selection is not None:
            action_list, i, absolute_index = current_selection
            if action_list is None:
                action = None
            else:
                action = action_list[i]

        self.action_mode = None
        self.ui.labelActionOption1.setText("")
        self.ui.labelActionOption2.setText("")
        self.ui.labelActionOption3.setText("")
        self.ui.lineActionOption1.setText("")
        self.ui.lineActionOption2.setText("")
        self.ui.lineActionOption3.setText("")
        self.ui.lineActionOption1.setEnabled(False)
        self.ui.lineActionOption2.setEnabled(False)
        self.ui.lineActionOption3.setEnabled(False)

        if action is not None:
            if i == absolute_index:
                self.ui.radioIsSetup.setChecked(True)
            else:
                self.ui.radioIsPost.setChecked(True)

            if action.get("wait") is not None:
                self.action_mode = 'wait'
                self.ui.labelActionOption1.setText("Wait Time (s):")
                self.ui.lineActionOption1.setEnabled(True)
                self.ui.lineActionOption1.setText(str(action.get("wait")))
            elif action['action'] == 'execute':
                self.action_mode = 'execute'
                self.ui.labelActionOption1.setText("Action Name:")
                self.ui.lineActionOption1.setEnabled(True)
                self.ui.lineActionOption1.setText(action.get("action_name"))
            elif action['action'] == 'run':
                self.action_mode = 'run'
                self.ui.labelActionOption1.setText("File Location:")
                self.ui.lineActionOption1.setEnabled(True)
                self.ui.lineActionOption1.setText(action.get("file_name"))
                self.ui.labelActionOption2.setText("Class Name:")
                self.ui.lineActionOption2.setEnabled(True)
                self.ui.lineActionOption2.setText(action.get("class_name"))
            elif action['action'] == 'set':
                self.action_mode = 'set'
                self.ui.labelActionOption1.setText("GEECS Device Name:")
                self.ui.lineActionOption1.setEnabled(True)
                self.ui.lineActionOption1.setText(action.get("device"))
                self.ui.labelActionOption2.setText("Variable Name:")
                self.ui.lineActionOption2.setEnabled(True)
                self.ui.lineActionOption2.setText(action.get("variable"))
                self.ui.labelActionOption3.setText("Set Value:")
                self.ui.lineActionOption3.setEnabled(True)
                self.ui.lineActionOption3.setText(str(action.get("value")))
            elif action['action'] == 'get':
                self.action_mode = 'get'
                self.ui.labelActionOption1.setText("GEECS Device Name:")
                self.ui.lineActionOption1.setEnabled(True)
                self.ui.lineActionOption1.setText(action.get("device"))
                self.ui.labelActionOption2.setText("Variable Name:")
                self.ui.lineActionOption2.setEnabled(True)
                self.ui.lineActionOption2.setText(action.get("variable"))
                self.ui.labelActionOption3.setText("Expected Value:")
                self.ui.lineActionOption3.setEnabled(True)
                self.ui.lineActionOption3.setText(str(action.get("expected_value")))

    def update_action_info(self):
        """Updates the backend action dictionary when one of the line edits for options is changed"""
        current_selection = self.get_action_list_and_index()
        if current_selection is None:
            action = None
        else:
            action_list, i, absolute_index = current_selection
            if action_list is None:
                action = None
            else:
                action = action_list[i]

        if action is None:
            return
        if action.get("wait") is not None:
            action['wait'] = parse_variable_text(self.ui.lineActionOption1.text().strip())
        elif action['action'] == 'execute':
            action['action_name'] = self.ui.lineActionOption1.text().strip()
        elif action['action'] == 'run':
            action["file_name"] = self.ui.lineActionOption1.text().strip()
            action["class_name"] = self.ui.lineActionOption2.text().strip()
        elif action['action'] == 'set':
            action["device"] = self.ui.lineActionOption1.text().strip()
            action["variable"] = self.ui.lineActionOption2.text().strip()
            action["value"] = parse_variable_text(self.ui.lineActionOption3.text().strip())
        elif action['action'] == 'get':
            action["device"] = self.ui.lineActionOption1.text().strip()
            action["variable"] = self.ui.lineActionOption2.text().strip()
            action["expected_value"] = parse_variable_text(self.ui.lineActionOption3.text().strip())

        # TODO There is a weird bug where hitting enter on Option 1 presses "Move Sooner"
        current_selection = self.get_action_list_and_index()
        if current_selection is None:
            return
        action_list, i, index = current_selection
        self.update_action_list(index=index)

    def get_action_list_and_index(self) -> Optional[tuple[Optional[list], Optional[int], int]]:
        """Finds the location of the currently-selected action in the visible list on the GUI.  This is tricky since
        the GUI has two different lists separated by a dummy list element.

        :return: If no selection, returns None.  If the dummy element is selected, return a tuple with None as the list
        and the current index in the list.  If a real action is selected, return the action list where the action is
        located (setup or closeout), the relative index in that list, and the index in the GUI list
        """
        selected_action = self.ui.listActions.selectedItems()
        if not selected_action:
            return
        for action in selected_action:
            absolute_index = self.ui.listActions.row(action)
            setup_length = len(self.actions_dict['setup'])
            if absolute_index < setup_length:
                action_list = self.actions_dict['setup']
                index = absolute_index
            elif absolute_index == setup_length:
                return None, None, absolute_index
            else:
                action_list = self.actions_dict['closeout']
                index = absolute_index - 1 - setup_length
            return action_list, index, absolute_index

    def move_action_sooner(self):
        """Moves the selected action to an earlier position in the same list"""
        current_selection = self.get_action_list_and_index()
        if current_selection is None:
            return
        action_list, i, index = current_selection
        if action_list is not None and 0 < i < len(action_list):
            action_list[i], action_list[i - 1] = action_list[i - 1], action_list[i]
            index -= 1
        self.update_action_list(index=index)

    def move_action_later(self):
        """Moves the selected action to a later position in the same list"""
        current_selection = self.get_action_list_and_index()
        if current_selection is None:
            return
        action_list, i, index = current_selection
        if action_list is not None and 0 <= i < len(action_list) - 1:
            action_list[i], action_list[i + 1] = action_list[i + 1], action_list[i]
            index += 1
        self.update_action_list(index=index)

    def set_as_setup(self):
        """Toggle radio button to 'setup' mode"""
        self.toggle_radio(button="setup")

    def set_as_closeout(self):
        """Toggle radio button to 'closeout' mode"""
        self.toggle_radio(button="closeout")

    def toggle_radio(self, button: str):
        """Either changes the parent list of the currently-selected action, or if no action is selected then this
        updates which parent list a future new action will be assigned to"""

        # If the action is either not selected or is --Scan--, do nothing
        current_selection = self.get_action_list_and_index()
        if current_selection is None:
            return
        action_list, i, index = current_selection
        if action_list is None:
            return

        # Else, Move the selected action to the back of the other action list
        if action_list == self.actions_dict['setup'] and button == "closeout":
            source = self.actions_dict['setup']
            destination = self.actions_dict['closeout']
        elif action_list == self.actions_dict['closeout'] and button == "setup":
            source = self.actions_dict['closeout']
            destination = self.actions_dict['setup']
        else:
            return

        action = source[i]
        del source[i]
        destination.append(action)

        if index == i:
            new_position = len(self.actions_dict['setup']) + len(self.actions_dict['closeout'])
        else:
            new_position = len(self.actions_dict['setup']) - 1

        self.update_action_list(index=new_position)

    def initialize_action_control(self):
        if self.config_folder is None:
            logging.error("No defined path for save devices")
            return

        exp_name = self.config_folder.parent.name
        self.action_control = ActionControl(experiment_name=exp_name)
        self.ui.buttonEnableActions.setEnabled(False)
        self.ui.buttonPerformSetupActions.setEnabled(True)
        self.ui.buttonPerformPostscanActions.setEnabled(True)

    def perform_setup_actions(self):
        self.action_control.perform_action({'steps': self.actions_dict['setup']})

    def perform_postscan_actions(self):
        self.action_control.perform_action({'steps': self.actions_dict['closeout']})

    def save_element(self):
        """Save the current dictionaries as a new element in the experimental folder with the correct formatting"""
        if self.config_folder is None:
            logging.error("No defined path for save devices")
            return

        filename = self.ui.lineElementName.text().strip()
        if filename == "":
            logging.warning("Need an element name")
        else:
            file = self.config_folder / (filename + ".yaml")
            file.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving config to {file}")
            setup_action = {'steps': self.actions_dict['setup']}
            closeout_action = {'steps': self.actions_dict['closeout']}
            full_dictionary = {
                'Devices': self.devices_dict,
                'setup_action': setup_action,
                'closeout_action': closeout_action
            }
            if not setup_action['steps']:
                del full_dictionary['setup_action']
            if not closeout_action['steps']:
                del full_dictionary['closeout_action']
            logging.debug(full_dictionary)
            with open(file, 'w') as f:
                yaml.dump(full_dictionary, f, default_flow_style=False)

    def close_window(self):
        """Exits the window"""
        self.close()

    def open_element(self):
        """Prompts the user to select a .yaml file from which to load element dictionary information"""
        if self.config_folder is None:
            logging.error("No defined path for save devices")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select a YAML File", str(self.config_folder),
                                                   "YAML Files (*yaml)", options=options)
        if file_name:
            self.load_settings_from_file(Path(file_name))
            self.update_device_list()
            self.update_action_list()
            self.update_action_display()

    def load_settings_from_file(self, config_filename: Path):
        """Given a .yaml file, loads the dictionary and parses it to the backend dictionaries for the GUI"""
        with open(config_filename, 'r') as file:
            full_dictionary = yaml.safe_load(file)

        if 'Devices' in full_dictionary:
            self.devices_dict = full_dictionary['Devices']
        else:
            self.devices_dict = {}

        self.actions_dict = {
            'setup': [],
            'closeout': []
        }
        if 'setup_action' in full_dictionary:
            self.actions_dict['setup'] = full_dictionary['setup_action']['steps']
        if 'closeout_action' in full_dictionary:
            self.actions_dict['closeout'] = full_dictionary['closeout_action']['steps']

        self.ui.lineElementName.setText(config_filename.stem)
