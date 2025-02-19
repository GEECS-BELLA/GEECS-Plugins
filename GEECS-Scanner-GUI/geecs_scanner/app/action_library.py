"""
GUI that can organize a list of presets into a multi-scan script.  This list can be separated into two lists to
independently set presets for the save device elements

-Chris
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional
if TYPE_CHECKING:
    from . import GEECSScannerWindow
    from PyQt5.QtWidgets import QLineEdit, QListWidget, QListWidgetItem

import yaml
import copy
import time
import logging
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QInputDialog, QPushButton, QMessageBox, QCompleter
from PyQt5.QtCore import QEvent, Qt, QTimer, QObject, QThread, pyqtSignal, pyqtSlot
from .gui.ActionLibrary_ui import Ui_Form
from ..utils import multiscan_finish_jingle


def get_default_action() -> dict:
    return {'steps': []}


def get_new_action(action) -> Union[None, dict[str, str]]:  # TODO combine with save element version
    """
    Translates a given action keyword to a default dictionary that is populated into the action list.

    :param action: action keyword
    :return: default dictionary for the associated action
    """
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


def parse_variable_text(text) -> Union[int, float, str]:  # TODO combine with save_element_editor.py version
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


# List of available actions, to be used by the completer for the add action line edit
list_of_actions = [  # TODO combine with save_element_editor.py version
    'set',
    'get',
    'wait',
    'execute',
]


class ActionLibrary(QWidget):
    def __init__(self, main_window: GEECSScannerWindow, database_dict: dict, action_configurations_folder: Union[Path, str]):
        super().__init__()

        self.main_window = main_window
        self.database_dict = database_dict

        self.dummyButton = QPushButton("", self)
        self.dummyButton.setDefault(True)
        self.dummyButton.setVisible(False)

        # Initializes the gui elements
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Action Library")

        self.actions_file = Path(action_configurations_folder) / 'actions.yaml'
        self.actions_data = self.load_action_data()

        # Functionality to New, Copy, and Delete Buttons
        self.ui.listAvailableActions.itemSelectionChanged.connect(self.change_action_selection)
        self.ui.buttonNewAction.clicked.connect(self.create_new_action)
        self.ui.buttonCopyAction.clicked.connect(self.copy_action)
        self.ui.buttonDeleteAction.clicked.connect(self.delete_selected_action)

        # Functionality to add and remove steps
        self.ui.lineActionType.installEventFilter(self)

        self.ui.buttonAddStep.clicked.connect(self.add_action)
        self.ui.buttonRemoveStep.clicked.connect(self.remove_action)

        self.ui.listActionSteps.clicked.connect(self.update_action_display)
        self.action_mode = ''
        self.ui.lineActionOption1.installEventFilter(self)
        self.ui.lineActionOption2.installEventFilter(self)
        self.ui.lineActionOption1.editingFinished.connect(self.update_action_info)
        self.ui.lineActionOption2.editingFinished.connect(self.update_action_info)
        self.ui.lineActionOption3.editingFinished.connect(self.update_action_info)
        self.update_action_display()

        self.ui.buttonMoveSooner.clicked.connect(self.move_action_sooner)
        self.ui.buttonMoveLater.clicked.connect(self.move_action_later)

        # Functionality to Save All and Revert All buttons
        self.ui.buttonSaveAll.clicked.connect(self.save_all_changes)
        self.ui.buttonRevertAll.clicked.connect(self.discard_all_changes)

        # Functionality for close button
        self.ui.buttonCloseWindow.clicked.connect(self.close)

        # Set stylesheet to that of the main window
        self.setStyleSheet(main_window.styleSheet())

    def eventFilter(self, source, event):
        """Creates a custom event for the text boxes so that the completion suggestions are shown when mouse is clicked
        """
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionType:
            self.display_completer_list(location=self.ui.lineActionType,
                                        completer_list=list_of_actions)
            return True
        elif (event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionOption1
              and self.action_mode in ['set', 'get']):
            self.display_completer_list(location=self.ui.lineActionOption1,
                                        completer_list=sorted(self.database_dict.keys()))
            self.dummyButton.setDefault(True)
            return True
        elif (event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionOption2
              and self.action_mode in ['set', 'get']):
            self.display_completer_variable_list(list_location=self.ui.lineActionOption2,
                                                 device_location=self.ui.lineActionOption1)
            self.dummyButton.setDefault(True)
            return True
        return super().eventFilter(source, event)

    def display_completer_list(self, location: QLineEdit, completer_list: list[str]):  # TODO combine with scan var version
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

    def display_completer_variable_list(self, list_location: QLineEdit, device_location: QLineEdit):  # TODO combine with scan var version
        """ Displays list of variables at one location using the device name at another location

        :param list_location: GUI element at which to show the completer list
        :param device_location: GUI element where the device name is given
        """
        device_name = device_location.text().strip()
        if device_name in self.database_dict:
            variable_list = sorted(self.database_dict[device_name].keys())
            self.display_completer_list(location=list_location, completer_list=variable_list)

    @staticmethod
    def _write_updated_file(filename: Path, dictionary: dict):  # TODO this is a copy of scan variable editor...do better?
        """ Write the given dictionary to the given yaml file, used for either the 1d or composite scan variables

        :param filename: yaml filename
        :param dictionary: complete dictionary to be written
        """
        with open(filename, 'w') as f:
            yaml.dump(dictionary, f, default_flow_style=False)

    # # # # # # # # # # # GUI elements for using list of actions and saving/deleting actions # # # # # # # # # # #

    def load_action_data(self) -> dict:
        self.actions_data = {}
        if self.actions_file.exists():
            with open(self.actions_file) as f:
                self.actions_data = yaml.safe_load(f)
        self.populate_action_list()
        return self.actions_data

    def populate_action_list(self):
        self.ui.listAvailableActions.clear()
        for action_name in self.actions_data['actions'].keys():
            self.ui.listAvailableActions.addItem(action_name)

    def get_selected_name(self) -> Optional[str]:
        selected_action = self.ui.listAvailableActions.selectedItems()
        if not selected_action:
            return None
        for selection in selected_action:
            name = selection.text().strip()
            if name in self.actions_data['actions']:
                return name

    def _prompt_new_action(self, message: str, copy_base: Optional[str] = None):
        text, ok = QInputDialog.getText(self, 'New Action', message)
        if ok and text:
            name = str(text).strip()

            if name in self.actions_data['actions']:
                logging.warning(f"'{name}' already exists in dict, cannot create")
                return

            if copy_base:
                if copy_base not in self.actions_data['actions']:
                    logging.warning(f"'{copy_base}' does not exist in dict, cannot create copy")
                    return
                new_action = copy.deepcopy(self.actions_data['actions'][copy_base])
            else:
                new_action = get_default_action()

            self.actions_data['actions'][name] = new_action
            logging.info(f"New action '{name}' created.  Not yet saved.")

            self.populate_action_list()

    def create_new_action(self):
        """ Creates a new entry in composite_variables.yaml file with a user-specified name """
        self._prompt_new_action(message="Enter name for new action:")

    def copy_action(self):
        copy_base = self.get_selected_name()
        if copy_base not in self.actions_data['actions']:
            return

        self._prompt_new_action(message=f"Enter name for copy of '{copy_base}':", copy_base=copy_base)

    def delete_selected_action(self):
        name = self.get_selected_name()
        if not name or name not in self.actions_data['actions']:
            logging.warning(f"No valid action to delete.")
            return

        reply = QMessageBox.question(self, "Delete Action", f"Delete action '{name}' from file?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Read current version of file and delete only the specified element if it exists
            with open(self.actions_file, 'r') as file:
                actions_data_actual = yaml.safe_load(file)

            if name in actions_data_actual['actions']:
                del actions_data_actual['actions'][name]
                self._write_updated_file(filename=self.actions_file, dictionary=actions_data_actual)
                logging.info(f"Removed action '{name}' from '{self.actions_file}'")
            else:
                logging.info(f"Removed action '{name}' from unsaved dictionary")

            # Also delete from the GUI's version of composite data (leaving other changes unsaved)
            del self.actions_data['actions'][name]

            self.populate_action_list()

    # # # # # # # # # # # GUI elements for interacting with the list of steps # # # # # # # # # # #

    @staticmethod
    def generate_action_description(action: dict[str, list]) -> str:  # TODO combine with save element version
        """For each action in the list, generate a string that displays all the information for that action step"""
        description = "???"
        if action.get("wait") is not None:
            description = f"wait {action['wait']}"
        elif action['action'] == 'execute':
            description = f"execute {action['action_name']}"
        elif action['action'] == 'set':
            description = f"{action['action']} {action['device']}:{action['variable']} {action.get('value')}"
        elif action['action'] == 'get':
            description = f"{action['action']} {action['device']}:{action['variable']} {action.get('expected_value')}"
        return description

    def change_action_selection(self):
        self.update_action_list()
        self.update_action_display()

    def update_action_list(self, index: Optional[int] = None):
        """Updates the visible list of actions on the GUI according to the current state of the action dictionary"""
        self.ui.listActionSteps.clear()
        self.dummyButton.setDefault(True)

        name = self.get_selected_name()
        if not name:
            return

        for item in self.actions_data['actions'][name]['steps']:
            self.ui.listActionSteps.addItem(self.generate_action_description(item))

        if index is not None and 0 <= index < self.ui.listActionSteps.count():
            self.ui.listActionSteps.setCurrentRow(index)

    def add_action(self):
        """Appends action to the end of either the setup or closeout action list based on the currently-selected
        radio button.  The action is specified by the action line edit."""
        text = self.ui.lineActionType.text().strip()
        name = self.get_selected_name()
        if text and name:
            self.actions_data['actions'][name]['steps'].append(get_new_action(text))
            self.ui.lineActionType.clear()
            self.change_action_selection()

    def remove_action(self):
        """Removes the currently-selected action from the list of actions"""
        index = self.get_selected_step_index()
        name = self.get_selected_name()
        if index >= 0 and name:
            del self.actions_data['actions'][name]['steps'][index]

        self.change_action_selection()

    def update_action_display(self):
        """Updates the visible list of information based on the currently-selected action"""
        index = self.get_selected_step_index()
        name = self.get_selected_name()

        self.action_mode = ''
        self.ui.labelActionOption1.setText("")
        self.ui.labelActionOption2.setText("")
        self.ui.labelActionOption3.setText("")
        self.ui.lineActionOption1.setText("")
        self.ui.lineActionOption2.setText("")
        self.ui.lineActionOption3.setText("")
        self.ui.lineActionOption1.setEnabled(False)
        self.ui.lineActionOption2.setEnabled(False)
        self.ui.lineActionOption3.setEnabled(False)

        if index >= 0 and name:
            action = self.actions_data['actions'][name]['steps'][index]
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
        index = self.get_selected_step_index()
        name = self.get_selected_name()
        if index >= 0 and name:
            action = self.actions_data['actions'][name]['steps'][index]

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

            self.update_action_list(index=index)

    def get_selected_step_index(self) -> int:
        """Finds the location of the currently-selected action in the visible list on the GUI.

        :return: If no selection, returns -1.  Otherwise, return the index of the currently selected step
        """
        selected_action = self.ui.listActionSteps.selectedItems()
        if not selected_action:
            return -1
        for action in selected_action:
            index = self.ui.listActionSteps.row(action)
            return index

    def move_action_sooner(self):
        """Moves the selected action to an earlier position in the same list"""
        i = self.get_selected_step_index()
        name = self.get_selected_name()
        if i >= 0 and name:
            action_list = self.actions_data['actions'][name]['steps']
            if 0 < i < len(action_list):
                action_list[i], action_list[i - 1] = action_list[i - 1], action_list[i]
                i -= 1
                self.update_action_list(index=i)

    def move_action_later(self):
        """Moves the selected action to a later position in the same list"""
        i = self.get_selected_step_index()
        name = self.get_selected_name()
        if i >= 0 and name:
            action_list = self.actions_data['actions'][name]['steps']
            if 0 <= i < len(action_list) - 1:
                action_list[i], action_list[i + 1] = action_list[i + 1], action_list[i]
                i += 1
                self.update_action_list(index=i)

    # # # # # # # # # # # GUI elements for Execute, Save, Revert, and Close buttons # # # # # # # # # # #

    def save_all_changes(self):
        reply = QMessageBox.question(self, "Save Actions", f"Save all changes to {self.actions_file.name}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._write_updated_file(filename=self.actions_file, dictionary=self.actions_data)

    def discard_all_changes(self):
        reply = QMessageBox.question(self, "Discard Changes", f"Discard all unsaved changes?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.load_action_data()

    def closeEvent(self, event):
        self.main_window.exit_action_library()
        event.accept()
