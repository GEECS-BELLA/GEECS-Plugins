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
from PyQt5.QtWidgets import QWidget, QInputDialog, QFileDialog, QMessageBox, QCompleter
from PyQt5.QtCore import QEvent, Qt, QTimer, QObject, QThread, pyqtSignal, pyqtSlot
from .gui.ActionLibrary_ui import Ui_Form
from ..utils import multiscan_finish_jingle


def get_default_action() -> dict:
    return {'steps': []}


# List of available actions, to be used by the completer for the add action line edit
list_of_actions = [  # TODO combine with save_element_editor.py version
    'set',
    'get',
    'wait',
    'execute',
]


class ActionLibrary(QWidget):
    def __init__(self, main_window: GEECSScannerWindow, action_configurations_folder: Union[Path, str]):
        super().__init__()

        self.main_window = main_window

        # Initializes the gui elements
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Action Library")

        self.actions_file = Path(action_configurations_folder) / 'actions.yaml'
        self.actions_data = self.load_action_data()

        # Functionality to New, Copy, and Delete Buttons
        self.ui.buttonNewAction.clicked.connect(self.create_new_action)
        self.ui.buttonCopyAction.clicked.connect(self.copy_action)
        self.ui.buttonDeleteAction.clicked.connect(self.delete_selected_action)

        # Functionality to add and remove steps
        self.ui.lineActionType.installEventFilter(self)

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
            self.show_action_type_list()
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

    def show_action_type_list(self):
        self.display_completer_list(self.ui.lineActionType, list_of_actions)

    # # # # # # # # # # # GUI elements for Execute, Save, Revert, and Close buttons # # # # # # # # # # #

    def save_all_changes(self):
        reply = QMessageBox.question(self, "Save Actions", f"Save all changes to {self.actions_file.name}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._write_updated_file(filename=self.actions_file, dictionary=self.actions_data)
            self.load_action_data()

    def discard_all_changes(self):
        reply = QMessageBox.question(self, "Discard Changes", f"Discard all unsaved changes?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.load_action_data()

    def closeEvent(self, event):
        self.main_window.exit_action_library()
        event.accept()
