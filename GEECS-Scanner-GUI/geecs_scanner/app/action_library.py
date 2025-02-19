"""
GUI that can organize a list of presets into a multi-scan script.  This list can be separated into two lists to
independently set presets for the save device elements

-Chris
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional
if TYPE_CHECKING:
    from . import GEECSScannerWindow

import yaml
import copy
import logging
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QInputDialog, QMessageBox
from PyQt5.QtCore import QEvent
from .gui.ActionLibrary_ui import Ui_Form
from .lib.gui_utilities import (parse_variable_text, write_updated_file, display_completer_list,
                                display_completer_variable_list)
from .lib import ActionControl
from ..utils import multiscan_finish_jingle


def get_default_action() -> dict:
    return {'steps': []}


class ActionLibrary(QWidget):
    def __init__(self, main_window: GEECSScannerWindow, database_dict: dict, action_configurations_folder: Union[Path, str]):
        super().__init__()

        self.main_window = main_window
        self.database_dict = database_dict

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
            display_completer_list(self, location=self.ui.lineActionType, completer_list=ActionControl.list_of_actions)
            return True
        elif (event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionOption1
              and self.action_mode in ['set', 'get']):
            display_completer_list(self, location=self.ui.lineActionOption1,
                                   completer_list=sorted(self.database_dict.keys()))
            return True
        elif (event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionOption2
              and self.action_mode in ['set', 'get']):
            display_completer_variable_list(self, self.database_dict, list_location=self.ui.lineActionOption2,
                                            device_location=self.ui.lineActionOption1)
            return True
        elif (event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionOption1
              and self.action_mode in ['execute']):
            display_completer_list(self, location=self.ui.lineActionOption1, 
                                   completer_list=self.actions_data['actions'].keys())
            return True
        return super().eventFilter(source, event)

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
                write_updated_file(filename=self.actions_file, dictionary=actions_data_actual)
                logging.info(f"Removed action '{name}' from '{self.actions_file}'")
            else:
                logging.info(f"Removed action '{name}' from unsaved dictionary")

            # Also delete from the GUI's version of composite data (leaving other changes unsaved)
            del self.actions_data['actions'][name]

            self.populate_action_list()

    # # # # # # # # # # # GUI elements for interacting with the list of steps # # # # # # # # # # #

    def change_action_selection(self):
        self.update_action_list()
        self.update_action_display()

    def update_action_list(self, index: Optional[int] = None):
        """Updates the visible list of actions on the GUI according to the current state of the action dictionary"""
        self.ui.listActionSteps.clear()

        name = self.get_selected_name()
        if not name:
            return

        for item in self.actions_data['actions'][name]['steps']:
            self.ui.listActionSteps.addItem(ActionControl.generate_action_description(item))

        if index is not None and 0 <= index < self.ui.listActionSteps.count():
            self.ui.listActionSteps.setCurrentRow(index)

    def add_action(self):
        """Appends action to the end of either the setup or closeout action list based on the currently-selected
        radio button.  The action is specified by the action line edit."""
        text = self.ui.lineActionType.text().strip()
        name = self.get_selected_name()
        if text and name:
            self.actions_data['actions'][name]['steps'].append(ActionControl.get_new_action(text))
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
            write_updated_file(filename=self.actions_file, dictionary=self.actions_data)

    def discard_all_changes(self):
        reply = QMessageBox.question(self, "Discard Changes", f"Discard all unsaved changes?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.load_action_data()

    def closeEvent(self, event):
        self.main_window.exit_action_library()
        event.accept()
