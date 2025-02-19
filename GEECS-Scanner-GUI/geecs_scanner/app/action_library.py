"""
GUI that can organize a list of presets into a multi-scan script.  This list can be separated into two lists to
independently set presets for the save device elements

-Chris
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional
if TYPE_CHECKING:
    from . import GEECSScannerWindow
    from PyQt5.QtWidgets import QListWidget, QListWidgetItem

import yaml
import copy
import time
import logging
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QInputDialog, QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer, QObject, QThread, pyqtSignal, pyqtSlot
from .gui.ActionLibrary_ui import Ui_Form
from ..utils import multiscan_finish_jingle


def get_default_action() -> dict:
    return {'steps': []}


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
        self.populate_action_list()

        # Functionality to New, Copy, and Delete Buttons
        self.ui.buttonNewAction.clicked.connect(self.create_new_action)
        self.ui.buttonCopyAction.clicked.connect(self.copy_action)
        self.ui.buttonDeleteAction.clicked.connect(self.delete_selected_action)

        # Functionality for close button
        self.ui.buttonCloseWindow.clicked.connect(self.close)

        # Set stylesheet to that of the main window
        self.setStyleSheet(main_window.styleSheet())

    def load_action_data(self) -> dict:
        self.actions_data = {}
        if not self.actions_file.exists():
            return {}
        with open(self.actions_file) as f:
            self.actions_data = yaml.safe_load(f)
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

    @staticmethod
    def _write_updated_file(filename: Path, dictionary: dict):  # TODO this is a copy of scan variable editor...do better?
        """ Write the given dictionary to the given yaml file, used for either the 1d or composite scan variables

        :param filename: yaml filename
        :param dictionary: complete dictionary to be written
        """
        with open(filename, 'w') as f:
            yaml.dump(dictionary, f, default_flow_style=False)

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

        reply = QMessageBox.question(self, "Delete Action", f"Delete Action'{name}' from file?",
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

    def closeEvent(self, event):
        self.main_window.exit_action_library()
        event.accept()
