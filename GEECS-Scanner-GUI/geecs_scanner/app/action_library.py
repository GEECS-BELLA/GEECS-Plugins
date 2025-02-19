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
import time
import logging
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QInputDialog, QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer, QObject, QThread, pyqtSignal, pyqtSlot
from .gui.ActionLibrary_ui import Ui_Form
from ..utils import multiscan_finish_jingle


class ActionLibrary(QWidget):
    def __init__(self, main_window: GEECSScannerWindow, action_configurations_folder: Union[Path, str]):
        super().__init__()

        self.main_window = main_window

        # Initializes the gui elements
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Action Library")

        self.actions_file = Path(action_configurations_folder) / 'actions.yaml'
        self.actions_data = {}
        self.load_action_list()

        # Functionality for close button
        self.ui.buttonCloseWindow.clicked.connect(self.close)

        # Set stylesheet to that of the main window
        self.setStyleSheet(main_window.styleSheet())

    def load_action_list(self):
        self.ui.listAvailableActions.clear()
        if not self.actions_file.exists():
            return

        with open(self.actions_file) as f:
            self.actions_data = yaml.safe_load(f)
        for action_name in self.actions_data['actions'].keys():
            self.ui.listAvailableActions.addItem(action_name)

    def closeEvent(self, event):
        self.main_window.exit_action_library()
        event.accept()
