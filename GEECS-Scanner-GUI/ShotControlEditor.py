"""
GUI that interfaces with the shot control configurations which allows the user to specify what device controls timing
and what variables in that device need to be changed to reflect the three states of the system:

OFF:  State when the system is between scan steps or in the process of starting/stopping a scan.
ON (Scan): State when taking a scan, to be used when saving is turned "ON" for selected devices
ON (Standby): State when the system is not actively recording data, but devices are still being triggered.

-Chris
"""
from __future__ import annotations

import yaml
from typing import Optional, Union
from pathlib import Path
from ShotControlEditor_ui import Ui_Dialog
from PyQt5.QtWidgets import QDialog, QCompleter, QInputDialog, QPushButton
from PyQt5.QtCore import pyqtSignal, QEvent, Qt


class ShotControlEditor(QDialog):
    selected_configuration = pyqtSignal(str)

    def __init__(self, config_folder_path: Union[str, Path], current_config: Optional[str] = None,
                 database_dict: Optional[dict] = None):

        super().__init__()

        # Dummy button to try and contain where pressing "Enter" goes, but TODO find a better solution
        self.dummyButton = QPushButton("", self)
        self.dummyButton.setDefault(True)
        self.dummyButton.setVisible(False)

        self.device_name = ''
        self.variable_dictionary = {}

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Dictionary containing all the devices and variables in the experiment
        self.database_dict = database_dict or {}

        # Folder containing all the shot control configurations for the selected experiment
        self.config_folder_path = Path(config_folder_path)

        # Top half of the GUI for selecting the configuration and some file operations
        self.ui.lineConfigurationSelect.setReadOnly(True)
        self.ui.lineConfigurationSelect.installEventFilter(self)
        self.ui.lineConfigurationSelect.editingFinished.connect(self.configuration_selected)

        self.configuration_name = current_config or ''
        self.ui.lineConfigurationSelect.setText(self.configuration_name)

        self.ui.buttonNewConfiguration.clicked.connect(self.create_new_configuration)
        # TODO self.ui.buttonCopyConfiguration.clicked.connect(self.copy_current_configuration)
        # TODO self.ui.buttonDeleteConfiguration.clicked.connect(self.delete_current_configuration)

        # Line edit to specify the device name
        self.ui.lineDeviceName.installEventFilter(self)
        # TODO self.ui.lineDeviceName.editingFinished.connect(self.update_device_name)

        # GUI elements to add, and remove a given variable from the list of variables for that device
        self.ui.lineVariableName.installEventFilter(self)
        # TODO self.ui.buttonAddVariable.clicked.connect(self.add_variable)
        # TODO self.ui.buttonRemoveVariable.clicked.connect(self.remove_variable)
        # TODO self.ui.listShotControlVariables.itemSelectionChanged.connect(self.update_states_info)

        # Line edits to enter in the values for the given variable in the three scan states
        # TODO self.ui.lineOffState.editingFinished.connect(self.update_variable_dictionary)
        # TODO self.ui.lineScanState.editingFinished.connect(self.update_variable_dictionary)
        # TODO self.ui.lineStandbyState.editingFinished.connect(self.update_variable_dictionary)

        # Buttons to save and close the dialog
        # TODO self.ui.buttonSaveConfiguration.clicked.connect(self.save_configuration)
        self.ui.buttonCloseWindow.clicked.connect(self.close_window)

        # If a valid current_config was given, load that information
        # TODO

    def eventFilter(self, source, event):
        """Creates a custom event for the text boxes so that the completion suggestions are shown when mouse is clicked
        """
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineConfigurationSelect:
            self.show_configuration_list()
            self.dummyButton.setDefault(True)
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineDeviceName:
            self.show_device_list()
            self.dummyButton.setDefault(True)
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineVariableName:
            self.show_variable_list()
            self.ui.buttonAddVariable.setDefault(True)
            return True
        return super().eventFilter(source, event)

    def _get_list_of_configurations(self) -> list[str]:
        return [f.stem for f in self.config_folder_path.iterdir() if f.suffix == ".yaml"]

    def show_configuration_list(self):
        """ Displays the found experiments in the ./experiments/ subfolder for selecting experiment """
        files = self._get_list_of_configurations()
        completer = QCompleter(files, self)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.ui.lineConfigurationSelect.setCompleter(completer)
        self.ui.lineConfigurationSelect.setFocus()
        completer.complete()

    def configuration_selected(self):
        entered_name = self.ui.lineConfigurationSelect.text()
        if entered_name in self._get_list_of_configurations():
            self.configuration_name = entered_name
        else:
            self.ui.lineConfigurationSelect.setText('')

        # TODO change the rest of the GUI to reflect this

    def create_new_configuration(self):
        text, ok = QInputDialog.getText(self, 'New Configuration', 'Enter nickname:')
        if ok and text:
            config_file = self.config_folder_path / (text + ".yaml")
            config_file.parent.mkdir(parents=True, exist_ok=True)

            contents = {}
            with open(config_file, 'w') as file:
                yaml.dump(contents, file, default_flow_style=False)

    def close_window(self):
        """Upon exiting the window, set the main window's timing configuration to the currently displayed config"""
        self.selected_configuration.emit(self.configuration_name)
        self.close()

    def closeEvent(self, event):
        """Upon exiting the window, set the main window's timing configuration to the currently displayed config"""
        self.selected_configuration.emit(self.configuration_name)
        event.accept()
