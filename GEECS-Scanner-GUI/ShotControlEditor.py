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
from PyQt5.QtWidgets import QDialog, QCompleter, QInputDialog, QPushButton, QMessageBox
from PyQt5.QtCore import pyqtSignal, QEvent, Qt
# TODO change print statements to loggers


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
        self.ui.lineConfigurationSelect.textChanged.connect(self.configuration_selected)

        self.configuration_name = current_config or ''
        self.ui.lineConfigurationSelect.setText(self.configuration_name)

        self.ui.buttonNewConfiguration.clicked.connect(self.create_new_configuration)
        self.ui.buttonCopyConfiguration.clicked.connect(self.copy_current_configuration)
        self.ui.buttonDeleteConfiguration.clicked.connect(self.delete_current_configuration)

        # Line edit to specify the device name
        self.ui.lineDeviceName.installEventFilter(self)
        self.ui.lineDeviceName.textChanged.connect(self.update_device_name)

        # GUI elements to add, and remove a given variable from the list of variables for that device
        self.ui.lineVariableName.installEventFilter(self)
        self.ui.buttonAddVariable.clicked.connect(self.add_variable)
        self.ui.buttonRemoveVariable.clicked.connect(self.remove_variable)
        self.ui.listShotControlVariables.itemSelectionChanged.connect(self.show_states_info)

        # Line edits to enter in values for the given variable in the three scan states
        self.ui.lineOffState.editingFinished.connect(self.update_variable_dictionary)
        self.ui.lineScanState.editingFinished.connect(self.update_variable_dictionary)
        self.ui.lineStandbyState.editingFinished.connect(self.update_variable_dictionary)

        # Buttons to save and close the dialog
        self.ui.buttonSaveConfiguration.clicked.connect(self.save_configuration)
        self.ui.buttonCloseWindow.clicked.connect(self.close_window)

        # If a valid current_config was given, load that information
        self.configuration_selected()

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
        if source == self.ui.lineVariableName and event.type() == QEvent.MouseButtonPress:
            self.show_variable_list()
            self.ui.buttonAddVariable.setDefault(True)
            return True
        return super().eventFilter(source, event)

    # # # # Methods for selecting, creating, and deleting available timing system configurations # # # # #

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
        is_valid = bool(entered_name in self._get_list_of_configurations())
        if is_valid:
            self.configuration_name = entered_name
            config_file = self.config_folder_path / (self.configuration_name + ".yaml")
            with open(config_file, 'r') as file:
                settings = yaml.safe_load(file)
        else:
            self.ui.lineConfigurationSelect.setText('')
            settings = {}

        self.device_name = settings.get('device', '')
        self.variable_dictionary = settings.get('variables', {})

        self.ui.lineDeviceName.setText(self.device_name)
        self.ui.lineVariableName.setText('')
        self._update_variable_list()
        self.show_states_info()

        self.ui.lineDeviceName.setEnabled(is_valid)
        self.ui.lineVariableName.setEnabled(is_valid)
        self.ui.buttonAddVariable.setEnabled(is_valid)
        self.ui.buttonRemoveVariable.setEnabled(is_valid)
        self.ui.buttonCopyConfiguration.setEnabled(is_valid)
        self.ui.buttonDeleteConfiguration.setEnabled(is_valid)

    def create_new_configuration(self):
        text, ok = QInputDialog.getText(self, 'New Configuration', 'Enter nickname:')
        if ok and text:
            self._write_configuration_file(configuration_name=text, use_empty=True)

    def copy_current_configuration(self):
        text, ok = QInputDialog.getText(self, 'Copy Configuration', f"Enter nickname for new copy of '{self.configuration_name}'")
        if ok and text:
            self._write_configuration_file(configuration_name=text)

    def _write_configuration_file(self, configuration_name: str, use_empty: bool = False):
        if use_empty:
            contents = {}
        else:
            contents = {'device': self.device_name, 'variables': self.variable_dictionary}

        config_file = self.config_folder_path / (configuration_name + ".yaml")
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as file:
            yaml.dump(contents, file, default_flow_style=False)

        self.ui.lineConfigurationSelect.setText(configuration_name)
        self.configuration_selected()

    def delete_current_configuration(self):
        configuration_name = self.ui.lineConfigurationSelect.text()
        if configuration_name in self._get_list_of_configurations():
            reply = QMessageBox.question(self, 'Confirm Delete', f'Delete configuration "{configuration_name}"?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                config_file = self.config_folder_path / (configuration_name + ".yaml")
                if config_file.exists() and config_file.is_file():
                    try:
                        config_file.unlink()
                        print(f"'{configuration_name}' deleted")
                        self.ui.lineConfigurationSelect.setText('')
                        self.configuration_selected()
                    except Exception as e:
                        print(f"Error occurred while deleting '{configuration_name}': {e}")
                else:
                    print(f"Error occurred: '{configuration_name}' not located")

    # # # # Methods for setting device and variable names, and adding/removing to the variable list # # # #

    def show_device_list(self):
        if self.ui.lineDeviceName.isEnabled():
            self.ui.lineDeviceName.selectAll()
            completer = QCompleter(sorted(self.database_dict.keys()), self)
            completer.setCompletionMode(QCompleter.PopupCompletion)
            completer.setCaseSensitivity(Qt.CaseSensitive)

            self.ui.lineDeviceName.setCompleter(completer)
            self.ui.lineDeviceName.setFocus()
            completer.complete()

    def update_device_name(self):
        self.device_name = self.ui.lineDeviceName.text()

    def show_variable_list(self):
        if self.ui.lineVariableName.isEnabled() and self.device_name in self.database_dict.keys():
            self.ui.lineVariableName.selectAll()
            completer = QCompleter(sorted(self.database_dict[self.device_name].keys()), self)
            completer.setCompletionMode(QCompleter.PopupCompletion)
            completer.setCaseSensitivity(Qt.CaseSensitive)

            self.ui.lineVariableName.setCompleter(completer)
            self.ui.lineVariableName.setFocus()
            completer.complete()

    def _update_variable_list(self):
        self.ui.listShotControlVariables.clear()
        for var in self.variable_dictionary.keys():
            self.ui.listShotControlVariables.addItem(var)

    def add_variable(self):
        new_variable = self.ui.lineVariableName.text()
        if new_variable is None or new_variable.strip() == '':
            return
        if new_variable not in self.variable_dictionary.keys():
            new_states = {'OFF': '', 'SCAN': '', 'STANDBY': ''}
            self.variable_dictionary[new_variable] = new_states

            self._update_variable_list()
            self.ui.lineVariableName.setText('')

    def remove_variable(self):
        selected_variable = self.ui.listShotControlVariables.selectedItems()
        if not selected_variable:
            return
        for selection in selected_variable:
            text = selection.text()
            if text in self.variable_dictionary.keys():
                del self.variable_dictionary[text]
        self._update_variable_list()

    # # # # Methods for interacting with the state values and updating the configuration dictionary # # # #

    def show_states_info(self):
        selected_variable = self.ui.listShotControlVariables.selectedItems()
        has_selection = bool(selected_variable)
        self.ui.lineOffState.setEnabled(has_selection)
        self.ui.lineScanState.setEnabled(has_selection)
        self.ui.lineStandbyState.setEnabled(has_selection)

        if has_selection:
            for selection in selected_variable:
                variable = self.variable_dictionary[selection.text()]
                self.ui.lineOffState.setText(variable['OFF'])
                self.ui.lineScanState.setText(variable['SCAN'])
                self.ui.lineStandbyState.setText(variable['STANDBY'])
        else:
            self.ui.lineOffState.setText('')
            self.ui.lineScanState.setText('')
            self.ui.lineStandbyState.setText('')

    def update_variable_dictionary(self):
        selected_variable = self.ui.listShotControlVariables.selectedItems()
        if not selected_variable:
            return
        for selection in selected_variable:
            variable = self.variable_dictionary[selection.text()]
            variable['OFF'] = self.ui.lineOffState.text()
            variable['SCAN'] = self.ui.lineScanState.text()
            variable['STANDBY'] = self.ui.lineStandbyState.text()

    # # # # Methods for saving current configuration and closing the window # # # #

    def save_configuration(self):
        configuration_name = self.ui.lineConfigurationSelect.text()
        if configuration_name is None or configuration_name.strip() == '':
            print("Error: no configuration specified")
            return
        if self.device_name == '':
            print("Error: cannot save with empty device name")
            return
        if len(self.variable_dictionary) == 0:
            print("Error: cannot save with no variables")
            return
        for variable_name in self.variable_dictionary.keys():
            keys_to_check = ['OFF', 'SCAN', 'STANDBY']
            variable = self.variable_dictionary[variable_name]
            if not all(key in variable and variable[key] is not None and variable[key] != '' for key in keys_to_check):
                print("Error: missing information for variables")
                return

        self._write_configuration_file(configuration_name=configuration_name)

    def close_window(self):
        """Upon exiting the window, set the main window's timing configuration to the currently displayed config"""
        self.selected_configuration.emit(self.configuration_name)
        self.close()

    def closeEvent(self, event):
        """Upon exiting the window, set the main window's timing configuration to the currently displayed config"""
        self.selected_configuration.emit(self.configuration_name)
        event.accept()
