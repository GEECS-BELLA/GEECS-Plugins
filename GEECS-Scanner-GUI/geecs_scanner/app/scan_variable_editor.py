"""
Side dialog GUI that allows for the user to view and edit all 1D scan variables and composite variables.  The 1D scan
variables are in the top 1/3 of the window and is a simple mapping of a nickname to a device+variable combination.  The
composite variables are a bit more complex, and are saved in a different yaml file.  Currently composite variables that
are relative and absolute are supported, a "get-only" type of composite variable is available to specify but will not
work with scan_manager yet.

-Chris
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from . import GEECSScannerWindow

import yaml
import logging
import copy
from pathlib import Path

from PyQt5.QtWidgets import QDialog, QMessageBox, QInputDialog
from PyQt5.QtCore import QEvent

from .gui.ScanDeviceEditor_ui import Ui_Dialog
from .lib.gui_utilities import write_dict_to_yaml_file, display_completer_list, display_completer_variable_list


def default_composite_variable():
    """
    :return: a blank composite variable with the correct formatting, empty composite list and an unspecified mode
    """
    return {'components': [], 'mode': ""}


def list_of_modes() -> list[str]:
    """
    :return: list of available types of modes
    """
    return ['relative', 'absolute', 'get-only']  # TODO Need to better implement the intended `get-only` features


# Below is the text that is displayed when a mouse hovers over the "tips" button
relation_tool_tip = ("The numerical result of this expression is what the device variable is set to.\n"
                     "Use 'composite_var' for the composite variable.\n"
                     "Plan on scanning a 'relative' composite variable from -1 to +1.\n"
                     "'numpy' or 'np' not allowed, use math expressions (like 'sqrt').")


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

        # Initialize dictionaries for the scan variables and composite variables
        self.scan_variable_data = {}
        self.scan_composite_data = {}
        self.update_variable_information_from_files()

        # Functionality to Scan Variables section
        self.ui.lineVariableNickname.installEventFilter(self)
        self.ui.lineVariableDevice.installEventFilter(self)
        self.ui.lineVariableVariable.installEventFilter(self)

        self.ui.lineVariableNickname.textChanged.connect(self.check_variable_nickname)

        self.ui.buttonVariableSave.clicked.connect(self.save_scan_variable)
        self.ui.buttonVariableDelete.clicked.connect(self.delete_scan_variable)

        # Functionality to Composite Variables section
        self.ui.lineCompositeNickname.installEventFilter(self)
        self.ui.lineCompositeDevice.installEventFilter(self)
        self.ui.lineCompositeVariable.installEventFilter(self)

        self.ui.lineCompositeNickname.textChanged.connect(self.update_visible_composite_information)

        self.ui.buttonCompositeNew.clicked.connect(self.create_new_composite_variable)
        self.ui.buttonCompositeCopy.clicked.connect(self.copy_composite_variable)
        self.ui.buttonCompositeDelete.clicked.connect(self.delete_composite_variable)

        self.ui.lineCompositeMode.setReadOnly(True)
        self.ui.lineCompositeMode.installEventFilter(self)
        self.ui.lineCompositeMode.textChanged.connect(self.update_composite_mode)

        self.ui.buttonCompositeVarAdd.clicked.connect(self.add_composite_component)
        self.ui.buttonCompositeVarRemove.clicked.connect(self.remove_selected_component)
        self.ui.buttonCompositeSave.clicked.connect(self.save_composite_variables_file)

        self.ui.listCompositeComponents.itemSelectionChanged.connect(self.update_visible_relation_information)
        self.ui.lineCompositeRelation.editingFinished.connect(self.update_relation_data)
        self.ui.toolbuttonRelationTips.setToolTip(relation_tool_tip)
        self.ui.toolbuttonRelationTips.clicked.connect(self.show_tool_tip)

        self.update_visible_composite_information()
        self.update_visible_relation_information()

        # Button to close out of this dialog window
        self.ui.buttonClose.clicked.connect(self.close)

        # Apply the stylesheet of the main window
        self.setStyleSheet(main_window.styleSheet())

        # Initial state of child dialog window
        self.variable_order = None

    # # # # #  Utility methods for the whole GUI  # # # # #

    def eventFilter(self, source, event):
        """ Custom event for the text boxes so that the completion suggestions are shown when mouse is clicked """
        # Nickname completer prompts
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineVariableNickname:
            display_completer_list(self, location=self.ui.lineVariableNickname,
                                   completer_list=self.get_scan_variable_list())
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineCompositeNickname:
            display_completer_list(self, location=self.ui.lineCompositeNickname,
                                   completer_list=self.get_scan_composite_list())
            return True

        # Device name completer prompts
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineVariableDevice:
            display_completer_list(self, location=self.ui.lineVariableDevice,
                                   completer_list=sorted(self.database_dict.keys()))
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineCompositeDevice:
            display_completer_list(self, location=self.ui.lineCompositeDevice,
                                   completer_list=sorted(self.database_dict.keys()))
            return True

        # Variable name completer prompts
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineVariableVariable:
            display_completer_variable_list(self, database_dict=self.database_dict,
                                            list_location=self.ui.lineVariableVariable,
                                            device_location=self.ui.lineVariableDevice)
            return True
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineCompositeVariable:
            display_completer_variable_list(self, database_dict=self.database_dict,
                                            list_location=self.ui.lineCompositeVariable,
                                            device_location=self.ui.lineCompositeDevice)
            return True

        # Other completer prompts
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineCompositeMode and self.ui.lineCompositeMode.isEnabled():
            display_completer_list(self, location=self.ui.lineCompositeMode,
                                   completer_list=list_of_modes())
            return True

        return super().eventFilter(source, event)

    def update_variable_information_from_files(self):
        """ Loads the data from the two yaml files and populates the lists of nicknames """
        self.scan_variable_data = {}
        self.scan_composite_data = {}

        try:
            with open(self.file_variables, 'r') as file:
                self.scan_variable_data = yaml.safe_load(file)
            with open(self.file_composite, 'r') as file:
                self.scan_composite_data = yaml.safe_load(file)
        except FileNotFoundError as e:
            logging.error(f"Error loading file: {e}")

    def get_scan_variable_list(self) -> list[str]:
        """
        :return: list of nicknames for all 1D scan variables
        """
        return list(self.scan_variable_data['single_scan_devices'].keys())

    def get_scan_composite_list(self) -> list[str]:
        """
        :return: list of nicknames for all composite variables
        """
        return list(self.scan_composite_data['composite_variables'].keys())

    # # # # #  Functionality to the Scan Variables Section  # # # # #

    def check_variable_nickname(self):
        """ When nickname is entered, check if it exists in the variable list and load device and variable info """
        nickname = self.ui.lineVariableNickname.text().strip()
        if nickname in self.get_scan_variable_list():
            device_variable = self.scan_variable_data['single_scan_devices'][nickname].split(":")
            self.ui.lineVariableDevice.setText(device_variable[0])
            self.ui.lineVariableVariable.setText(device_variable[1])

    def save_scan_variable(self):
        """ Updates or appends the currently-displayed scan variable to the config .yaml file """
        nickname = self.ui.lineVariableNickname.text().strip()
        device = self.ui.lineVariableDevice.text().strip()
        variable = self.ui.lineVariableVariable.text().strip()

        if nickname == "" or device == "" or variable == "":
            logging.warning("Incomplete scan variable information, cannot save to file")
            return

        self.scan_variable_data['single_scan_devices'][nickname] = f"{device}:{variable}"
        write_dict_to_yaml_file(filename=self.file_variables, dictionary=self.scan_variable_data)
        logging.info(f"Wrote variable '{nickname}' to '{self.file_variables}'")

        self.update_variable_information_from_files()

    def delete_scan_variable(self):
        """ Deletes the currently-displayed scan variable from the config .yaml file and clears the line edits """
        nickname = self.ui.lineVariableNickname.text().strip()
        if nickname not in self.scan_variable_data['single_scan_devices']:
            logging.warning(f"Variable {nickname} not in dict, cannot delete")
            return

        reply = QMessageBox.question(self, "Delete Scan Variable", f"Delete Scan Variable '{nickname}' from list?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.scan_variable_data['single_scan_devices'][nickname]
            write_dict_to_yaml_file(filename=self.file_variables, dictionary=self.scan_variable_data)
            logging.info(f"Removed variable '{nickname}' from '{self.file_variables}'")

            self.ui.lineVariableNickname.setText("")
            self.ui.lineVariableDevice.setText("")
            self.ui.lineVariableVariable.setText("")

            self.update_variable_information_from_files()

    # # # # #  Functionality to the Composite Variables section  # # # # #

    def update_visible_composite_information(self):
        """ Updates the GUI with the information of the currently-specified composite variable """
        self.ui.listCompositeComponents.clear()
        name = self.ui.lineCompositeNickname.text().strip()
        if not name or name not in self.scan_composite_data['composite_variables']:
            self.ui.lineCompositeMode.setText("")
            self.ui.lineCompositeMode.setEnabled(False)
        else:
            composite_var = self.scan_composite_data['composite_variables'][name]
            for var_dict in composite_var['components']:
                self.ui.listCompositeComponents.addItem(f"{var_dict['device']}:{var_dict['variable']}")
            mode = composite_var.get('mode', "")
            self.ui.lineCompositeMode.setEnabled(True)
            self.ui.lineCompositeMode.setText(mode)

    def get_component_index(self, device: str, variable: str) -> Optional[int]:
        """
        Finds the list index of a composite variable component given its device and variable names

        :param device: device name of composite variable component
        :param variable: variable name of composite variable component
        :return: the index of the component in the composite variable's list, otherwise None if it doesn't exist
        """
        name = self.ui.lineCompositeNickname.text().strip()
        if not name or name not in self.scan_composite_data['composite_variables']:
            return None

        component_list = self.scan_composite_data['composite_variables'][name]['components']
        for i in range(len(component_list)):
            component = component_list[i]
            if component['device'] == device and component['variable'] == variable:
                return i

        return None

    def add_composite_component(self):
        """ Adds a valid device+variable pair to the selected composite variable if it doens't already exist """
        name = self.ui.lineCompositeNickname.text().strip()
        if not name or name not in self.scan_composite_data['composite_variables']:
            return
        device = self.ui.lineCompositeDevice.text().strip()
        variable = self.ui.lineCompositeVariable.text().strip()
        if not device or not variable:
            return

        if self.get_component_index(device, variable):
            print("Already in list")
            return

        new_component = {'device': device, 'variable': variable, 'relation': ""}
        self.scan_composite_data['composite_variables'][name]['components'].append(new_component)

        self.ui.lineCompositeDevice.setText("")
        self.ui.lineCompositeVariable.setText("")
        self.update_visible_composite_information()

    def remove_selected_component(self):
        """ Removes the selected component from the composite variable """
        name = self.ui.lineCompositeNickname.text().strip()
        if not name or name not in self.scan_composite_data['composite_variables']:
            return

        if component := self.get_selected_composite_component():
            index = self.get_component_index(component['device'], component['variable'])
            del self.scan_composite_data['composite_variables'][name]['components'][index]
            self.update_visible_composite_information()

    def get_selected_composite_component(self) -> Optional[dict]:
        """
        :return: the dict component that corresponds to the currently-selected list widget row
        """
        if selected_variable := self.ui.listCompositeComponents.selectedItems():
            device, variable = selected_variable[0].text().split(":")
            name = self.ui.lineCompositeNickname.text().strip()
            for device_variable in self.scan_composite_data['composite_variables'][name]['components']:
                if device_variable['device'] == device and device_variable['variable'] == variable:
                    return device_variable
            return None

    def update_visible_relation_information(self):
        """ Updates the visible text within the relation text box """
        selected_variable = self.ui.listCompositeComponents.selectedItems()
        has_selection = bool(selected_variable)
        self.ui.lineCompositeRelation.setEnabled(has_selection)

        if has_selection:
            device_variable = self.get_selected_composite_component()
            self.ui.lineCompositeRelation.setText(device_variable['relation'])
        else:
            self.ui.lineCompositeRelation.setText("")

    def update_relation_data(self):
        """ Stores the current contents of the relation textbox into the dict of the composite variable component """
        if self.ui.lineCompositeRelation.isEnabled():
            device_variable = self.get_selected_composite_component()
            device_variable['relation'] = self.ui.lineCompositeRelation.text()

    def show_tool_tip(self):
        """ Shows the tool tip message in a pop-up window (with added message about hovering with the mouse) """
        QMessageBox.about(self, "Relation Tips", f"{relation_tool_tip}\n"
                                                 "(Can also view these tips by hovering over button)")

    def update_composite_mode(self):
        """ Updates the mode of the composite variable with the current selection in the text box """
        variable_name = self.ui.lineCompositeNickname.text().strip()
        if not variable_name or variable_name not in self.scan_composite_data['composite_variables']:
            return

        self.scan_composite_data['composite_variables'][variable_name]['mode'] = self.ui.lineCompositeMode.text()

    def _prompt_new_composite_variable(self, message: str, copy_variable: Optional[str] = None):
        """ Prompts the user for a new composite variable name, through either a new variable or a copy of an old one

        :param message: message to be displayed in pop-up
        :param copy_variable: If given, will copy this variable's contents to the new variable
        """
        text, ok = QInputDialog.getText(self, 'New Composite Variable', message)
        if ok and text:
            name = str(text).strip()

            if name in self.scan_composite_data['composite_variables']:
                logging.warning(f"'{name}' already exists in '{self.file_composite}', cannot create")
                return

            if copy_variable:
                if copy_variable not in self.scan_composite_data['composite_variables']:
                    logging.warning(f"'{copy_variable}' does not exist in '{self.file_composite}', cannot create copy")
                    return
                new_variable = copy.deepcopy(self.scan_composite_data['composite_variables'][copy_variable])
            else:
                new_variable = default_composite_variable()

            self.scan_composite_data['composite_variables'][name] = new_variable
            logging.info(f"New composite variable '{name}' created.  Not yet saved.")

            self.ui.lineCompositeNickname.setText(name)
            self.ui.lineCompositeDevice.setText("")
            self.ui.lineCompositeVariable.setText("")

            self.update_visible_composite_information()

    def create_new_composite_variable(self):
        """ Creates a new entry in composite_variables.yaml file with a user-specified name """
        self._prompt_new_composite_variable(message="Enter name for new composite variable:")

    def copy_composite_variable(self):
        """ Copies existing composite variable into a new variable with a different name """
        copy_variable = self.ui.lineCompositeNickname.text().strip()
        if copy_variable not in self.scan_composite_data['composite_variables']:
            return

        self._prompt_new_composite_variable(message=f"Enter name for copy of '{copy_variable}':",
                                            copy_variable=copy_variable)

    def delete_composite_variable(self):
        """ Prompts the user if they would like to delete the currently-selected composite variable.  This will delete
         the variable from both the .yaml file and any unsaved changes in the GUI window. """
        name = self.ui.lineCompositeNickname.text().strip()
        if name not in self.scan_composite_data['composite_variables']:
            logging.warning(f"Variable {name} not in dict, cannot delete")
            return

        reply = QMessageBox.question(self, "Delete Composite Variable", f"Delete Composite Variable '{name}' from list?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Read current version of file and delete only the specified element if it exists
            with open(self.file_composite, 'r') as file:
                scan_composite_data_actual = yaml.safe_load(file)

            if name in scan_composite_data_actual['composite_variables']:
                del scan_composite_data_actual['composite_variables'][name]
                write_dict_to_yaml_file(filename=self.file_composite, dictionary=scan_composite_data_actual)
                logging.info(f"Removed composite variable '{name}' from '{self.file_composite}'")
            else:
                logging.info(f"Removed composite variable '{name}' from unsaved dictionary")

            # Also delete from the GUI's version of composite data (leaving other changes unsaved)
            del self.scan_composite_data['composite_variables'][name]

            self.ui.lineCompositeNickname.setText("")
            self.ui.lineCompositeDevice.setText("")
            self.ui.lineCompositeVariable.setText("")

            self.update_visible_composite_information()

    def save_composite_variables_file(self):
        """ Saves all changes to composite variables """
        reply = QMessageBox.question(self, "Save",
                                     f"Save all changes to {self.file_composite.name}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            write_dict_to_yaml_file(filename=self.file_composite, dictionary=self.scan_composite_data)
