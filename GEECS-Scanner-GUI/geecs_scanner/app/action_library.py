"""
GUI that shows the list of available actions and allows the user to edit and create these actions.  Additionally, they
can be assigned to quick-access buttons and be executed in this window.  Allows for complex actions to be scripted and
executed within the GEECS Scanner framework.

-Chris
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional
if TYPE_CHECKING:
    from . import GEECSScannerWindow
    from geecs_scanner.app.lib.action_control import ActionControl

import copy
import logging
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QInputDialog, QMessageBox, QLineEdit, QPushButton, QFileDialog
from PyQt5.QtCore import QEvent, Qt
from .gui.ActionLibrary_ui import Ui_Form
from .lib.gui_utilities import (parse_variable_text, write_dict_to_yaml_file, read_yaml_file_to_dict,
                                display_completer_list, display_completer_variable_list)

from .lib import action_api


def get_default_action() -> dict:
    """
    :return: The default formatting for an action as saved in yaml files: a dict with an empty list of name 'steps'
    """
    return {'steps': []}


class ActionLibrary(QWidget):
    def __init__(self, main_window: GEECSScannerWindow, database_dict: dict, action_control: Optional[ActionControl]):
        """ GUI Window that holds all the action library elements

        :param main_window: Reference to the main GEECS Scanner window
        :param database_dict: Dictionary containing all devices and variables in the experiment
        :param action_control: instance of action control if experiment was successfully connected
        """
        super().__init__()

        self.main_window = main_window
        self.database_dict = database_dict

        # Initializes the gui elements
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Action Library")

        if self.main_window.app_paths is None:
            self.actions_file = None
            self.assigned_action_file = None
        else:
            action_configurations_folder = self.main_window.app_paths.action_library()
            self.actions_file = Path(action_configurations_folder) / 'actions.yaml'
            self.assigned_action_file = Path(action_configurations_folder) / 'assigned_actions.yaml'

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

        # Functionality to execute actions
        self.action_control = action_control
        self.enable_execute = False
        self.ui.buttonExecuteAction.setEnabled(False)
        self.ui.checkboxEnableExecute.toggled.connect(self.toggle_execution_enable)
        self.ui.buttonExecuteAction.clicked.connect(self.execute_action)

        # Functionality to assign actions to the buttons at the bottom
        self.assigned_action_list: list[AssignedAction] = []
        self.ui.buttonRemoveAssigned_1.setEnabled(False)
        self.ui.buttonExecuteAssigned_1.setEnabled(False)
        self.ui.lineAssignedName_1.setEnabled(False)

        self.ui.buttonAddSaveElement.clicked.connect(self.add_assigned_action_from_save_element)

        self.populate_assigned_action_list()
        self.ui.buttonAssignAction_1.clicked.connect(self.add_assigned_action)

        # Functionality for close button
        self.ui.buttonCloseWindow.clicked.connect(self.close)

        # Set stylesheet to that of the main window
        self.setStyleSheet(main_window.styleSheet())

    def eventFilter(self, source, event):
        """Creates a custom event for the text boxes so that the completion suggestions are shown when mouse is clicked
        """
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionType:
            display_completer_list(self, location=self.ui.lineActionType, completer_list=action_api.list_of_actions)
            return True
        elif (event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionOption1
              and self.action_mode in ['set', 'get']):
            display_completer_list(self, location=self.ui.lineActionOption1,
                                   completer_list=list(self.database_dict.keys()))
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
        """ Loads all actions from the actions.yaml file into the `actions_data` instance variable dict

        :return: the dictionary that was loaded
        """
        self.actions_data = {'actions': {}}
        if self.actions_file and self.actions_file.exists():
            self.actions_data = read_yaml_file_to_dict(self.actions_file)
        self.populate_action_list()
        return self.actions_data

    def populate_action_list(self):
        """ Clears the list of available actions and adds all actions within the instance variable dict """
        self.ui.listAvailableActions.clear()
        for action_name in self.actions_data['actions'].keys():
            self.ui.listAvailableActions.addItem(action_name)

    def get_selected_name(self) -> Optional[str]:
        """  Gets the name of the currently-selected action from among the list of actions

        :return: String of selected action name, None if nothing selected
        """
        selected_action = self.ui.listAvailableActions.selectedItems()
        if not selected_action:
            return None
        name = selected_action[0].text().strip()
        if name in self.actions_data['actions']:
            return name

    def _prompt_new_action(self, message: str, copy_base: Optional[str] = None):
        """ Prompts the user for a name when creating a new action or copying an existing action

        :param message: Message displayed on pop-up window
        :param copy_base: Optional name of existing action to be copied, ignored if not given
        """
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
        """ Creates a new action in the instance variable dict with a user-specified name """
        self._prompt_new_action(message="Enter name for new action:")

    def copy_action(self):
        """ Creates a new action as a copy of the currently-selected action """
        copy_base = self.get_selected_name()
        if copy_base not in self.actions_data['actions']:
            return

        self._prompt_new_action(message=f"Enter name for copy of '{copy_base}':", copy_base=copy_base)

    def delete_selected_action(self):
        """ Prompts the user if they want to delete the selected action, doing so if they reply 'yes' """
        name = self.get_selected_name()
        if not name or name not in self.actions_data['actions']:
            logging.warning(f"No valid action to delete.")
            return

        if self.actions_file is None:
            logging.error("No path to actions.yaml defined, need to specify experiment")
            return

        reply = QMessageBox.question(self, "Delete Action", f"Delete action '{name}' from file?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Read current version of file and delete only the specified element if it exists
            actions_data_actual = read_yaml_file_to_dict(self.actions_file)

            if name in actions_data_actual['actions']:
                del actions_data_actual['actions'][name]
                write_dict_to_yaml_file(filename=self.actions_file, dictionary=actions_data_actual)
                logging.info(f"Removed action '{name}' from '{self.actions_file}'")
            else:
                logging.info(f"Removed action '{name}' from unsaved dictionary")

            # Also delete from the GUI's version of composite data (leaving other changes unsaved)
            del self.actions_data['actions'][name]

            self.populate_action_list()

    # # # # # # # # # # # GUI elements for interacting with the list of steps # # # # # # # # # # #

    def change_action_selection(self):
        """ Refreshes the list of steps when a step is added or deleted, or when the action selection changes """
        self.update_action_list()
        self.update_action_display()

    def update_action_list(self, index: Optional[int] = None):
        """ Updates the visible list of actions on the GUI according to the current state of the action dictionary.

        :param index: Optional, if given then the current selection is updated to this index after refreshing
        """
        self.ui.listActionSteps.clear()
        if name := self.get_selected_name():
            for item in self.actions_data['actions'][name]['steps']:
                self.ui.listActionSteps.addItem(action_api.generate_action_description(item))

            if index is not None and 0 <= index < self.ui.listActionSteps.count():
                self.ui.listActionSteps.setCurrentRow(index)

    def add_action(self):
        """ Appends action to the end of either the setup or closeout action list based on the currently-selected
        radio button.  The action is specified by the action line edit. """
        if text := self.ui.lineActionType.text().strip():
            if name := self.get_selected_name():
                self.actions_data['actions'][name]['steps'].append(action_api.get_new_action(text))
                self.ui.lineActionType.clear()
                self.change_action_selection()

    def remove_action(self):
        """ Removes the currently-selected action from the list of actions """
        index = self.get_selected_step_index()
        if index is not None:
            name = self.get_selected_name()
            if index >= 0 and name:
                del self.actions_data['actions'][name]['steps'][index]

            self.change_action_selection()

    def update_action_display(self):
        """ Updates the visible list of information based on the currently-selected action """
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

        index = self.get_selected_step_index()
        if index is not None:
            name = self.get_selected_name()
            if index >= 0 and name:
                action = self.actions_data['actions'][name]['steps'][index]
                if action['action'] == 'wait':
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
        if index is not None:
            name = self.get_selected_name()
            if index >= 0 and name:
                action = self.actions_data['actions'][name]['steps'][index]

                if action['action'] == 'wait':
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

    def get_selected_step_index(self) -> Optional[int]:
        """Finds the location of the currently-selected action in the visible list on the GUI.

        :return: If no selection, returns -1.  Otherwise, return the index of the currently selected step
        """
        selected_action = self.ui.listActionSteps.selectedItems()
        if not selected_action:
            return None

        index = self.ui.listActionSteps.row(selected_action[0])
        return index

    def move_action_sooner(self):
        """Moves the selected action to an earlier position in the same list"""
        i = self.get_selected_step_index()
        if i is not None:
            if name := self.get_selected_name():
                action_list = self.actions_data['actions'][name]['steps']
                if 0 < i < len(action_list):
                    action_list[i], action_list[i - 1] = action_list[i - 1], action_list[i]
                    i -= 1
                    self.update_action_list(index=i)

    def move_action_later(self):
        """Moves the selected action to a later position in the same list"""
        i = self.get_selected_step_index()
        if i is not None:
            if name := self.get_selected_name():
                action_list = self.actions_data['actions'][name]['steps']
                if 0 <= i < len(action_list) - 1:
                    action_list[i], action_list[i + 1] = action_list[i + 1], action_list[i]
                    i += 1
                    self.update_action_list(index=i)

    # # # # # # # # # # # GUI elements for Execute, Save, Revert, and Close buttons # # # # # # # # # # #

    def toggle_execution_enable(self):
        """ In an effort to avoid accidental clicks, the "execute action" buttons are only enabled once the user
         checks this box.  If this is the first time the box is checked, an ActionControl instance is created """
        self.enable_execute = bool(self.ui.checkboxEnableExecute.isChecked() and self.main_window.experiment)

        self.ui.buttonExecuteAction.setEnabled(self.enable_execute)
        for assigned_action in self.assigned_action_list:
            assigned_action.buttonExecute.setEnabled(self.enable_execute)

    def execute_action(self, name: Optional[str] = None, element_filename: Optional[str] = None):
        """ Executes an action by sending the contents of the instance variable dict to ActionControl

        :param name: Optional name for action to execute.  If None given, will try the currently-selected action
        :param element_filename: If given, will try to execute the named action in the associated element yaml file
        """
        name = name or self.get_selected_name()
        if element_filename is None and (name is None or name not in self.actions_data['actions']):
            return
        else:
            message = name if element_filename is None else f"{name}:  {element_filename}"
            reply = QMessageBox.question(self, "Execute Action", f"Execute action '{message}'?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                if element_filename is None:
                    self.action_control.perform_action(self.actions_data['actions'][name])
                else:
                    save_device_folder = self.main_window.app_paths.save_devices()
                    element = read_yaml_file_to_dict(save_device_folder / f"{element_filename}")
                    self.action_control.perform_action(element[name])

    def save_all_changes(self):
        """ Save the current version of the instance variable dict to the actions.yaml file """
        if self.actions_file is None:
            logging.error("No path to actions.yaml defined, need to specify experiment")
            return

        reply = QMessageBox.question(self, "Save Actions", f"Save all changes to {self.actions_file.name}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            write_dict_to_yaml_file(filename=self.actions_file, dictionary=self.actions_data)

            if self.action_control:
                self.action_control = self.main_window.refresh_action_control()

    def discard_all_changes(self):
        """ Replace the current version of the instance variable dict with the contents of the actions.yaml file """
        reply = QMessageBox.question(self, "Discard Changes", f"Discard all unsaved changes?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.load_action_data()

    def closeEvent(self, event):
        """ When closing this window, saved assigned actions to its yaml file and tell the main window we've exited """
        if self.assigned_action_file:
            updated_names = []
            for action in self.assigned_action_list:
                assigned_dict_element = {'name': action.get_name()}
                element_filename = action.get_element_filename()
                if element_filename:
                    assigned_dict_element['element_filename'] = element_filename
                updated_names.append(assigned_dict_element)
            write_dict_to_yaml_file(self.assigned_action_file, {"assigned_actions": updated_names})

        self.main_window.exit_action_library()
        event.accept()

    # # # # # # # # # # # Code to set up the assigned actions # # # # # # # # # # #

    def populate_assigned_action_list(self):
        """ Upon opening this GUI, read the assigned actions yaml file for the assigned actions from last time """
        assigned_action_dict = {}
        if self.assigned_action_file and self.assigned_action_file.exists():
            assigned_action_dict = read_yaml_file_to_dict(self.assigned_action_file)

        # For each action in the list, populate the list of AssignedAction class instances
        self.assigned_action_list = []
        for action in assigned_action_dict.get("assigned_actions", []):
            name = action.get('name')
            element_filename = action.get('element_filename', None)
            self.assigned_action_list.append(AssignedAction(parent_gui=self, action_name=name,
                                                            element_filename=element_filename))

        self.refresh_assigned_action_gui()

    def add_assigned_action(self):
        """ Add an action to the list of assigned actions, creating a row of GUI elements in the process """
        if name := self.get_selected_name():
            self.assigned_action_list.append(AssignedAction(parent_gui=self, action_name=name))
            self.refresh_assigned_action_gui()

    def add_assigned_action_from_save_element(self):
        """ Prompts the user to add an action from a save element yaml to the list of assigned actions """
        if self.main_window.app_paths is None:
            logging.error("Paths set incorrectly for experiment")
            return

        # Prompt the user for a file from the save element folder
        save_device_folder = self.main_window.app_paths.save_devices()
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select a YAML File", str(save_device_folder),
                                                   "YAML Files (*yaml)", options=options)
        if file_name:
            contents = read_yaml_file_to_dict(Path(file_name))

            # Get either the setup or closeout actions, prompting the user if both exist
            selection = ""
            if 'setup_action' in contents and 'closeout_action' in contents:
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Import Save Element Action")
                msg_box.setText("Use 'setup_action' or 'closeout_action'?")
                button_setup = msg_box.addButton("Setup", QMessageBox.ActionRole)
                button_closeout = msg_box.addButton("Closeout", QMessageBox.ActionRole)
                msg_box.addButton("Cancel", QMessageBox.ActionRole)
                msg_box.exec_()

                if msg_box.clickedButton() == button_setup:
                    selection = 'setup_action'
                elif msg_box.clickedButton() == button_closeout:
                    selection = 'closeout_action'
                else:
                    return

            elif not selection and 'setup_action' in contents:
                selection = 'setup_action'
            elif not selection and 'closeout_action' in contents:
                selection = 'closeout_action'
            else:
                logging.info(f"No actions in '{file_name}'")
                return

            self.assigned_action_list.append(AssignedAction(parent_gui=self, action_name=selection,
                                                            element_filename=Path(file_name).name))
            self.refresh_assigned_action_gui()

    def refresh_assigned_action_gui(self):
        """ Refreshes the GUI elements to reflect current list of assigned actions, moving elements as needed """
        y_position = self.ui.line_2.pos().y() + 40
        color_flag = True
        updated_list: list[AssignedAction] = []
        for action in self.assigned_action_list:
            if action.get_name():
                action.set_y_pos(y_position)
                action.set_color(color_flag)
                updated_list.append(action)

                y_position += 30
                color_flag = not color_flag

        self.assigned_action_list = updated_list

        default_widgets = [self.ui.buttonAssignAction_1, self.ui.buttonRemoveAssigned_1,
                           self.ui.buttonExecuteAssigned_1, self.ui.lineAssignedName_1]
        for widget in default_widgets:
            widget.move(widget.pos().x(), y_position)

        self.resize(self.width(), y_position+46)


class AssignedAction:
    """
    A GUI class that represents a single assigned action.  This includes a "reassign button, a delete button, an execute
    button, and a text box that displays the name of the assigned action.
    """

    # noinspection PyUnresolvedReferences
    def __init__(self, parent_gui: ActionLibrary, action_name: str, element_filename: Optional[str] = None):
        """
        :param parent_gui: The "ActionLibrary" gui instance that spawned this AssignedAction
        :param action_name: stored action for this instance
        :param element_filename: If given, this is the filename for the element that owns this action
        """
        self.parent = parent_gui
        self.action_name = action_name

        self.element_filename = element_filename

        self.buttonAssign = QPushButton(self.parent)
        self.buttonAssign.setText("Assign")
        self.buttonAssign.move(20, 420)
        self.buttonAssign.resize(61, 28)

        self.buttonRemove = QPushButton(self.parent)
        self.buttonRemove.setText("Remove")
        self.buttonRemove.move(90, 420)
        self.buttonRemove.resize(61, 28)

        self.buttonExecute = QPushButton(self.parent)
        self.buttonExecute.setText("Execute")
        self.buttonExecute.setEnabled(self.parent.enable_execute)
        self.buttonExecute.move(540, 420)
        self.buttonExecute.resize(61, 28)

        self.lineName = QLineEdit(self.parent)
        self.lineName.setReadOnly(True)
        self.lineName.setAlignment(Qt.AlignRight)
        self._set_line_text()
        self.lineName.move(160, 420)
        self.lineName.resize(371, 28)

        self.widgets = [self.buttonAssign, self.buttonRemove, self.buttonExecute, self.lineName]
        for widget in self.widgets:
            widget.show()

        self.buttonAssign.clicked.connect(self.reassign_self)
        self.buttonRemove.clicked.connect(self.remove_self)
        self.buttonExecute.clicked.connect(self.execute_action)

    def set_y_pos(self, y_pos: int):
        """ Aligns the assigned action elements to the specified vertical position """
        for widget in self.widgets:
            widget.move(widget.pos().x(), y_pos)

    def set_color(self, flag: bool):
        """ Sets the color of the text box to be either white or lightgray, according to an alternating pattern in
         `refresh_assigned_action_gui()`

        :param flag: True for white, False for light gray
        """
        if flag:
            self.lineName.setStyleSheet("background-color: white; color: black")
        else:
            self.lineName.setStyleSheet("background-color: lightgray; color: black")

    def _set_line_text(self):
        """ Sets the text for the line edit to be either the action name or additionally include the filename """
        if self.element_filename is None:
            self.lineName.setText(self.action_name)
        else:
            self.lineName.setText(f"{self.action_name}:  {self.element_filename}")

    def reassign_self(self):
        """ Give a new name to this instance of AssignedAction, without deleting the buttons.  Uses current selection
         in ActionLibrary's list widget """
        new_name = self.parent.get_selected_name()
        if not new_name:
            return

        self.action_name = new_name
        self.element_filename = None
        self._set_line_text()

    def remove_self(self):
        """ Deletes itself and removes all associated widgets from memory, then calls ActionLibrary's refresh """
        self.action_name = ""

        for widget in self.widgets:
            widget.setParent(None)
            widget.deleteLater()
            widget = None

        self.parent.refresh_assigned_action_gui()

    def execute_action(self):
        """ Passes the assigned action name to ActionLibrary's `execute` function """
        self.parent.execute_action(name=self.action_name, element_filename=self.element_filename)

    def get_name(self) -> str:
        return self.action_name

    def get_element_filename(self) -> Optional[str]:
        return self.element_filename
