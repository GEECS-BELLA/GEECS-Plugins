"""
Action Library GUI.

A GUI to view, create, edit, and execute "actions" within the GEECS Scanner
framework. Actions are defined as structured steps (e.g., wait, set/get
device variables, or execute other actions), saved to YAML, and can be
assigned to quick-access buttons for one-click execution.

Notes
-----
- This window integrates with the main GEECS Scanner application and
  expects valid `app_paths` for reading/writing YAML files.
- Execution is gated behind a checkbox to minimize accidental runs.

-Chris
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import GEECSScannerWindow
    from geecs_scanner.app.lib.action_control import ActionControl

import copy
import logging
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget,
    QInputDialog,
    QMessageBox,
    QLineEdit,
    QPushButton,
    QFileDialog,
)
from PyQt5.QtCore import QEvent, Qt
from .gui.ActionLibrary_ui import Ui_Form
from .lib.gui_utilities import (
    parse_variable_text,
    write_dict_to_yaml_file,
    read_yaml_file_to_dict,
    display_completer_list,
    display_completer_variable_list,
)
from .lib import action_api

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def get_default_action() -> dict:
    """
    Return a default (empty) action structure.

    Returns
    -------
    dict
        Dictionary with a single key ``'steps'`` mapped to an empty list.
    """
    return {"steps": []}


class ActionLibrary(QWidget):
    """
    GUI window for managing and executing actions.

    Provides tools to:
      - List, create, copy, delete, and edit actions.
      - Append, remove, and reorder steps in an action.
      - Assign actions to quick-access buttons.
      - Execute actions (optionally from element YAML files).

    Parameters
    ----------
    main_window : GEECSScannerWindow
        Reference to the main GEECS Scanner window.
    database_dict : dict
        Mapping of device names to available variables for autocompletion.
    action_control : Optional[ActionControl]
        Controller used to execute actions. If not provided, execution
        remains disabled until one is available.

    Attributes
    ----------
    actions_file : Optional[pathlib.Path]
        Path to ``actions.yaml`` or ``None`` if paths are not configured.
    assigned_action_file : Optional[pathlib.Path]
        Path to ``assigned_actions.yaml`` or ``None`` if paths are not configured.
    actions_data : dict
        In-memory dictionary of actions (mirrors the YAML schema).
    assigned_action_list : list[AssignedAction]
        UI rows representing quick-access actions.
    enable_execute : bool
        Whether the UI allows executing actions.
    action_mode : str
        Indicates the type of the currently selected step
        (e.g., ``'wait'``, ``'execute'``, ``'set'``, ``'get'``).
    """

    def __init__(
        self,
        main_window: GEECSScannerWindow,
        database_dict: dict,
        action_control: Optional[ActionControl],
    ):
        super().__init__()

        self.main_window = main_window
        self.database_dict = database_dict

        # Initializes the GUI elements
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Action Library")

        if self.main_window.app_paths is None:
            self.actions_file = None
            self.assigned_action_file = None
        else:
            action_configurations_folder = self.main_window.app_paths.action_library()
            self.actions_file = Path(action_configurations_folder) / "actions.yaml"
            self.assigned_action_file = (
                Path(action_configurations_folder) / "assigned_actions.yaml"
            )

        self.actions_data = self.load_action_data()

        # New/Copy/Delete
        self.ui.listAvailableActions.itemSelectionChanged.connect(
            self.change_action_selection
        )
        self.ui.buttonNewAction.clicked.connect(self.create_new_action)
        self.ui.buttonCopyAction.clicked.connect(self.copy_action)
        self.ui.buttonDeleteAction.clicked.connect(self.delete_selected_action)

        # Step add/remove
        self.ui.lineActionType.installEventFilter(self)
        self.ui.buttonAddStep.clicked.connect(self.add_action)
        self.ui.buttonRemoveStep.clicked.connect(self.remove_action)

        self.ui.listActionSteps.clicked.connect(self.update_action_display)
        self.action_mode = ""
        self.ui.lineActionOption1.installEventFilter(self)
        self.ui.lineActionOption2.installEventFilter(self)
        self.ui.lineActionOption1.editingFinished.connect(self.update_action_info)
        self.ui.lineActionOption2.editingFinished.connect(self.update_action_info)
        self.ui.lineActionOption3.editingFinished.connect(self.update_action_info)
        self.update_action_display()

        self.ui.buttonMoveSooner.clicked.connect(self.move_action_sooner)
        self.ui.buttonMoveLater.clicked.connect(self.move_action_later)

        # Save/Revert
        self.ui.buttonSaveAll.clicked.connect(self.save_all_changes)
        self.ui.buttonRevertAll.clicked.connect(self.discard_all_changes)

        # Execute
        self.action_control = action_control
        self.enable_execute = False
        self.ui.buttonExecuteAction.setEnabled(False)
        self.ui.checkboxEnableExecute.toggled.connect(self.toggle_execution_enable)
        self.ui.buttonExecuteAction.clicked.connect(self.execute_action)

        # Assigned actions (quick-access buttons)
        self.assigned_action_list: list[AssignedAction] = []
        self.ui.buttonRemoveAssigned_1.setEnabled(False)
        self.ui.buttonExecuteAssigned_1.setEnabled(False)
        self.ui.lineAssignedName_1.setEnabled(False)

        self.ui.buttonAddSaveElement.clicked.connect(
            self.add_assigned_action_from_save_element
        )

        self.populate_assigned_action_list()
        self.ui.buttonAssignAction_1.clicked.connect(self.add_assigned_action)

        # Close button
        self.ui.buttonCloseWindow.clicked.connect(self.close)

        # Style
        self.setStyleSheet(main_window.styleSheet())

    # -----------------------------------------------------------------------
    # Qt Overrides / Event Filters
    # -----------------------------------------------------------------------
    def eventFilter(self, source, event):
        """
        Show context-aware completer popups on mouse click in option fields.

        Parameters
        ----------
        source : QObject
            The widget receiving the event.
        event : QEvent
            The event instance.

        Returns
        -------
        bool
            True if the event is handled; otherwise False.
        """
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionType:
            display_completer_list(
                self,
                location=self.ui.lineActionType,
                completer_list=action_api.list_of_actions,
            )
            return True
        elif (
            event.type() == QEvent.MouseButtonPress
            and source == self.ui.lineActionOption1
            and self.action_mode in ["set", "get"]
        ):
            display_completer_list(
                self,
                location=self.ui.lineActionOption1,
                completer_list=list(self.database_dict.keys()),
            )
            return True
        elif (
            event.type() == QEvent.MouseButtonPress
            and source == self.ui.lineActionOption2
            and self.action_mode in ["set", "get"]
        ):
            display_completer_variable_list(
                self,
                self.database_dict,
                list_location=self.ui.lineActionOption2,
                device_location=self.ui.lineActionOption1,
            )
            return True
        elif (
            event.type() == QEvent.MouseButtonPress
            and source == self.ui.lineActionOption1
            and self.action_mode in ["execute"]
        ):
            display_completer_list(
                self,
                location=self.ui.lineActionOption1,
                completer_list=self.actions_data["actions"].keys(),
            )
            return True
        return super().eventFilter(source, event)

    # -----------------------------------------------------------------------
    # Actions I/O and List Management
    # -----------------------------------------------------------------------
    def load_action_data(self) -> dict:
        """
        Load actions from ``actions.yaml`` (if present) and refresh the list.

        Returns
        -------
        dict
            The loaded (or initialized) actions dictionary.
        """
        self.actions_data = {"actions": {}}
        if self.actions_file and self.actions_file.exists():
            self.actions_data = read_yaml_file_to_dict(self.actions_file)
        self.populate_action_list()
        return self.actions_data

    def populate_action_list(self):
        """
        Populate the "Available Actions" list widget from in-memory data.

        Notes
        -----
        Clears the list and re-adds entries from ``self.actions_data['actions']``.
        """
        self.ui.listAvailableActions.clear()
        for action_name in self.actions_data["actions"].keys():
            self.ui.listAvailableActions.addItem(action_name)

    def get_selected_name(self) -> Optional[str]:
        """
        Return the currently selected action name.

        Returns
        -------
        str or None
            The selected action name, or ``None`` if nothing is selected or
            the selection is invalid.
        """
        selected_action = self.ui.listAvailableActions.selectedItems()
        if not selected_action:
            return None
        name = selected_action[0].text().strip()
        if name in self.actions_data["actions"]:
            return name
        return None

    def _prompt_new_action(self, message: str, copy_base: Optional[str] = None):
        """
        Prompt for a new action name and create it (optionally as a copy).

        Parameters
        ----------
        message : str
            Message displayed in the name dialog.
        copy_base : str or None, optional
            Name of an existing action to copy. If ``None``, a default empty
            action is created.

        Notes
        -----
        - If the target name already exists, creation is aborted.
        - If ``copy_base`` does not exist, creation is aborted.
        """
        text, ok = QInputDialog.getText(self, "New Action", message)
        if ok and text:
            name = str(text).strip()

            if name in self.actions_data["actions"]:
                logger.warning("'%s' already exists in dict, cannot create", name)
                return

            if copy_base:
                if copy_base not in self.actions_data["actions"]:
                    logger.warning(
                        "'%s' does not exist in dict, cannot create copy", copy_base
                    )
                    return
                new_action = copy.deepcopy(self.actions_data["actions"][copy_base])
            else:
                new_action = get_default_action()

            self.actions_data["actions"][name] = new_action
            logger.info("New action '%s' created. Not yet saved.", name)

            self.populate_action_list()

    # -----------------------------------------------------------------------
    # GUI: Action CRUD
    # -----------------------------------------------------------------------
    def create_new_action(self):
        """
        Create a new (empty) action.

        Notes
        -----
        Opens a dialog to request a unique action name.
        """
        self._prompt_new_action(message="Enter name for new action:")

    def copy_action(self):
        """
        Create a new action by copying the currently selected action.

        Notes
        -----
        - If no selection or selection not found, this is a no-op.
        - A dialog requests the new action name.
        """
        copy_base = self.get_selected_name()
        if copy_base not in self.actions_data["actions"]:
            return
        self._prompt_new_action(
            message=f"Enter name for copy of '{copy_base}':", copy_base=copy_base
        )

    def delete_selected_action(self):
        """
        Delete the selected action (with confirmation).

        Notes
        -----
        - If paths are not configured, deletion is aborted and an error is logged.
        - Removes the action from disk (if present) and from in-memory data.
        """
        name = self.get_selected_name()
        if not name or name not in self.actions_data["actions"]:
            logger.warning("No valid action to delete.")
            return

        if self.actions_file is None:
            logger.error("No path to actions.yaml defined, need to specify experiment")
            return

        reply = QMessageBox.question(
            self,
            "Delete Action",
            f"Delete action '{name}' from file?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            # Read current version of file and delete only the specified element if it exists
            actions_data_actual = read_yaml_file_to_dict(self.actions_file)

            if name in actions_data_actual["actions"]:
                del actions_data_actual["actions"][name]
                write_dict_to_yaml_file(
                    filename=self.actions_file, dictionary=actions_data_actual
                )
                logger.info("Removed action '%s' from '%s'", name, self.actions_file)
            else:
                logger.info("Removed action '%s' from unsaved dictionary", name)

            # Also delete from the GUI's version (leaving other changes unsaved)
            del self.actions_data["actions"][name]

            self.populate_action_list()

    # -----------------------------------------------------------------------
    # GUI: Step List Management
    # -----------------------------------------------------------------------
    def change_action_selection(self):
        """
        Refresh the step list and display when the action selection changes.

        Notes
        -----
        Called when selection changes or when steps are added/removed.
        """
        self.update_action_list()
        self.update_action_display()

    def update_action_list(self, index: Optional[int] = None):
        """
        Refresh the step list widget for the currently selected action.

        Parameters
        ----------
        index : int or None, optional
            If provided and valid, sets the current row selection after refresh.
        """
        self.ui.listActionSteps.clear()
        if name := self.get_selected_name():
            for item in self.actions_data["actions"][name]["steps"]:
                self.ui.listActionSteps.addItem(
                    action_api.generate_action_description(item)
                )

            if index is not None and 0 <= index < self.ui.listActionSteps.count():
                self.ui.listActionSteps.setCurrentRow(index)

    def add_action(self):
        """
        Append a new step to the end of the current action.

        Notes
        -----
        The step type is read from the "Action Type" line edit.
        """
        if text := self.ui.lineActionType.text().strip():
            if name := self.get_selected_name():
                self.actions_data["actions"][name]["steps"].append(
                    action_api.get_new_action(text)
                )
                self.ui.lineActionType.clear()
                self.change_action_selection()

    def remove_action(self):
        """
        Remove the currently selected step from the current action.

        Notes
        -----
        No-op if selection is invalid.
        """
        index = self.get_selected_step_index()
        if index is not None:
            name = self.get_selected_name()
            if index >= 0 and name:
                del self.actions_data["actions"][name]["steps"][index]
            self.change_action_selection()

    def update_action_display(self):
        """
        Update the option labels/fields to match the selected step.

        Notes
        -----
        Enables/disables and populates the three option fields depending
        on the selected step type.
        """
        self.action_mode = ""
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
                action = self.actions_data["actions"][name]["steps"][index]
                if action["action"] == "wait":
                    self.action_mode = "wait"
                    self.ui.labelActionOption1.setText("Wait Time (s):")
                    self.ui.lineActionOption1.setEnabled(True)
                    self.ui.lineActionOption1.setText(str(action.get("wait")))
                elif action["action"] == "execute":
                    self.action_mode = "execute"
                    self.ui.labelActionOption1.setText("Action Name:")
                    self.ui.lineActionOption1.setEnabled(True)
                    self.ui.lineActionOption1.setText(action.get("action_name"))
                elif action["action"] == "set":
                    self.action_mode = "set"
                    self.ui.labelActionOption1.setText("GEECS Device Name:")
                    self.ui.lineActionOption1.setEnabled(True)
                    self.ui.lineActionOption1.setText(action.get("device"))
                    self.ui.labelActionOption2.setText("Variable Name:")
                    self.ui.lineActionOption2.setEnabled(True)
                    self.ui.lineActionOption2.setText(action.get("variable"))
                    self.ui.labelActionOption3.setText("Set Value:")
                    self.ui.lineActionOption3.setEnabled(True)
                    self.ui.lineActionOption3.setText(str(action.get("value")))
                elif action["action"] == "get":
                    self.action_mode = "get"
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
        """
        Write edits from the option fields back into the current step.

        Notes
        -----
        Parses numeric expressions for numeric fields using
        :func:`parse_variable_text`.
        """
        index = self.get_selected_step_index()
        if index is not None:
            name = self.get_selected_name()
            if index >= 0 and name:
                action = self.actions_data["actions"][name]["steps"][index]

                if action["action"] == "wait":
                    action["wait"] = parse_variable_text(
                        self.ui.lineActionOption1.text().strip()
                    )
                elif action["action"] == "execute":
                    action["action_name"] = self.ui.lineActionOption1.text().strip()
                elif action["action"] == "run":
                    action["file_name"] = self.ui.lineActionOption1.text().strip()
                    action["class_name"] = self.ui.lineActionOption2.text().strip()
                elif action["action"] == "set":
                    action["device"] = self.ui.lineActionOption1.text().strip()
                    action["variable"] = self.ui.lineActionOption2.text().strip()
                    action["value"] = parse_variable_text(
                        self.ui.lineActionOption3.text().strip()
                    )
                elif action["action"] == "get":
                    action["device"] = self.ui.lineActionOption1.text().strip()
                    action["variable"] = self.ui.lineActionOption2.text().strip()
                    action["expected_value"] = parse_variable_text(
                        self.ui.lineActionOption3.text().strip()
                    )

                self.update_action_list(index=index)

    def get_selected_step_index(self) -> Optional[int]:
        """
        Return the index of the selected step in the step list.

        Returns
        -------
        int or None
            Row index of the selected step, or ``None`` if no selection.
        """
        selected_action = self.ui.listActionSteps.selectedItems()
        if not selected_action:
            return None
        index = self.ui.listActionSteps.row(selected_action[0])
        return index

    def move_action_sooner(self):
        """
        Move the selected step one position earlier in the list.

        Notes
        -----
        No-op if the selected step is the first or selection is invalid.
        """
        i = self.get_selected_step_index()
        if i is not None:
            if name := self.get_selected_name():
                action_list = self.actions_data["actions"][name]["steps"]
                if 0 < i < len(action_list):
                    action_list[i], action_list[i - 1] = (
                        action_list[i - 1],
                        action_list[i],
                    )
                    i -= 1
                    self.update_action_list(index=i)

    def move_action_later(self):
        """
        Move the selected step one position later in the list.

        Notes
        -----
        No-op if the selected step is the last or selection is invalid.
        """
        i = self.get_selected_step_index()
        if i is not None:
            if name := self.get_selected_name():
                action_list = self.actions_data["actions"][name]["steps"]
                if 0 <= i < len(action_list) - 1:
                    action_list[i], action_list[i + 1] = (
                        action_list[i + 1],
                        action_list[i],
                    )
                    i += 1
                    self.update_action_list(index=i)

    # -----------------------------------------------------------------------
    # Execute / Persist
    # -----------------------------------------------------------------------
    def toggle_execution_enable(self):
        """
        Enable or disable execution controls based on checkbox and experiment state.

        Notes
        -----
        Execution requires both:
          1) the checkbox enabled, and
          2) an active experiment in the main window.
        """
        self.enable_execute = bool(
            self.ui.checkboxEnableExecute.isChecked() and self.main_window.experiment
        )
        self.ui.buttonExecuteAction.setEnabled(self.enable_execute)
        for assigned_action in self.assigned_action_list:
            assigned_action.buttonExecute.setEnabled(self.enable_execute)

    def execute_action(
        self, name: Optional[str] = None, element_filename: Optional[str] = None
    ):
        """
        Execute an action using :class:`ActionControl`.

        Parameters
        ----------
        name : str or None, optional
            Name of the action to execute. If ``None``, uses the selected action.
        element_filename : str or None, optional
            If provided, executes the action from the specified save-element YAML
            file instead of the main actions dictionary.

        Notes
        -----
        Displays a confirmation dialog before executing.
        """
        name = name or self.get_selected_name()
        if element_filename is None and (
            name is None or name not in self.actions_data["actions"]
        ):
            return
        else:
            message = (
                name if element_filename is None else f"{name}:  {element_filename}"
            )
            reply = QMessageBox.question(
                self,
                "Execute Action",
                f"Execute action '{message}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                if element_filename is None:
                    self.action_control.perform_action(
                        self.actions_data["actions"][name]
                    )
                else:
                    save_device_folder = self.main_window.app_paths.save_devices()
                    element = read_yaml_file_to_dict(
                        save_device_folder / f"{element_filename}"
                    )
                    self.action_control.perform_action(element[name])

    def save_all_changes(self):
        """
        Save the in-memory actions dictionary to ``actions.yaml``.

        Notes
        -----
        If paths are not configured, logs an error and returns.
        """
        if self.actions_file is None:
            logger.error("No path to actions.yaml defined, need to specify experiment")
            return

        reply = QMessageBox.question(
            self,
            "Save Actions",
            f"Save all changes to {self.actions_file.name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            write_dict_to_yaml_file(
                filename=self.actions_file, dictionary=self.actions_data
            )
            if self.action_control:
                self.action_control = self.main_window.refresh_action_control()

    def discard_all_changes(self):
        """
        Reload actions from disk, discarding unsaved changes.

        Notes
        -----
        A confirmation dialog is shown prior to reloading.
        """
        reply = QMessageBox.question(
            self,
            "Discard Changes",
            "Discard all unsaved changes?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.load_action_data()

    def closeEvent(self, event):
        """
        Persist assigned actions on close and notify the main window.

        Parameters
        ----------
        event : QCloseEvent
            Close event passed by Qt.
        """
        if self.assigned_action_file:
            updated_names = []
            for action in self.assigned_action_list:
                assigned_dict_element = {"name": action.get_name()}
                element_filename = action.get_element_filename()
                if element_filename:
                    assigned_dict_element["element_filename"] = element_filename
                updated_names.append(assigned_dict_element)
            write_dict_to_yaml_file(
                self.assigned_action_file, {"assigned_actions": updated_names}
            )

        self.main_window.exit_action_library()
        event.accept()

    # -----------------------------------------------------------------------
    # Assigned Action Rows (Quick-Access)
    # -----------------------------------------------------------------------
    def populate_assigned_action_list(self):
        """
        Populate quick-access rows from ``assigned_actions.yaml`` (if present).

        Notes
        -----
        Replaces the in-memory list and refreshes the GUI layout.
        """
        assigned_action_dict = {}
        if self.assigned_action_file and self.assigned_action_file.exists():
            assigned_action_dict = read_yaml_file_to_dict(self.assigned_action_file)

        # For each action in the list, create AssignedAction instances
        self.assigned_action_list = []
        for action in assigned_action_dict.get("assigned_actions", []):
            name = action.get("name")
            element_filename = action.get("element_filename", None)
            self.assigned_action_list.append(
                AssignedAction(
                    parent_gui=self, action_name=name, element_filename=element_filename
                )
            )

        self.refresh_assigned_action_gui()

    def add_assigned_action(self):
        """
        Assign the currently selected action as a new quick-access row.

        Notes
        -----
        No-op if there is no valid current selection.
        """
        if name := self.get_selected_name():
            self.assigned_action_list.append(
                AssignedAction(parent_gui=self, action_name=name)
            )
            self.refresh_assigned_action_gui()

    def add_assigned_action_from_save_element(self):
        """
        Assign an action from a save-element YAML file as a quick-access row.

        Notes
        -----
        Prompts the user to select a YAML. If both ``setup_action`` and
        ``closeout_action`` exist, the user is asked to choose which to import.
        """
        if self.main_window.app_paths is None:
            logger.error("Paths set incorrectly for experiment")
            return

        # Prompt the user for a file from the save element folder
        save_device_folder = self.main_window.app_paths.save_devices()
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select a YAML File",
            str(save_device_folder),
            "YAML Files (*yaml)",
            options=options,
        )
        if file_name:
            contents = read_yaml_file_to_dict(Path(file_name))

            # Get either the setup or closeout actions, prompting the user if both exist
            selection = ""
            if "setup_action" in contents and "closeout_action" in contents:
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Import Save Element Action")
                msg_box.setText("Use 'setup_action' or 'closeout_action'?")
                button_setup = msg_box.addButton("Setup", QMessageBox.ActionRole)
                button_closeout = msg_box.addButton("Closeout", QMessageBox.ActionRole)
                msg_box.addButton("Cancel", QMessageBox.ActionRole)
                msg_box.exec_()

                if msg_box.clickedButton() == button_setup:
                    selection = "setup_action"
                elif msg_box.clickedButton() == button_closeout:
                    selection = "closeout_action"
                else:
                    return

            elif not selection and "setup_action" in contents:
                selection = "setup_action"
            elif not selection and "closeout_action" in contents:
                selection = "closeout_action"
            else:
                logger.info("No actions in '%s'", file_name)
                return

            self.assigned_action_list.append(
                AssignedAction(
                    parent_gui=self,
                    action_name=selection,
                    element_filename=Path(file_name).name,
                )
            )
            self.refresh_assigned_action_gui()

    def refresh_assigned_action_gui(self):
        """
        Lay out and style the quick-access rows, then move the default widgets.

        Notes
        -----
        - Filters out any rows with empty names.
        - Alternates background color for readability.
        """
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

        default_widgets = [
            self.ui.buttonAssignAction_1,
            self.ui.buttonRemoveAssigned_1,
            self.ui.buttonExecuteAssigned_1,
            self.ui.lineAssignedName_1,
        ]
        for widget in default_widgets:
            widget.move(widget.pos().x(), y_position)

        self.resize(self.width(), y_position + 46)


class AssignedAction:
    """
    A single quick-access row for an assigned action.

    Provides buttons to assign a different action, remove the row, and execute
    the current action, along with a read-only line displaying the action name
    (and optional element filename).

    Parameters
    ----------
    parent_gui : ActionLibrary
        The owning ActionLibrary instance.
    action_name : str
        The assigned action name.
    element_filename : str or None, optional
        If set, indicates the save-element file that owns this action.
    """

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        parent_gui: ActionLibrary,
        action_name: str,
        element_filename: Optional[str] = None,
    ):
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

        self.widgets = [
            self.buttonAssign,
            self.buttonRemove,
            self.buttonExecute,
            self.lineName,
        ]
        for widget in self.widgets:
            widget.show()

        self.buttonAssign.clicked.connect(self.reassign_self)
        self.buttonRemove.clicked.connect(self.remove_self)
        self.buttonExecute.clicked.connect(self.execute_action)

    def set_y_pos(self, y_pos: int):
        """
        Align the row's widgets to a specific vertical position.

        Parameters
        ----------
        y_pos : int
            The y-coordinate (in pixels) for the row baseline.
        """
        for widget in self.widgets:
            widget.move(widget.pos().x(), y_pos)

    def set_color(self, flag: bool):
        """
        Apply alternating background color to the line edit.

        Parameters
        ----------
        flag : bool
            ``True`` for white, ``False`` for light gray.
        """
        if flag:
            self.lineName.setStyleSheet("background-color: white; color: black")
        else:
            self.lineName.setStyleSheet("background-color: lightgray; color: black")

    def _set_line_text(self):
        """
        Render the display text (action name and optional filename).

        Notes
        -----
        Uses a simple "Action:  filename" format if a filename is present.
        """
        if self.element_filename is None:
            self.lineName.setText(self.action_name)
        else:
            self.lineName.setText(f"{self.action_name}:  {self.element_filename}")

    def reassign_self(self):
        """
        Reassign this row to the currently selected action (no UI rebuild).

        Notes
        -----
        Clears ``element_filename`` when reassigned to a named action.
        """
        new_name = self.parent.get_selected_name()
        if not new_name:
            return
        self.action_name = new_name
        self.element_filename = None
        self._set_line_text()

    def remove_self(self):
        """
        Remove this row and destroy associated widgets.

        Notes
        -----
        Triggers a layout refresh on the parent GUI.
        """
        self.action_name = ""

        for widget in self.widgets:
            widget.setParent(None)
            widget.deleteLater()
            widget = None  # noqa: F841

        self.parent.refresh_assigned_action_gui()

    def execute_action(self):
        """Trigger execution of the assigned action via the parent GUI."""
        self.parent.execute_action(
            name=self.action_name, element_filename=self.element_filename
        )

    def get_name(self) -> str:
        """
        Return the assigned action name.

        Returns
        -------
        str
            The action name (may be an empty string if removed).
        """
        return self.action_name

    def get_element_filename(self) -> Optional[str]:
        """
        Return the associated element filename, if any.

        Returns
        -------
        str or None
            Save-element filename or ``None``.
        """
        return self.element_filename
