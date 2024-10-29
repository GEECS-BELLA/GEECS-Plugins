import sys
from PyQt5.QtWidgets import QDialog, QCompleter, QPushButton
from PyQt5.QtCore import Qt, QEvent
from ScanElementEditor_ui import Ui_Dialog


def get_default_device_dictionary():
    return {
        'variable_list': ['timestamp'],
        'synchronous': True,
        'save_nonscalar_data': True
    }


def get_new_action(action):
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


list_of_actions = [
    'set',
    'get',
    'wait',
    'execute',
    #'run'
]


class ScanElementEditor(QDialog):
    def __init__(self, database_dict=None):
        super().__init__()

        self.dummyButton = QPushButton("", self)
        self.dummyButton.setDefault(True)
        self.dummyButton.setVisible(False)

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.database_dict = database_dict

        self.devices_dict = {}
        self.actions_dict = {
            'setup': [],
            'closeout': []
        }

        self.ui.lineDeviceName.installEventFilter(self)
        self.ui.buttonAddDevice.clicked.connect(self.add_device)
        self.ui.buttonRemoveDevice.clicked.connect(self.remove_device)
        self.ui.listDevices.itemSelectionChanged.connect(self.update_variable_list)

        self.ui.lineVariableName.installEventFilter(self)
        self.ui.buttonAddVariable.clicked.connect(self.add_variable)
        self.ui.buttonRemoveVariable.clicked.connect(self.remove_variable)

        self.ui.checkboxSynchronous.clicked.connect(self.update_device_checkboxes)
        self.ui.checkboxSaveNonscalar.clicked.connect(self.update_device_checkboxes)

        self.ui.lineActionName.setReadOnly(True)
        self.ui.lineActionName.installEventFilter(self)

        self.ui.listActions.itemSelectionChanged.connect(self.focus_action)
        self.ui.buttonAddAction.clicked.connect(self.add_action)
        self.ui.buttonRemoveAction.clicked.connect(self.remove_action)

        self.action_mode = None
        self.ui.lineActionOption1.installEventFilter(self)
        self.ui.lineActionOption2.installEventFilter(self)
        self.ui.lineActionOption1.editingFinished.connect(self.update_action_info)
        self.ui.lineActionOption2.editingFinished.connect(self.update_action_info)
        self.ui.lineActionOption3.editingFinished.connect(self.update_action_info)

        self.ui.buttonMoveSooner.clicked.connect(self.move_action_sooner)
        self.ui.buttonMoveLater.clicked.connect(self.move_action_later)

        self.ui.buttonWindowSave.clicked.connect(self.save_element)
        self.ui.buttonWindowCancel.clicked.connect(self.close_window)
        self.ui.buttonWindowLoad.clicked.connect(self.open_element)

        self.update_device_list()
        self.update_action_list()
        self.update_action_display()

    def eventFilter(self, source, event):
        # Creates a custom event for the text boxes so that the completion suggestions are shown when mouse is clicked
        if event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionName:
            self.show_action_list()
            self.ui.buttonAddAction.setDefault(True)
            return True
        elif event.type() == QEvent.MouseButtonPress and source == self.ui.lineDeviceName:
            self.show_device_list(self.ui.lineDeviceName)
            self.ui.buttonAddDevice.setDefault(True)
            return True
        elif event.type() == QEvent.MouseButtonPress and source == self.ui.lineVariableName:
            self.show_variable_list(self.ui.lineVariableName)
            self.ui.buttonAddVariable.setDefault(True)
            return True
        elif event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionOption1 and self.action_mode in ['set', 'get']:
            self.show_device_list(self.ui.lineActionOption1)
            self.dummyButton.setDefault(True)
            return True
        elif event.type() == QEvent.MouseButtonPress and source == self.ui.lineActionOption2 and self.action_mode in ['set', 'get']:
            self.show_variable_list(self.ui.lineActionOption2, source='action')
            self.dummyButton.setDefault(True)
            return True
        return super().eventFilter(source, event)

    def show_device_list(self, location):
        if location.isEnabled():
            location.selectAll()
            completer = QCompleter(self.database_dict.keys(), self)
            completer.setCompletionMode(QCompleter.PopupCompletion)
            completer.setCaseSensitivity(Qt.CaseSensitive)

            location.setCompleter(completer)
            location.setFocus()
            completer.complete()

    def show_variable_list(self, location, source='device'):
        if location.isEnabled():
            if source == 'device':
                device_name = self.get_selected_device_name()
            elif source == 'action':
                device_name = self.ui.lineActionOption1.text().strip()
            else:
                device_name = None

            if device_name in self.database_dict:
                location.selectAll()
                completer = QCompleter(self.database_dict[device_name].keys(), self)
                completer.setCompletionMode(QCompleter.PopupCompletion)
                completer.setCaseSensitivity(Qt.CaseSensitive)

                location.setCompleter(completer)
                location.setFocus()
                completer.complete()

    def update_device_list(self):
        self.ui.listDevices.clear()

        for device in self.devices_dict:
            self.ui.listDevices.addItem(device)

        self.update_variable_list()

    def add_device(self):
        text = self.ui.lineDeviceName.text().strip()
        if text and text not in self.devices_dict:
            self.devices_dict[text] = get_default_device_dictionary()
            self.update_device_list()

    def remove_device(self):
        selected_device = self.ui.listDevices.selectedItems()
        if not selected_device:
            return
        for selection in selected_device:
            text = selection.text()
            if text in self.devices_dict:
                del self.devices_dict[text]
        self.update_device_list()

    def get_selected_device_name(self):
        selected_device = self.ui.listDevices.selectedItems()
        no_selection = not selected_device
        if no_selection:
            return None

        device_name = None
        for selection in selected_device:
            text = selection.text()
            if text in self.devices_dict:
                device_name = text
        return device_name

    def get_selected_device(self):
        device_name = self.get_selected_device_name()
        if device_name is None:
            return None
        else:
            return self.devices_dict[device_name]

    def update_variable_list(self):
        self.ui.listVariables.clear()

        device = self.get_selected_device()
        enable_variables = device is not None

        self.ui.checkboxSynchronous.setEnabled(enable_variables)
        self.ui.checkboxSaveNonscalar.setEnabled(enable_variables)
        self.ui.buttonAddVariable.setEnabled(enable_variables)
        self.ui.buttonRemoveVariable.setEnabled(enable_variables)
        self.ui.lineVariableName.setEnabled(enable_variables)

        if device is None:
            return

        for variable in device['variable_list']:
            self.ui.listVariables.addItem(variable)
        self.ui.checkboxSynchronous.setChecked(device['synchronous'])
        self.ui.checkboxSaveNonscalar.setChecked(device['save_nonscalar_data'])

    def add_variable(self):
        device = self.get_selected_device()
        if device is not None:
            text = self.ui.lineVariableName.text().strip()
            if text and text not in device['variable_list']:
                if "variable_list" in device:
                    device["variable_list"].append(text)
                else:
                    device["variable_list"] = [text]
                self.update_variable_list()

    def remove_variable(self):
        device = self.get_selected_device()
        if device is not None:
            selected_variable = self.ui.listVariables.selectedItems()
            if not selected_variable:
                return
            for selection in selected_variable:
                text = selection.text()
                if text in device['variable_list']:
                    device["variable_list"].remove(text)
            self.update_variable_list()

    def update_device_checkboxes(self):
        device = self.get_selected_device()
        if device is not None:
            device['synchronous'] = self.ui.checkboxSynchronous.isChecked()
            device['save_nonscalar_data'] = self.ui.checkboxSaveNonscalar.isChecked()
            self.update_variable_list()

    def show_action_list(self):
        self.ui.listActions.clearSelection()
        completer = QCompleter(list_of_actions, self)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.ui.lineActionName.setCompleter(completer)
        self.ui.lineActionName.setFocus()
        completer.complete()

    def generate_action_description(self, action):
        description = "???"
        if action.get("wait") is not None:
            description = f"wait {action['wait']}"
        elif action['action'] == 'execute':
            description = f"execute {action['action_name']}"
        elif action['action'] == 'run':
            description = "run"
        elif action['action'] == 'set':
            description = f"{action['action']} {action['device']}:{action['variable']} {action.get('value')}"
        elif action['action'] == 'get':
            description = f"{action['action']} {action['device']}:{action['variable']} {action.get('expected_value')}"
        return description

    def update_action_list(self, index=None):
        self.ui.listActions.clear()
        self.dummyButton.setDefault(True)
        for item in self.actions_dict['setup']:
            self.ui.listActions.addItem(self.generate_action_description(item))
        self.ui.listActions.addItem("---Scan---")
        for item in self.actions_dict['closeout']:
            self.ui.listActions.addItem(self.generate_action_description(item))
        if index is not None and 0 <= index < self.ui.listActions.count():
            self.ui.listActions.setCurrentRow(index)

    def add_action(self):
        text = self.ui.lineActionName.text().strip()
        if text:
            self.actions_dict['setup'].append(get_new_action(text))
        self.update_action_list()

    def remove_action(self):
        self.get_selected_action(do_remove=True)
        self.update_action_list()

    def get_selected_action(self, do_remove=False):
        selected_action = self.ui.listActions.selectedItems()
        if not selected_action:
            return
        for action in selected_action:
            index = self.ui.listActions.row(action)
            setup_length = len(self.actions_dict['setup'])
            if index < setup_length:
                action_list = self.actions_dict['setup']
            elif index == setup_length:
                return
            else:
                action_list = self.actions_dict['closeout']
                index = index - 1 - setup_length
            if do_remove:
                del action_list[index]
            else:
                return action_list[index]

    def focus_action(self):
        self.ui.lineActionName.clear()
        self.update_action_display()

    def update_action_display(self):
        action = self.get_selected_action()

        if action is None:
            self.action_mode = None
            self.ui.labelActionOption1.setText("")
            self.ui.labelActionOption2.setText("")
            self.ui.labelActionOption3.setText("")
            self.ui.lineActionOption1.setText("")
            self.ui.lineActionOption2.setText("")
            self.ui.lineActionOption3.setText("")
            self.ui.lineActionOption1.setEnabled(False)
            self.ui.lineActionOption2.setEnabled(False)
            self.ui.lineActionOption3.setEnabled(False)
        elif action.get("wait") is not None:
            self.action_mode = 'wait'
            self.ui.labelActionOption1.setText("Wait Time (s):")
            self.ui.lineActionOption1.setEnabled(True)
            self.ui.lineActionOption1.setText(action.get("wait"))
        elif action['action'] == 'execute':
            self.action_mode = 'execute'
            self.ui.labelActionOption1.setText("Action Name:")
            self.ui.lineActionOption1.setEnabled(True)
            self.ui.lineActionOption1.setText(action.get("action_name"))
        elif action['action'] == 'run':
            self.action_mode = 'run'
            self.ui.labelActionOption1.setText("File Location:")
            self.ui.lineActionOption1.setEnabled(True)
            self.ui.lineActionOption1.setText(action.get("file_name"))
            self.ui.labelActionOption2.setText("Class Name:")
            self.ui.lineActionOption2.setEnabled(True)
            self.ui.lineActionOption2.setText(action.get("class_name"))
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
            self.ui.lineActionOption3.setText(action.get("value"))
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
            self.ui.lineActionOption3.setText(action.get("expected_value"))

    def update_action_info(self):
        action = self.get_selected_action()
        if action is None:
            return
        if action.get("wait") is not None:
            action['wait'] = self.ui.lineActionOption1.text().strip()
        elif action['action'] == 'execute':
            action['action_name'] = self.ui.lineActionOption1.text().strip()
        elif action['action'] == 'run':
            action["file_name"] = self.ui.lineActionOption1.text().strip()
            action["class_name"] = self.ui.lineActionOption2.text().strip()
        elif action['action'] == 'set':
            action["device"] = self.ui.lineActionOption1.text().strip()
            action["variable"] = self.ui.lineActionOption2.text().strip()
            action["value"] = self.ui.lineActionOption3.text().strip()
        elif action['action'] == 'get':
            action["device"] = self.ui.lineActionOption1.text().strip()
            action["variable"] = self.ui.lineActionOption2.text().strip()
            action["expected_value"] = self.ui.lineActionOption3.text().strip()

        # TODO There is a weird bug where hitting enter on Option 1 presses "Move Sooner"
        current_selection = self.get_action_list_and_index()
        if current_selection is None:
            return
        action_list, i, index = current_selection
        self.update_action_list(index=index)

    def get_action_list_and_index(self):
        selected_action = self.ui.listActions.selectedItems()
        if not selected_action:
            return
        for action in selected_action:
            absolute_index = self.ui.listActions.row(action)
            setup_length = len(self.actions_dict['setup'])
            if absolute_index < setup_length:
                action_list = self.actions_dict['setup']
                index = absolute_index
            elif absolute_index == setup_length:
                return None, None, absolute_index
            else:
                action_list = self.actions_dict['closeout']
                index = absolute_index - 1 - setup_length
            return action_list, index, absolute_index

    def move_action_sooner(self):
        current_selection = self.get_action_list_and_index()
        if current_selection is None:
            return
        action_list, i, index = current_selection
        if action_list is not None and 0 < i < len(action_list):
            action_list[i], action_list[i-1] = action_list[i-1], action_list[i]
            index = index-1
        self.update_action_list(index=index)

    def move_action_later(self):
        current_selection = self.get_action_list_and_index()
        if current_selection is None:
            return
        action_list, i, index = current_selection
        if action_list is not None and 0 <= i < len(action_list)-1:
            action_list[i], action_list[i + 1] = action_list[i + 1], action_list[i]
            index = index+1
        self.update_action_list(index=index)

    def save_element(self):
        print("Save")

    def close_window(self):
        print("Cancel")
        self.close()

    def open_element(self):
        print("Open")
