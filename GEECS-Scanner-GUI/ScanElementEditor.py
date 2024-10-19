import sys
from PyQt5.QtWidgets import QDialog, QDialogButtonBox
from ScanElementEditor_ui import Ui_Dialog


def get_default_device_dictionary():
    return {
        'variable_list': ['timestamp'],
        'synchronous': True,
        'save_nonscalar_data': True
    }


class ScanElementEditor(QDialog):
    def __init__(self):
        super().__init__()

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.devices_dict = {}

        self.ui.buttonAddDevice.clicked.connect(self.add_device)
        self.ui.buttonRemoveDevice.clicked.connect(self.remove_device)
        self.ui.listDevices.itemSelectionChanged.connect(self.update_variable_list)

        self.ui.buttonWindowSave.clicked.connect(self.save_element)
        self.ui.buttonWindowCancel.clicked.connect(self.close_window)
        self.ui.buttonWindowLoad.clicked.connect(self.open_element)

        self.update_device_list()

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

    def update_variable_list(self):
        self.ui.listVariables.clear()

        selected_device = self.ui.listDevices.selectedItems()
        no_selection = not selected_device
        self.ui.checkboxSynchronous.setEnabled(not no_selection)
        self.ui.checkboxSaveNonscalar.setEnabled(not no_selection)
        self.ui.buttonAddVariable.setEnabled(not no_selection)
        self.ui.buttonRemoveVariable.setEnabled(not no_selection)
        self.ui.lineVariableName.setEnabled(not no_selection)

        if no_selection:
            return
        for selection in selected_device:
            text = selection.text()
            if text in self.devices_dict:
                device = self.devices_dict[text]
                for variable in device['variable_list']:
                    self.ui.listVariables.addItem(variable)
                self.ui.checkboxSynchronous.setChecked(device['synchronous'])
                self.ui.checkboxSaveNonscalar.setChecked(device['save_nonscalar_data'])

    def save_element(self):
        print("Save")

    def close_window(self):
        print("Cancel")
        self.close()

    def open_element(self):
        print("Open")
