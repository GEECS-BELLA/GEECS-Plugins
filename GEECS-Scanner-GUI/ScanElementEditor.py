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

        self.ui.checkboxSynchronous.clicked.connect(self.update_device_checkboxes)
        self.ui.checkboxSaveNonscalar.clicked.connect(self.update_device_checkboxes)

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

    def get_selected_device(self):
        selected_device = self.ui.listDevices.selectedItems()
        no_selection = not selected_device
        if no_selection:
            return None

        device = None
        for selection in selected_device:
            text = selection.text()
            if text in self.devices_dict:
                device = self.devices_dict[text]
        return device

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

    def update_device_checkboxes(self):
        device = self.get_selected_device()
        if device is not None:
            device['synchronous'] = self.ui.checkboxSynchronous.isChecked()
            device['save_nonscalar_data'] = self.ui.checkboxSaveNonscalar.isChecked()
            self.update_variable_list()

    def save_element(self):
        print("Save")

    def close_window(self):
        print("Cancel")
        self.close()

    def open_element(self):
        print("Open")
