from PyQt5.QtWidgets import QWidget
from MultiScanner_ui import Ui_Form


class MultiScanner(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Multi-Scanner")

        self.ui.buttonExit.clicked.connect(self.close_window)

        self.ui.buttonRefreshPresetList.clicked.connect(self.populate_preset_list)
        self.populate_preset_list()
        self.ui.listAvailablePresets.itemDoubleClicked.connect(self.apply_preset_to_main_window)

        self.element_preset_list = []
        self.scan_preset_list = []

        self.ui.buttonAddElement.clicked.connect(self.add_element_preset)
        self.ui.buttonRemoveElement.clicked.connect(self.remove_element_preset)
        self.ui.buttonAddScan.clicked.connect(self.add_scan_preset)
        self.ui.buttonRemoveScan.clicked.connect(self.remove_scan_preset)

        self.ui.checkBoxEnableScanList.clicked.connect(self.toggle_split_preset_mode)
        self.toggle_split_preset_mode()

    def populate_preset_list(self):
        self.ui.listAvailablePresets.clear()
        for preset in self.main_window.load_preset_list():
            self.ui.listAvailablePresets.addItem(preset)

    def apply_preset_to_main_window(self):
        self.main_window.apply_preset_from_selected_element(self.ui.listAvailablePresets.selectedItems())

    def refresh_multiscan_lists(self):
        self.ui.listElementPresets.clear()
        self.ui.listScanPresets.clear()
        number_scans = max(len(self.element_preset_list), len(self.scan_preset_list))
        for i in range(number_scans):
            try: self.ui.listElementPresets.addItem(f"{i}. {self.element_preset_list[i]}")
            except IndexError: self.ui.listElementPresets.addItem(f"{i}.")
            if self.ui.checkBoxEnableScanList.isChecked():
                try: self.ui.listScanPresets.addItem(f"{i}. {self.scan_preset_list[i]}")
                except IndexError:  self.ui.listScanPresets.addItem(f"{i}.")

    def add_element_preset(self):
        self.add_to_list(self.ui.listAvailablePresets.selectedItems(), self.element_preset_list)

    def remove_element_preset(self):
        self.remove_from_list(self.ui.listElementPresets, self.element_preset_list)

    def add_scan_preset(self):
        self.add_to_list(self.ui.listAvailablePresets.selectedItems(), self.scan_preset_list)

    def remove_scan_preset(self):
        self.remove_from_list(self.ui.listScanPresets, self.scan_preset_list)

    def add_to_list(self, selected_items, target_list):
        if not selected_items:
            return
        for selection in selected_items:
            target_list.append(selection.text())
        self.refresh_multiscan_lists()

    def remove_from_list(self, list_widget, target_list):
        selected_items = list_widget.selectedItems()
        if not selected_items:
            return
        for selection in selected_items:
            index = list_widget.row(selection)
            if index < len(target_list):
                del target_list[list_widget.row(selection)]
        self.refresh_multiscan_lists()

    def toggle_split_preset_mode(self):
        is_enabled = self.ui.checkBoxEnableScanList.isChecked()
        self.ui.listScanPresets.setEnabled(is_enabled)
        self.ui.buttonAddScan.setEnabled(is_enabled)
        self.ui.buttonRemoveScan.setEnabled(is_enabled)
        self.ui.buttonScanSooner.setEnabled(is_enabled)
        self.ui.buttonScanLater.setEnabled(is_enabled)
        self.ui.buttonCopyElementToScan.setEnabled(is_enabled)
        self.ui.buttonCopyScanToElement.setEnabled(is_enabled)
        self.ui.buttonCopyRowScan.setEnabled(is_enabled)
        self.ui.buttonCopyRowElement.setEnabled(is_enabled)

        self.refresh_multiscan_lists()

    def close_window(self):
        self.main_window.exit_multiscan_mode()
        self.close()

    def closeEvent(self, event):
        self.main_window.exit_multiscan_mode()
        event.accept()
