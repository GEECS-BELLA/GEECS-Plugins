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

        self.ui.buttonElementSooner.clicked.connect(self.move_element_sooner)
        self.ui.buttonElementLater.clicked.connect(self.move_element_later)
        self.ui.buttonScanSooner.clicked.connect(self.move_scan_sooner)
        self.ui.buttonScanLater.clicked.connect(self.move_scan_later)

        self.ui.buttonCopyElementToScan.clicked.connect(self.copy_list_element_to_scan)
        self.ui.buttonCopyScanToElement.clicked.connect(self.copy_list_scan_to_element)

    def populate_preset_list(self):
        self.ui.listAvailablePresets.clear()
        for preset in self.main_window.load_preset_list():
            self.ui.listAvailablePresets.addItem(preset)

    def apply_preset_to_main_window(self):
        self.main_window.apply_preset_from_selected_element(self.ui.listAvailablePresets.selectedItems())

    def refresh_multiscan_lists(self, list_widget=None, index=None):
        self.ui.listElementPresets.clear()
        self.ui.listScanPresets.clear()
        number_scans = max(len(self.element_preset_list), len(self.scan_preset_list))
        for i in range(number_scans):
            try: self.ui.listElementPresets.addItem(f"{i+1}:  {self.element_preset_list[i]}")
            except IndexError: self.ui.listElementPresets.addItem(f"{i+1}.")
            if self.ui.checkBoxEnableScanList.isChecked():
                try: self.ui.listScanPresets.addItem(f"{i+1}:  {self.scan_preset_list[i]}")
                except IndexError:  self.ui.listScanPresets.addItem(f"{i+1}.")
        if index is not None and list_widget is not None:
            list_widget.setCurrentRow(index)
            list_widget.setCurrentRow(index)

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

    def move_element_sooner(self):
        self.move_ordering(list_widget=self.ui.listElementPresets, target_list=self.element_preset_list, sooner=True)

    def move_element_later(self):
        self.move_ordering(list_widget=self.ui.listElementPresets, target_list=self.element_preset_list, later=True)

    def move_scan_sooner(self):
        self.move_ordering(list_widget=self.ui.listScanPresets, target_list=self.scan_preset_list, sooner=True)

    def move_scan_later(self):
        self.move_ordering(list_widget=self.ui.listScanPresets, target_list=self.scan_preset_list, later=True)

    def move_ordering(self, list_widget, target_list, sooner=False, later=False):
        """Moves the selected action to an earlier or later position in the same list"""
        selected_items = list_widget.selectedItems()
        if not selected_items:
            return
        for selection in selected_items:
            i = list_widget.row(selection)
        if sooner and 0 < i < len(target_list):
            target_list[i], target_list[i - 1] = target_list[i - 1], target_list[i]
            i = i - 1
        if later and 0 <= i < len(target_list) - 1:
            target_list[i], target_list[i + 1] = target_list[i + 1], target_list[i]
            i = i + 1
        self.refresh_multiscan_lists(list_widget=list_widget, index=i)

    def copy_list_element_to_scan(self):
        self.scan_preset_list = self.element_preset_list.copy()
        self.refresh_multiscan_lists()

    def copy_list_scan_to_element(self):
        self.element_preset_list = self.scan_preset_list.copy()
        self.refresh_multiscan_lists()

    def close_window(self):
        self.main_window.exit_multiscan_mode()
        self.close()

    def closeEvent(self, event):
        self.main_window.exit_multiscan_mode()
        event.accept()
