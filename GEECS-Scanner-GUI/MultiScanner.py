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

    def populate_preset_list(self):
        self.ui.listAvailablePresets.clear()
        for preset in self.main_window.load_preset_list():
            self.ui.listAvailablePresets.addItem(preset)

    def apply_preset_to_main_window(self):
        self.main_window.apply_preset_from_selected_element(self.ui.listAvailablePresets.selectedItems())

    def close_window(self):
        self.main_window.exit_multiscan_mode()
        self.close()

    def closeEvent(self, event):
        self.main_window.exit_multiscan_mode()
        event.accept()
