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

    def close_window(self):
        self.main_window.exit_multiscan_mode()
        self.close()

    def closeEvent(self, event):
        self.main_window.exit_multiscan_mode()
        event.accept()
