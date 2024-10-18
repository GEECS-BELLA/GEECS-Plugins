import sys
from PyQt5.QtWidgets import QDialog, QDialogButtonBox
from ScanElementEditor_ui import Ui_Dialog

class ScanElementEditor(QDialog):
    def __init__(self):
        super().__init__()

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.buttonBox.accepted.connect(self.save)
        self.ui.buttonBox.rejected.connect(self.cancel)
        self.ui.buttonBox.clicked.connect(self.handle_button_click)

    def handle_button_click(self, button):
        role = self.ui.buttonBox.buttonRole(button)
        if role == QDialogButtonBox.ActionRole and button.text() == "Open":
            self.open()

    def save(self):
        print("Save")

    def cancel(self):
        print("Cancel")
        self.close()

    def open(self):
        print("Open")
