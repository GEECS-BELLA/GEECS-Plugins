from __future__ import annotations
from PyQt5.QtWidgets import QDialog
from .gui.ScanDeviceOrder_ui import Ui_Dialog


class ScanVariableOrder(QDialog):
    """
    Small Dialog window that displays the two lists of interest:  scan variables and composite variable.  Two buttons
    are available to adjust the ordering of these lists.  Then the user can save and return to the previous dialog, or
    cancel and make no changes.  TODO do I have write access to the files here or instead pass back the new ordered list
    """

    def __init__(self, stylesheet, scan_variable_list, composite_variable_list):
        """
        Initializes the GUI

        :param stylesheet: Stylesheet to use, taken from parent dialog window
        :param scan_variable_list:
        :param composite_variable_list:
        """
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.scan_variable_list = scan_variable_list
        self.composite_variable_list = composite_variable_list

        self.ui.buttonClose.clicked.connect(self.close)

        self.setStyleSheet(stylesheet)
