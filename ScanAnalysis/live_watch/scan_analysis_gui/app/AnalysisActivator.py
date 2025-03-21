"""
Analysis Activator Logic

Provides a dialog interface for enabling/disabling various analysis components.

Authors:
Kyle Jensen (kjensen11, kjensen@lbl.gov)
"""
# =============================================================================
# %% imports
from typing import Optional, NamedTuple
from PyQt5 import QtWidgets, QtCore
from operator import attrgetter

from live_watch.scan_analysis_gui.app.gui.AnalysisDialog_ui import Ui_Dialog
# =============================================================================
# %% global variables

# =============================================================================
# %% classes

class ActivatorTuple(NamedTuple):
    """
    Named tuple for storing analyzer configuration.
    
    Attributes:
        analyzer: Name of the analyzer class
        device: Name of the device being analyzed (optional)
        is_active: Whether the analyzer is currently active
    """
    analyzer: str
    device: str = ''
    is_active: bool = True

class AnalysisDialog(QtWidgets.QDialog):
    """
    Dialog for configuring which analyzers are active.
    
    Provides a table interface where users can enable or disable
    various analysis components.
    """
    def __init__(self, analysis_list: list[ActivatorTuple], parent=None) -> None:
        """
        Initialize the analysis dialog.
        
        Args:
            analysis_list: List of analyzer configurations
            parent: Parent widget
        """
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # setup buttons
        self.setup_button_box()
        self.setup_button_activate_all()
        self.setup_button_disable_all()

        # configure the table
        self.table = self.ui.tableWidget
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Analysis (Device)",
                                              "Active"])

        # make first column stretch
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)

        # populate the table with data
        self.analysis_list = analysis_list
        self.populate_table()

    def setup_button_box(self) -> None:
        """
        Configure the dialog button box.
        
        Returns:
            None
        """
        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)

    def setup_button_activate_all(self) -> None:
        """
        Configure the 'Activate All' button.
        
        Returns:
            None
        """
        self.ui.buttonActivateAll.setEnabled(True)
        self.ui.buttonActivateAll.clicked.connect(lambda: self.event_set_all_states(True))

    def setup_button_disable_all(self) -> None:
        """
        Configure the 'Disable All' button.
        
        Returns:
            None
        """
        self.ui.buttonDisableAll.setEnabled(True)
        self.ui.buttonDisableAll.clicked.connect(lambda: self.event_set_all_states(False))

    def event_set_all_states(self, state_to_set: bool) -> None:
        """
        Set all checkboxes to either checked or unchecked state.
        
        Returns:
            None
        """
        # block table signals temporarily
        self.table.blockSignals(True)

        # set all check boxes to checked (i.e. active)
        for row in range(self.table.rowCount()):
            # get checkbox object
            cell_widget = self.table.cellWidget(row, 1)
            if cell_widget:
                checkbox = cell_widget.findChild(QtWidgets.QCheckBox)
                if checkbox:
                    # block signal, modify state, unblock signal
                    checkbox.blockSignals(True)
                    checkbox.setChecked(state_to_set)
                    checkbox.blockSignals(False)

        # unblock table signals
        self.table.blockSignals(False)

        # force visual refresh
        self.table.viewport().update()

    def create_checkbox_widget(self, checkbox) -> QtWidgets.QWidget:
        """
        Create a widget to center the checkbox in a table cell.
        
        Args:
            checkbox: The checkbox to center
            
        Returns:
            QWidget containing the centered checkbox
        """
        # create a widget to center the checkbox
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.addWidget(checkbox)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return widget

    def populate_table(self) -> None:
        """
        Populate the table with analysis items.
        
        Returns:
            None
        """
        # clear existing rows
        self.table.setRowCount(0)

        # sort the analysis list alphabetically
        sorted_analysis = sorted(self.analysis_list, key=attrgetter('device', 'analyzer'))

        # add rows dynamically
        for i, analysis in enumerate(sorted_analysis):
            # append new row
            self.table.insertRow(i)

            # organize text label
            analysis_label = self.construct_analysis_label(analysis)

            # set analysis name (non-editable)
            item = QtWidgets.QTableWidgetItem(analysis_label)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(i, 0, item)

            # add 'Active' checkbox
            active_checkbox = QtWidgets.QCheckBox()
            active_checkbox.setChecked(analysis.is_active)
            self.table.setCellWidget(i, 1, self.create_checkbox_widget(active_checkbox))

    def get_analysis_states(self) -> list[ActivatorTuple]:
        """
        Get the current state of all analysis items from the table.
        
        Returns:
            List of ActivatorTuple with updated is_active states
            
        Raises:
            NoMatchingAnalysisError: If a matching table row cannot be found
        """
        for ind, analysis in enumerate(self.analysis_list):
            # create analysis label
            label = self.construct_analysis_label(analysis)

            # find corresponding table row
            row_index = self.find_corresponding_table_row(label)

            if row_index >= 0:
                # get checkbox state
                state = self.table.cellWidget(row_index, 1).findChild(QtWidgets.QCheckBox).isChecked()
                self.analysis_list[ind] = analysis._replace(is_active=state)
            else:
                raise NoMatchingAnalysisError('Could not find matching Analysis Activator table element.')

        return self.analysis_list

    def find_corresponding_table_row(self, analysis_label: str) -> int:
        """
        Find the row index where column 0 contains the specified text.
        
        Args:
            analysis_label: The text to search for in column 0
            
        Returns:
            int: The row index if found, or -1 if not found
        """
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == analysis_label:
                return row
        return -1

    @staticmethod
    def construct_analysis_label(analysis: NamedTuple) -> str:
        """
        Construct a display label for an analysis item.
        
        Args:
            analysis: The analysis item
            
        Returns:
            str: Formatted label string
        """
        if analysis.device:
            return f"{analysis.device} ({analysis.analyzer})"
        else:
            return f"{analysis.analyzer}"
# =============================================================================
# %% error handling

class CustomError(Exception):
    def __init__(self, custom_message: str = None) -> None:
        """
        Initialize custom error with optional message.
        
        Args:
            custom_message (str, optional): Custom message to prepend
        """
        super().__init__(custom_message)
        self.custom_message = custom_message
        
    def __str__(self) -> str:
        """
        Returns formatted error message.
        
        Returns:
            str: Formatted error message including exception type and details
        """
        return (
            f"ERROR: {self.custom_message if self.custom_message else ''}\n"
            f"Type: {self.get_class_name()}"
        )

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__

class NoMatchingAnalysisError(CustomError):
    """Error raised when analysis is already running."""
    pass

# =============================================================================
