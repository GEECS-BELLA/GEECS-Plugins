"""
Analysis Activator Logic.

Provides a dialog interface for enabling/disabling various analysis components.

Authors
-------
Kyle Jensen (kjensen11, kjensen@lbl.gov)

Notes
-----
This module supplies:
- `ActivatorTuple`: a lightweight container describing an analyzer row (name/device/state)
- `AnalysisDialog`: a PyQt5 dialog that renders those rows in a table with per‑row checkboxes
"""

# =============================================================================
# %% imports
from typing import NamedTuple
from PyQt5 import QtWidgets, QtCore
from operator import attrgetter

from live_watch.scan_analysis_gui.app.gui.AnalysisDialog_ui import Ui_Dialog
# =============================================================================
# %% global variables

# =============================================================================
# %% classes


class ActivatorTuple(NamedTuple):
    """
    Lightweight container for analyzer configuration shown in the dialog.

    Attributes
    ----------
    analyzer : str
        Name of the analyzer class.
    device : str, optional
        Device associated with the analyzer (empty string if N/A).
    is_active : bool, default True
        Whether the analyzer is currently enabled.
    """

    analyzer: str
    device: str = ""
    is_active: bool = True


class AnalysisDialog(QtWidgets.QDialog):
    """
    Dialog for enabling/disabling analyzers prior to running scan analysis.

    The dialog presents a two‑column table:
      1. "Analysis (Device)" — text label combining device (if any) and analyzer class
      2. "Active" — centered checkbox indicating whether the analyzer is enabled

    Parameters
    ----------
    analysis_list : list[ActivatorTuple]
        Analyzers to display/edit.
    parent : QWidget, optional
        Parent widget for the dialog.
    """

    def __init__(self, analysis_list: list[ActivatorTuple], parent=None) -> None:
        """Build the dialog, configure the table, and populate rows from `analysis_list`."""
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
        self.table.setHorizontalHeaderLabels(["Analysis (Device)", "Active"])

        # make first column stretch
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)

        # populate the table with data
        self.analysis_list = analysis_list
        self.populate_table()

    def setup_button_box(self) -> None:
        """
        Wire the OK/Cancel button box to accept/reject.

        Returns
        -------
        None
        """
        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)

    def setup_button_activate_all(self) -> None:
        """
        Wire the 'Activate All' button to check all rows.

        Returns
        -------
        None
        """
        self.ui.buttonActivateAll.setEnabled(True)
        self.ui.buttonActivateAll.clicked.connect(
            lambda: self.event_set_all_states(True)
        )

    def setup_button_disable_all(self) -> None:
        """
        Wire the 'Disable All' button to uncheck all rows.

        Returns
        -------
        None
        """
        self.ui.buttonDisableAll.setEnabled(True)
        self.ui.buttonDisableAll.clicked.connect(
            lambda: self.event_set_all_states(False)
        )

    def event_set_all_states(self, state_to_set: bool) -> None:
        """
        Set all checkboxes to a given state.

        Parameters
        ----------
        state_to_set : bool
            True to check all, False to uncheck all.

        Returns
        -------
        None
        """
        # block table signals temporarily
        self.table.blockSignals(True)

        # set all check boxes to checked/unchecked
        for row in range(self.table.rowCount()):
            cell_widget = self.table.cellWidget(row, 1)
            if cell_widget:
                checkbox = cell_widget.findChild(QtWidgets.QCheckBox)
                if checkbox:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(state_to_set)
                    checkbox.blockSignals(False)

        # unblock table signals and force repaint
        self.table.blockSignals(False)
        self.table.viewport().update()

    def create_checkbox_widget(self, checkbox) -> QtWidgets.QWidget:
        """
        Center a checkbox inside a table cell by placing it in a small widget.

        Parameters
        ----------
        checkbox : QCheckBox
            The checkbox to center.

        Returns
        -------
        QWidget
            The container widget with centered checkbox.
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.addWidget(checkbox)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return widget

    def populate_table(self) -> None:
        """
        Populate the table with analyzer rows from `self.analysis_list`.

        Notes
        -----
        Rows are sorted by `(device, analyzer)`. The first column is read‑only text,
        and the second column hosts a centered checkbox.

        Returns
        -------
        None
        """
        # clear existing rows
        self.table.setRowCount(0)

        # sort the analysis list alphabetically
        sorted_analysis = sorted(
            self.analysis_list, key=attrgetter("device", "analyzer")
        )

        # add rows dynamically
        for i, analysis in enumerate(sorted_analysis):
            # append new row
            self.table.insertRow(i)

            # label "Device (Analyzer)" or just "Analyzer"
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
        Read the current checkbox states back into `self.analysis_list`.

        Returns
        -------
        list[ActivatorTuple]
            Updated list with `is_active` reflecting the UI state.

        Raises
        ------
        NoMatchingAnalysisError
            If a corresponding row cannot be located for an entry.
        """
        for ind, analysis in enumerate(self.analysis_list):
            # create analysis label
            label = self.construct_analysis_label(analysis)

            # find corresponding table row
            row_index = self.find_corresponding_table_row(label)

            if row_index >= 0:
                # get checkbox state
                state = (
                    self.table.cellWidget(row_index, 1)
                    .findChild(QtWidgets.QCheckBox)
                    .isChecked()
                )
                self.analysis_list[ind] = analysis._replace(is_active=state)
            else:
                raise NoMatchingAnalysisError(
                    "Could not find matching Analysis Activator table element."
                )

        return self.analysis_list

    def find_corresponding_table_row(self, analysis_label: str) -> int:
        """
        Find the row index whose first column exactly matches `analysis_label`.

        Parameters
        ----------
        analysis_label : str
            The label to search for in column 0.

        Returns
        -------
        int
            Row index if found, otherwise -1.
        """
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == analysis_label:
                return row
        return -1

    @staticmethod
    def construct_analysis_label(analysis: NamedTuple) -> str:
        """
        Construct a display label from an analyzer tuple.

        Parameters
        ----------
        analysis : NamedTuple
            Expected to have fields `device` and `analyzer`.

        Returns
        -------
        str
            `"Device (Analyzer)"` if `device` is non‑empty, else `"Analyzer"`.
        """
        if analysis.device:
            return f"{analysis.device} ({analysis.analyzer})"
        else:
            return f"{analysis.analyzer}"


# =============================================================================
# %% error handling


class CustomError(Exception):
    """
    Base class for custom dialog errors with consistent formatting.

    Parameters
    ----------
    custom_message : str, optional
        Additional context to include in the message.
    """

    def __init__(self, custom_message: str = None) -> None:
        super().__init__(custom_message)
        self.custom_message = custom_message

    def __str__(self) -> str:
        """
        Return a formatted error message including the class name.

        Returns
        -------
        str
        """
        return (
            f"ERROR: {self.custom_message if self.custom_message else ''}\n"
            f"Type: {self.get_class_name()}"
        )

    @classmethod
    def get_class_name(cls) -> str:
        """Return the error class name."""
        return cls.__name__


class NoMatchingAnalysisError(CustomError):
    """Raised when the dialog cannot match an analyzer entry to a table row."""

    pass


# =============================================================================
