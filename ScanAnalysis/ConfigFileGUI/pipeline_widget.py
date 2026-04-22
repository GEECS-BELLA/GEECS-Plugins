"""Drag-and-drop pipeline ordering widget for the Config File GUI.

Provides a ``QListWidget`` with internal-move drag-and-drop so users
can reorder processing steps.  The widget is always rendered at the
bottom of the editor, after all processing sections.

This module is consumed by ``config_editor_window.py``.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional, Type

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


def _format_step_name(name: str) -> str:
    """Convert a snake_case step name to a Title Case display label.

    Parameters
    ----------
    name : str
        A snake_case identifier (e.g. ``"crosshair_masking"``).

    Returns
    -------
    str
        Title-cased string (e.g. ``"Crosshair Masking"``).
    """
    return name.replace("_", " ").title()


class PipelineWidget(QGroupBox):
    """Drag-and-drop widget for reordering processing pipeline steps.

    Each list item represents a processing step whose display name is
    the title-cased version of the enum value.  The canonical order is
    defined by the member order of *step_enum*.

    Parameters
    ----------
    step_enum : Type[Enum]
        The enum class whose members define the available pipeline
        steps (e.g. ``ProcessingStepType`` or ``PipelineStepType``).
    parent : QWidget, optional
        Parent Qt widget.

    Signals
    -------
    orderChanged()
        Emitted whenever the user reorders items via drag-and-drop.
    """

    orderChanged = pyqtSignal()

    def __init__(
        self,
        step_enum: Type[Enum],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__("Processing Pipeline Order", parent)
        self._step_enum = step_enum

        # Build the canonical order from enum definition order
        self._canonical_order: List[str] = [member.value for member in step_enum]

        # --- Layout ---
        layout = QVBoxLayout(self)

        # Status label — shown when using runtime default (Bug 3)
        self._status_label = QLabel(
            "Using runtime default \u2014 pipeline not saved to file"
        )
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(
            "color: gray; font-size: 11px; font-style: italic;"
        )
        layout.addWidget(self._status_label)

        # List widget with drag-and-drop
        self._list_widget = QListWidget()
        self._list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self._list_widget.setDefaultDropAction(Qt.MoveAction)
        layout.addWidget(self._list_widget)

        # Connect the internal model's rowsMoved signal to emit orderChanged
        self._list_widget.model().rowsMoved.connect(self._on_rows_moved)

        # Button row
        button_layout = QHBoxLayout()
        self._reset_btn = QPushButton("Reset to Default Order")
        self._reset_btn.clicked.connect(self.reset_to_default)
        button_layout.addWidget(self._reset_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Help text
        help_label = QLabel(
            "Drag items to reorder processing steps. Only enabled sections are shown."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(help_label)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_rows_moved(self) -> None:
        """Slot connected to the list model's ``rowsMoved`` signal."""
        self.orderChanged.emit()

    def _add_step_item(self, step_value: str) -> None:
        """Add a single step to the end of the list.

        Parameters
        ----------
        step_value : str
            The enum ``.value`` string for the step.
        """
        item = QListWidgetItem(_format_step_name(step_value))
        # Store the raw enum value as item data
        item.setData(Qt.UserRole, step_value)
        self._list_widget.addItem(item)

    def _find_item_row(self, step_value: str) -> int:
        """Return the row index of the item with *step_value*, or -1.

        Parameters
        ----------
        step_value : str
            The enum value to search for.

        Returns
        -------
        int
            Row index, or ``-1`` if not found.
        """
        for row in range(self._list_widget.count()):
            item = self._list_widget.item(row)
            if item is not None and item.data(Qt.UserRole) == step_value:
                return row
        return -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_using_default(self, is_default: bool) -> None:
        """Show or hide the 'using runtime default' status indicator.

        Parameters
        ----------
        is_default : bool
            ``True`` when the pipeline was not present in the source YAML
            and the widget is showing the runtime fallback order.
            ``False`` when the pipeline was explicitly configured in the
            source file or the user has manually reordered steps.
        """
        self._status_label.setVisible(is_default)

    def set_enabled_steps(self, step_names: List[str]) -> None:
        """Update the list to show only the given steps.

        Steps that are already present retain their current position.
        New steps are appended at the end.  Steps not in *step_names*
        are removed.

        Parameters
        ----------
        step_names : list of str
            The enum values of steps that should be visible.
        """
        step_set = set(step_names)

        # Remove steps that are no longer enabled
        rows_to_remove: List[int] = []
        for row in range(self._list_widget.count()):
            item = self._list_widget.item(row)
            if item is not None and item.data(Qt.UserRole) not in step_set:
                rows_to_remove.append(row)
        # Remove in reverse order to keep indices stable
        for row in reversed(rows_to_remove):
            self._list_widget.takeItem(row)

        # Determine which steps are already shown
        current_values = set()
        for row in range(self._list_widget.count()):
            item = self._list_widget.item(row)
            if item is not None:
                current_values.add(item.data(Qt.UserRole))

        # Append new steps that aren't already present
        for name in step_names:
            if name not in current_values:
                self._add_step_item(name)

        self.orderChanged.emit()

    def get_step_order(self) -> List[str]:
        """Return the current ordered list of step names.

        Returns
        -------
        list of str
            Enum values as strings, in the user's chosen order.
        """
        order: List[str] = []
        for row in range(self._list_widget.count()):
            item = self._list_widget.item(row)
            if item is not None:
                order.append(item.data(Qt.UserRole))
        return order

    def set_step_order(self, steps: List[str]) -> None:
        """Set the list to show exactly these steps in this order.

        Parameters
        ----------
        steps : list of str
            Enum values in the desired display order.
        """
        self._list_widget.clear()
        for step_value in steps:
            self._add_step_item(step_value)

    def on_section_enabled_changed(self, section_name: str, enabled: bool) -> None:
        """Slot to connect to ``SectionWidget.sectionEnabledChanged``.

        Maps section names to step names (they are identical in most
        cases) and adds or removes the corresponding pipeline step.

        Parameters
        ----------
        section_name : str
            The section that was toggled (e.g. ``"background"``).
        enabled : bool
            Whether the section is now enabled.
        """
        # Section names map directly to step enum values
        step_value = section_name

        # Verify this is a valid step for the current enum
        valid_values = {m.value for m in self._step_enum}
        if step_value not in valid_values:
            return

        if enabled:
            # Add the step if not already present
            if self._find_item_row(step_value) == -1:
                self._add_step_item(step_value)
                self.orderChanged.emit()
        else:
            # Remove the step
            row = self._find_item_row(step_value)
            if row >= 0:
                self._list_widget.takeItem(row)
                self.orderChanged.emit()

    def reset_to_default(self) -> None:
        """Restore the canonical enum order for all currently-shown steps.

        Only steps that are currently visible are kept; they are
        reordered to match the enum definition order.
        """
        current_values = set(self.get_step_order())
        # Rebuild in canonical order, keeping only currently-shown steps
        ordered = [v for v in self._canonical_order if v in current_values]
        self.set_step_order(ordered)
        self.orderChanged.emit()
