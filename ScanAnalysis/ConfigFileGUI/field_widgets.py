"""Field widget factory for the Config File GUI.

Maps Pydantic ``FieldInfo`` metadata to appropriate PyQt5 widgets.
Each widget knows how to get/set values in the types expected by
Pydantic and can display inline validation errors.

This module is consumed by ``section_widget.py`` which builds form
sections from Pydantic sub-models.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin

import yaml
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Sentinel re-exported so callers don't need a pydantic import
try:
    from pydantic_core import PydanticUndefined
except ImportError:  # pydantic v1 fallback
    from pydantic.fields import Undefined as PydanticUndefined  # type: ignore[attr-defined,no-redef]


# ---------------------------------------------------------------------------
# Helper: extract numeric constraints from FieldInfo.metadata
# ---------------------------------------------------------------------------


def extract_constraints(field_info: FieldInfo) -> dict:
    """Extract numeric constraints from Pydantic ``FieldInfo.metadata``.

    Pydantic v2 stores constraints as ``annotated_types`` objects
    (e.g. ``Ge(ge=0)``, ``Lt(lt=100)``) inside the *metadata* list
    attached to each ``FieldInfo``.  This helper iterates through
    them and returns a flat dictionary of recognised constraint names.

    Parameters
    ----------
    field_info : FieldInfo
        The Pydantic field descriptor to inspect.

    Returns
    -------
    dict
        Dictionary with any of the keys ``ge``, ``gt``, ``le``, ``lt``,
        ``min_length``, ``max_length`` that were found, mapped to their
        numeric values.
    """
    constraints: dict = {}
    for item in getattr(field_info, "metadata", []):
        for attr in ("ge", "gt", "le", "lt", "min_length", "max_length"):
            if hasattr(item, attr):
                constraints[attr] = getattr(item, attr)
    return constraints


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseFieldWidget(QWidget):
    """Abstract base class for all Pydantic-field widgets.

    Provides the common layout (label + input + error label) and the
    interface that ``SectionWidget`` relies on.

    Parameters
    ----------
    field_name : str
        The Pydantic field name this widget represents.
    field_info : FieldInfo
        Pydantic field descriptor (carries description, default, metadata).
    parent : QWidget, optional
        Parent Qt widget.

    Signals
    -------
    valueChanged
        Emitted whenever the user changes the widget value.
    """

    valueChanged = pyqtSignal()

    def __init__(
        self,
        field_name: str,
        field_info: FieldInfo,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._field_name = field_name
        self._field_info = field_info

        # --- outer vertical layout (row + error) ---
        self._outer_layout = QVBoxLayout(self)
        self._outer_layout.setContentsMargins(0, 2, 0, 2)

        # --- main row: label | input ---
        self._row_layout = QHBoxLayout()
        self._label = QLabel(field_name)
        self._label.setMinimumWidth(140)
        self._row_layout.addWidget(self._label)

        self._outer_layout.addLayout(self._row_layout)

        # --- error label (hidden by default) ---
        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red; font-size: 11px;")
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        self._outer_layout.addWidget(self._error_label)

        # Tooltip from field description
        description = getattr(field_info, "description", None) or ""
        if description:
            self.setToolTip(description)

    # -- public interface --------------------------------------------------

    @property
    def field_name(self) -> str:
        """The Pydantic field name this widget represents."""
        return self._field_name

    def get_value(self) -> Any:
        """Return the current widget value in the type expected by Pydantic.

        Raises
        ------
        NotImplementedError
            Subclasses must override this method.
        """
        raise NotImplementedError

    def set_value(self, value: Any) -> None:
        """Set the widget from a Pydantic-compatible value.

        Parameters
        ----------
        value : Any
            The value to display in the widget.

        Raises
        ------
        NotImplementedError
            Subclasses must override this method.
        """
        raise NotImplementedError

    def set_error(self, message: Optional[str] = None) -> None:
        """Show or clear a validation error message.

        Parameters
        ----------
        message : str, optional
            Error text to display.  Pass ``None`` or empty string to clear.
        """
        if message:
            self._error_label.setText(message)
            self._error_label.show()
        else:
            self._error_label.setText("")
            self._error_label.hide()


# ---------------------------------------------------------------------------
# Concrete widget: str
# ---------------------------------------------------------------------------


class StringFieldWidget(BaseFieldWidget):
    """Widget for ``str`` fields using a ``QLineEdit``.

    Parameters
    ----------
    field_name : str
        Pydantic field name.
    field_info : FieldInfo
        Pydantic field descriptor.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        field_name: str,
        field_info: FieldInfo,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(field_name, field_info, parent)
        self._line_edit = QLineEdit()
        description = getattr(field_info, "description", None) or ""
        if description:
            self._line_edit.setPlaceholderText(description)
        self._row_layout.addWidget(self._line_edit, stretch=1)
        self._line_edit.textChanged.connect(self.valueChanged.emit)

        # Apply default
        default = getattr(field_info, "default", PydanticUndefined)
        if default is not PydanticUndefined and default is not None:
            self._line_edit.setText(str(default))

    def get_value(self) -> str:
        """Return the current text."""
        return self._line_edit.text()

    def set_value(self, value: Any) -> None:
        """Set the line-edit text.

        Parameters
        ----------
        value : Any
            Converted to ``str`` before display.
        """
        self._line_edit.setText(str(value) if value is not None else "")


# ---------------------------------------------------------------------------
# Concrete widget: int
# ---------------------------------------------------------------------------


class IntFieldWidget(BaseFieldWidget):
    """Widget for ``int`` fields using a ``QSpinBox``.

    Min/max are derived from Pydantic ``ge``, ``gt``, ``le``, ``lt``
    constraints when available; otherwise defaults to -999999..999999.

    Parameters
    ----------
    field_name : str
        Pydantic field name.
    field_info : FieldInfo
        Pydantic field descriptor.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        field_name: str,
        field_info: FieldInfo,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(field_name, field_info, parent)
        self._spin = QSpinBox()

        constraints = extract_constraints(field_info)
        min_val = -999999
        max_val = 999999
        if "ge" in constraints:
            min_val = int(constraints["ge"])
        elif "gt" in constraints:
            min_val = int(constraints["gt"]) + 1
        if "le" in constraints:
            max_val = int(constraints["le"])
        elif "lt" in constraints:
            max_val = int(constraints["lt"]) - 1

        self._spin.setRange(min_val, max_val)
        self._row_layout.addWidget(self._spin, stretch=1)
        self._spin.valueChanged.connect(self.valueChanged.emit)

        # Apply default
        default = getattr(field_info, "default", PydanticUndefined)
        if default is not PydanticUndefined and default is not None:
            self._spin.setValue(int(default))

    def get_value(self) -> int:
        """Return the current spin-box value."""
        return self._spin.value()

    def set_value(self, value: Any) -> None:
        """Set the spin-box value.

        Parameters
        ----------
        value : Any
            Converted to ``int`` before setting.
        """
        if value is not None:
            self._spin.setValue(int(value))


# ---------------------------------------------------------------------------
# Concrete widget: float
# ---------------------------------------------------------------------------


class FloatFieldWidget(BaseFieldWidget):
    """Widget for ``float`` fields using a ``QDoubleSpinBox``.

    Min/max are derived from Pydantic constraints; decimals default to 6.
    Default range is -1e12 to 1e12 when no constraints are present.

    Parameters
    ----------
    field_name : str
        Pydantic field name.
    field_info : FieldInfo
        Pydantic field descriptor.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        field_name: str,
        field_info: FieldInfo,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(field_name, field_info, parent)
        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(6)

        constraints = extract_constraints(field_info)
        min_val = -1e12
        max_val = 1e12
        if "ge" in constraints:
            min_val = float(constraints["ge"])
        elif "gt" in constraints:
            min_val = float(constraints["gt"])
        if "le" in constraints:
            max_val = float(constraints["le"])
        elif "lt" in constraints:
            max_val = float(constraints["lt"])

        self._spin.setRange(min_val, max_val)
        self._row_layout.addWidget(self._spin, stretch=1)
        self._spin.valueChanged.connect(self.valueChanged.emit)

        # Apply default
        default = getattr(field_info, "default", PydanticUndefined)
        if default is not PydanticUndefined and default is not None:
            self._spin.setValue(float(default))

    def get_value(self) -> float:
        """Return the current spin-box value."""
        return self._spin.value()

    def set_value(self, value: Any) -> None:
        """Set the spin-box value.

        Parameters
        ----------
        value : Any
            Converted to ``float`` before setting.
        """
        if value is not None:
            self._spin.setValue(float(value))


# ---------------------------------------------------------------------------
# Concrete widget: bool
# ---------------------------------------------------------------------------


class BoolFieldWidget(BaseFieldWidget):
    """Widget for ``bool`` fields using a ``QCheckBox``.

    Parameters
    ----------
    field_name : str
        Pydantic field name.
    field_info : FieldInfo
        Pydantic field descriptor.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        field_name: str,
        field_info: FieldInfo,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(field_name, field_info, parent)
        self._checkbox = QCheckBox()
        self._row_layout.addWidget(self._checkbox, stretch=1)
        self._checkbox.stateChanged.connect(self.valueChanged.emit)

        # Apply default
        default = getattr(field_info, "default", PydanticUndefined)
        if default is not PydanticUndefined and default is not None:
            self._checkbox.setChecked(bool(default))

    def get_value(self) -> bool:
        """Return whether the checkbox is checked."""
        return self._checkbox.isChecked()

    def set_value(self, value: Any) -> None:
        """Set the checkbox state.

        Parameters
        ----------
        value : Any
            Interpreted as a boolean.
        """
        self._checkbox.setChecked(bool(value) if value is not None else False)


# ---------------------------------------------------------------------------
# Concrete widget: Enum
# ---------------------------------------------------------------------------


class EnumFieldWidget(BaseFieldWidget):
    """Widget for ``Enum`` fields using a ``QComboBox``.

    The combo box is populated with the *values* of the enum members
    (not the member names), which matches how Pydantic serialises
    ``str``-enums to YAML.

    Parameters
    ----------
    field_name : str
        Pydantic field name.
    field_info : FieldInfo
        Pydantic field descriptor.
    enum_class : type
        The ``Enum`` subclass whose members populate the combo box.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        field_name: str,
        field_info: FieldInfo,
        enum_class: Type[Enum],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(field_name, field_info, parent)
        self._enum_class = enum_class
        self._combo = QComboBox()

        for member in enum_class:
            self._combo.addItem(str(member.value), member)

        self._row_layout.addWidget(self._combo, stretch=1)
        self._combo.currentIndexChanged.connect(self.valueChanged.emit)

        # Apply default
        default = getattr(field_info, "default", PydanticUndefined)
        if default is not PydanticUndefined and default is not None:
            self.set_value(default)

    def get_value(self) -> Any:
        """Return the value of the currently selected enum member.

        Returns the plain ``.value`` attribute (typically a ``str``) rather
        than the ``Enum`` member itself, so that downstream YAML
        serialization produces clean scalar output instead of
        ``!!python/object/apply:`` tags.

        Returns
        -------
        Any
            The ``.value`` of the selected enum member, or ``None`` if
            nothing is selected.
        """
        member = self._combo.currentData()
        if member is None:
            return None
        return member.value

    def set_value(self, value: Any) -> None:
        """Set the combo box to the given enum member or value.

        Parameters
        ----------
        value : Any
            An enum member, its ``.value``, or a string representation.
        """
        if value is None:
            return
        # If it's already an enum member, use its value for lookup
        if isinstance(value, Enum):
            value = value.value
        for i in range(self._combo.count()):
            member = self._combo.itemData(i)
            if member is not None and (
                member.value == value or str(member.value) == str(value)
            ):
                self._combo.setCurrentIndex(i)
                return


# ---------------------------------------------------------------------------
# Concrete widget: Optional[X] wrapper
# ---------------------------------------------------------------------------


class OptionalFieldWidget(BaseFieldWidget):
    """Wraps another field widget with an enable/disable checkbox.

    When the checkbox is unchecked the inner widget is disabled and
    :meth:`get_value` returns ``None``.  When checked, it delegates
    to the inner widget.

    Parameters
    ----------
    field_name : str
        Pydantic field name.
    field_info : FieldInfo
        Pydantic field descriptor.
    inner_widget : BaseFieldWidget
        The widget for the non-``None`` type.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        field_name: str,
        field_info: FieldInfo,
        inner_widget: BaseFieldWidget,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(field_name, field_info, parent)
        self._inner = inner_widget

        # Remove the inner widget's own label (we already have one)
        inner_label = inner_widget._label
        inner_label.hide()

        self._enable_cb = QCheckBox("Enabled")
        self._row_layout.addWidget(self._enable_cb)
        self._row_layout.addWidget(self._inner, stretch=1)

        self._enable_cb.stateChanged.connect(self._on_toggle)
        self._enable_cb.stateChanged.connect(self.valueChanged.emit)
        self._inner.valueChanged.connect(self.valueChanged.emit)

        # Default: check whether the default is None
        default = getattr(field_info, "default", PydanticUndefined)
        has_value = default is not PydanticUndefined and default is not None
        self._enable_cb.setChecked(has_value)
        self._inner.setEnabled(has_value)

    def _on_toggle(self, state: int) -> None:
        """Enable or disable the inner widget based on checkbox state."""
        enabled = bool(state)
        self._inner.setEnabled(enabled)

    def get_value(self) -> Any:
        """Return ``None`` when unchecked, otherwise the inner value."""
        if not self._enable_cb.isChecked():
            return None
        return self._inner.get_value()

    def set_value(self, value: Any) -> None:
        """Set the widget value.

        Parameters
        ----------
        value : Any
            If ``None``, unchecks the enable checkbox.  Otherwise
            checks it and delegates to the inner widget.
        """
        if value is None:
            self._enable_cb.setChecked(False)
            self._inner.setEnabled(False)
        else:
            self._enable_cb.setChecked(True)
            self._inner.setEnabled(True)
            self._inner.set_value(value)


# ---------------------------------------------------------------------------
# Concrete widget: Tuple[int, int]
# ---------------------------------------------------------------------------


class TupleFieldWidget(BaseFieldWidget):
    """Widget for ``Tuple[int, int]`` fields using two ``QSpinBox`` widgets.

    Parameters
    ----------
    field_name : str
        Pydantic field name.
    field_info : FieldInfo
        Pydantic field descriptor.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        field_name: str,
        field_info: FieldInfo,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(field_name, field_info, parent)

        self._spin_a = QSpinBox()
        self._spin_b = QSpinBox()
        self._spin_a.setRange(-999999, 999999)
        self._spin_b.setRange(-999999, 999999)

        lbl_a = QLabel("X:")
        lbl_b = QLabel("Y:")
        self._row_layout.addWidget(lbl_a)
        self._row_layout.addWidget(self._spin_a, stretch=1)
        self._row_layout.addWidget(lbl_b)
        self._row_layout.addWidget(self._spin_b, stretch=1)

        self._spin_a.valueChanged.connect(self.valueChanged.emit)
        self._spin_b.valueChanged.connect(self.valueChanged.emit)

        # Apply default
        default = getattr(field_info, "default", PydanticUndefined)
        if default is not PydanticUndefined and default is not None:
            self.set_value(default)

    def get_value(self) -> Tuple[int, int]:
        """Return the current ``(x, y)`` tuple."""
        return (self._spin_a.value(), self._spin_b.value())

    def set_value(self, value: Any) -> None:
        """Set both spin boxes from a 2-element sequence.

        Parameters
        ----------
        value : Any
            A tuple, list, or other 2-element iterable of ints.
        """
        if value is not None and len(value) >= 2:
            self._spin_a.setValue(int(value[0]))
            self._spin_b.setValue(int(value[1]))


# ---------------------------------------------------------------------------
# Concrete widget: List[BaseModel]
# ---------------------------------------------------------------------------


class ListFieldWidget(BaseFieldWidget):
    """Widget for ``List[SubModel]`` fields (e.g. ``List[CrosshairConfig]``).

    Renders each list item as a collapsible group of field widgets with
    *Add* / *Remove* buttons.  New items are created with the sub-model's
    defaults.

    Parameters
    ----------
    field_name : str
        Pydantic field name.
    field_info : FieldInfo
        Pydantic field descriptor.
    item_model : type
        The Pydantic ``BaseModel`` subclass for each list element.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        field_name: str,
        field_info: FieldInfo,
        item_model: Type[BaseModel],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(field_name, field_info, parent)
        self._item_model = item_model
        self._item_widgets: List[QGroupBox] = []

        # Container for list items
        self._items_layout = QVBoxLayout()
        self._row_layout.addLayout(self._items_layout, stretch=1)

        # Re-arrange: put items below the label row
        self._outer_layout.removeItem(self._row_layout)
        header_layout = QHBoxLayout()
        header_layout.addWidget(self._label)

        add_btn = QPushButton("+ Add")
        add_btn.setMaximumWidth(80)
        add_btn.clicked.connect(self._add_item)
        header_layout.addStretch()
        header_layout.addWidget(add_btn)

        self._outer_layout.insertLayout(0, header_layout)
        self._outer_layout.insertLayout(1, self._items_layout)

        # Apply default
        default = getattr(field_info, "default_factory", None)
        if default is not None:
            try:
                default_val = default()
                if default_val:
                    self.set_value(default_val)
            except Exception:
                pass

    def _add_item(self, data: Optional[dict] = None) -> None:
        """Add a new list item with default or provided values.

        Parameters
        ----------
        data : dict, optional
            Field values to populate the new item with.
        """
        idx = len(self._item_widgets)
        group = QGroupBox(f"Item {idx + 1}")
        group_layout = QVBoxLayout(group)

        # Build field widgets for the sub-model
        item_field_widgets: List[BaseFieldWidget] = []
        for sub_name, sub_field in self._item_model.model_fields.items():
            sub_type = self._item_model.__annotations__.get(sub_name, str)
            widget = create_field_widget(sub_name, sub_field, sub_type, group)
            if data and sub_name in data:
                widget.set_value(data[sub_name])
            widget.valueChanged.connect(self.valueChanged.emit)
            item_field_widgets.append(widget)
            group_layout.addWidget(widget)

        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.setMaximumWidth(80)
        remove_btn.clicked.connect(lambda checked, g=group: self._remove_item(g))
        group_layout.addWidget(remove_btn)

        group.setProperty("field_widgets", item_field_widgets)
        self._item_widgets.append(group)
        self._items_layout.addWidget(group)
        self.valueChanged.emit()

    def _remove_item(self, group: QGroupBox) -> None:
        """Remove a list item group box.

        Parameters
        ----------
        group : QGroupBox
            The group box to remove.
        """
        if group in self._item_widgets:
            self._item_widgets.remove(group)
            self._items_layout.removeWidget(group)
            group.deleteLater()
            # Re-number remaining items
            for i, g in enumerate(self._item_widgets):
                g.setTitle(f"Item {i + 1}")
            self.valueChanged.emit()

    def get_value(self) -> List[Any]:
        """Return a list of dictionaries, one per item."""
        result = []
        for group in self._item_widgets:
            item_data = {}
            widgets = group.property("field_widgets")
            if widgets:
                for w in widgets:
                    item_data[w.field_name] = w.get_value()
            result.append(item_data)
        return result

    def set_value(self, value: Any) -> None:
        """Replace all items with the given list of dicts/models.

        Parameters
        ----------
        value : Any
            A list of dicts or Pydantic model instances.
        """
        # Clear existing items
        for group in list(self._item_widgets):
            self._remove_item(group)

        if value is None:
            return

        for item in value:
            if isinstance(item, BaseModel):
                data = item.model_dump()
            elif isinstance(item, dict):
                data = item
            else:
                data = {}
            self._add_item(data)


# ---------------------------------------------------------------------------
# Concrete widget: Dict[str, Any]
# ---------------------------------------------------------------------------


class DictFieldWidget(BaseFieldWidget):
    """Widget for ``Dict[str, Any]`` fields using a ``QTextEdit`` for YAML.

    The user edits raw YAML text.  On every change the text is validated
    as parseable YAML and an error is shown if it is not.

    Parameters
    ----------
    field_name : str
        Pydantic field name.
    field_info : FieldInfo
        Pydantic field descriptor.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        field_name: str,
        field_info: FieldInfo,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(field_name, field_info, parent)
        self._text_edit = QTextEdit()
        self._text_edit.setMaximumHeight(100)  # ~3-4 lines
        self._text_edit.setAcceptRichText(False)
        self._row_layout.addWidget(self._text_edit, stretch=1)
        self._text_edit.textChanged.connect(self._on_text_changed)

        # Apply default
        default = getattr(field_info, "default", PydanticUndefined)
        if default is not PydanticUndefined and default is not None:
            self.set_value(default)

    def _on_text_changed(self) -> None:
        """Validate YAML on every keystroke and emit valueChanged."""
        text = self._text_edit.toPlainText().strip()
        if text:
            try:
                yaml.safe_load(text)
                self.set_error(None)
            except yaml.YAMLError as exc:
                self.set_error(f"Invalid YAML: {exc}")
        else:
            self.set_error(None)
        self.valueChanged.emit()

    def get_value(self) -> Optional[Dict[str, Any]]:
        """Parse and return the YAML text as a dictionary."""
        text = self._text_edit.toPlainText().strip()
        if not text:
            return None
        try:
            result = yaml.safe_load(text)
            return result if isinstance(result, dict) else None
        except yaml.YAMLError:
            return None

    def set_value(self, value: Any) -> None:
        """Set the text edit from a dictionary.

        Parameters
        ----------
        value : Any
            A dictionary to serialise as YAML, or ``None`` to clear.
        """
        if value is None:
            self._text_edit.clear()
        elif isinstance(value, dict):
            self._text_edit.setPlainText(yaml.dump(value, default_flow_style=False))
        else:
            self._text_edit.setPlainText(str(value))


# ---------------------------------------------------------------------------
# Concrete widget: Path / Optional[Union[str, Path]]
# ---------------------------------------------------------------------------


class PathFieldWidget(BaseFieldWidget):
    """Widget for ``Path`` fields with a line edit and *Browse* button.

    Parameters
    ----------
    field_name : str
        Pydantic field name.
    field_info : FieldInfo
        Pydantic field descriptor.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        field_name: str,
        field_info: FieldInfo,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(field_name, field_info, parent)
        self._line_edit = QLineEdit()
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.setMaximumWidth(80)

        self._row_layout.addWidget(self._line_edit, stretch=1)
        self._row_layout.addWidget(self._browse_btn)

        self._line_edit.textChanged.connect(self.valueChanged.emit)
        self._browse_btn.clicked.connect(self._on_browse)

        # Apply default
        default = getattr(field_info, "default", PydanticUndefined)
        if default is not PydanticUndefined and default is not None:
            self._line_edit.setText(str(default))

    def _on_browse(self) -> None:
        """Open a file dialog and set the result."""
        path, _ = QFileDialog.getOpenFileName(self, f"Select {self._field_name}")
        if path:
            self._line_edit.setText(path)

    def get_value(self) -> Optional[Path]:
        """Return the path, or ``None`` if empty."""
        text = self._line_edit.text().strip()
        return Path(text) if text else None

    def set_value(self, value: Any) -> None:
        """Set the line-edit text from a path or string.

        Parameters
        ----------
        value : Any
            A ``Path``, string, or ``None``.
        """
        if value is None:
            self._line_edit.clear()
        else:
            self._line_edit.setText(str(value))


# ---------------------------------------------------------------------------
# Type-inspection helpers (private)
# ---------------------------------------------------------------------------


def _is_optional(field_type: type) -> bool:
    """Return ``True`` if *field_type* is ``Optional[X]`` (i.e. ``Union[X, None]``)."""
    origin = get_origin(field_type)
    if origin is Union:
        args = get_args(field_type)
        return type(None) in args and len(args) == 2
    return False


def _unwrap_optional(field_type: type) -> type:
    """Return the inner type ``X`` from ``Optional[X]``."""
    args = get_args(field_type)
    return next(a for a in args if a is not type(None))


def _contains_path(field_type: type) -> bool:
    """Return ``True`` if *field_type* involves ``Path`` anywhere."""
    if field_type is Path:
        return True
    for arg in get_args(field_type):
        if arg is Path or _contains_path(arg):
            return True
    return False


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_field_widget(
    field_name: str,
    field_info: FieldInfo,
    field_type: type,
    parent: Optional[QWidget] = None,
) -> BaseFieldWidget:
    """Create the appropriate field widget for a Pydantic field.

    Inspects *field_type* and the metadata in *field_info* to choose
    the right concrete :class:`BaseFieldWidget` subclass.

    Parameters
    ----------
    field_name : str
        The Pydantic field name.
    field_info : FieldInfo
        The Pydantic ``FieldInfo`` descriptor for the field.
    field_type : type
        The resolved Python type annotation for the field.
    parent : QWidget, optional
        Parent Qt widget.

    Returns
    -------
    BaseFieldWidget
        A widget instance ready to be added to a layout.

    Notes
    -----
    The resolution order is:

    1. ``Optional[X]`` → wrap the inner widget in :class:`OptionalFieldWidget`
    2. ``Enum`` subclass → :class:`EnumFieldWidget`
    3. ``bool`` → :class:`BoolFieldWidget`  (checked before ``int``!)
    4. ``int`` → :class:`IntFieldWidget`
    5. ``float`` → :class:`FloatFieldWidget`
    6. ``str`` → :class:`StringFieldWidget`
    7. ``Tuple[int, int]`` → :class:`TupleFieldWidget`
    8. ``List[BaseModel]`` → :class:`ListFieldWidget`
    9. ``Dict`` → :class:`DictFieldWidget`
    10. ``Path`` or union containing ``Path`` → :class:`PathFieldWidget`
    11. Fallback → :class:`StringFieldWidget` with ``str()`` conversion
    """
    # --- 1. Optional[X] ---------------------------------------------------
    if _is_optional(field_type):
        inner_type = _unwrap_optional(field_type)

        # Special-case: Optional[Union[str, Path]] → PathFieldWidget
        if _contains_path(inner_type) or inner_type is Path:
            inner_widget = PathFieldWidget(field_name, field_info, parent)
        # Special-case: Optional[BaseModel subclass] → handled by section_widget
        elif isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
            # For Optional sub-models, fall back to a string widget as a
            # placeholder; section_widget.py will handle these specially.
            inner_widget = StringFieldWidget(field_name, field_info, parent)
        else:
            inner_widget = _create_inner_widget(
                field_name, field_info, inner_type, parent
            )
        return OptionalFieldWidget(field_name, field_info, inner_widget, parent)

    # --- Non-optional types ------------------------------------------------
    return _create_inner_widget(field_name, field_info, field_type, parent)


def _create_inner_widget(
    field_name: str,
    field_info: FieldInfo,
    field_type: type,
    parent: Optional[QWidget] = None,
) -> BaseFieldWidget:
    """Create a widget for a non-optional type.

    Parameters
    ----------
    field_name : str
        The Pydantic field name.
    field_info : FieldInfo
        The Pydantic ``FieldInfo`` descriptor.
    field_type : type
        The resolved (non-optional) Python type.
    parent : QWidget, optional
        Parent Qt widget.

    Returns
    -------
    BaseFieldWidget
        A concrete widget instance.
    """
    origin = get_origin(field_type)
    args = get_args(field_type)

    # --- Enum --------------------------------------------------------------
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return EnumFieldWidget(field_name, field_info, field_type, parent)

    # --- bool (must come before int, since bool is a subclass of int) ------
    if field_type is bool:
        return BoolFieldWidget(field_name, field_info, parent)

    # --- int ---------------------------------------------------------------
    if field_type is int:
        return IntFieldWidget(field_name, field_info, parent)

    # --- float -------------------------------------------------------------
    if field_type is float:
        return FloatFieldWidget(field_name, field_info, parent)

    # --- str ---------------------------------------------------------------
    if field_type is str:
        return StringFieldWidget(field_name, field_info, parent)

    # --- Tuple[int, int] ---------------------------------------------------
    if origin is tuple and args and len(args) == 2:
        return TupleFieldWidget(field_name, field_info, parent)

    # --- List[BaseModel] ---------------------------------------------------
    if origin is list and args:
        item_type = args[0]
        if isinstance(item_type, type) and issubclass(item_type, BaseModel):
            return ListFieldWidget(field_name, field_info, item_type, parent)
        # List of non-model types (e.g. List[float]) → YAML text editor
        return DictFieldWidget(field_name, field_info, parent)

    # --- Dict --------------------------------------------------------------
    if origin is dict:
        return DictFieldWidget(field_name, field_info, parent)

    # --- Path --------------------------------------------------------------
    if field_type is Path or _contains_path(field_type):
        return PathFieldWidget(field_name, field_info, parent)

    # --- BaseModel subclass (sub-config) -----------------------------------
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        # Sub-models are typically handled by section_widget.py as nested
        # sections.  Return a placeholder string widget here.
        return StringFieldWidget(field_name, field_info, parent)

    # --- Fallback: treat as string -----------------------------------------
    logger.debug(
        "No specific widget for field %r of type %r; falling back to StringFieldWidget",
        field_name,
        field_type,
    )
    return StringFieldWidget(field_name, field_info, parent)
