"""PyQt5 widget builders for Pydantic model fields.

``ModelFormWidget`` is the main entry point: give it a model class (and
optionally a pre-loaded data dict) and it builds a scrollable form with the
right control for every field type.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Type

import yaml
from pydantic import BaseModel, ValidationError
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from .introspect import FieldDescriptor, FieldKind, describe_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNSET = object()  # sentinel for "no value provided"


def _safe_label(name: str) -> str:
    return name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Primitive field widgets
# ---------------------------------------------------------------------------


class _BoolWidget(QCheckBox):
    def get_value(self) -> bool:
        return self.isChecked()

    def set_value(self, v: Any) -> None:
        self.setChecked(bool(v))


class _IntWidget(QSpinBox):
    def __init__(self, constraints: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setRange(
            int(constraints.get("ge", constraints.get("gt", -10_000_000))),
            int(constraints.get("le", constraints.get("lt", 10_000_000))),
        )
        self.setStepType(QSpinBox.AdaptiveDecimalStepType)

    def get_value(self) -> int:
        return self.value()

    def set_value(self, v: Any) -> None:
        try:
            self.setValue(int(v))
        except (TypeError, ValueError):
            pass


class _FloatWidget(QDoubleSpinBox):
    def __init__(self, constraints: Dict[str, Any], parent=None):
        super().__init__(parent)
        lo = constraints.get("ge", constraints.get("gt", -1e18))
        hi = constraints.get("le", constraints.get("lt", 1e18))
        self.setRange(float(lo), float(hi))
        self.setDecimals(6)
        self.setSingleStep(0.001)
        self.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)

    def get_value(self) -> float:
        return self.value()

    def set_value(self, v: Any) -> None:
        try:
            self.setValue(float(v))
        except (TypeError, ValueError):
            pass


class _StrWidget(QWidget):
    """QLineEdit with an optional browse button for path-like fields."""

    def __init__(self, is_path: bool = False, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._edit = QLineEdit()
        layout.addWidget(self._edit)
        if is_path:
            btn = QToolButton()
            btn.setText("…")
            btn.clicked.connect(self._browse)
            layout.addWidget(btn)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file")
        if path:
            self._edit.setText(path)

    def get_value(self) -> Optional[str]:
        t = self._edit.text().strip()
        return t if t else None

    def set_value(self, v: Any) -> None:
        self._edit.setText(str(v) if v is not None else "")


class _EnumWidget(QComboBox):
    def __init__(self, enum_class: Type[Enum], parent=None):
        super().__init__(parent)
        self._enum_class = enum_class
        for member in enum_class:
            self.addItem(str(member.value), member)

    def get_value(self) -> Any:
        return self.currentData().value  # return the raw str value

    def set_value(self, v: Any) -> None:
        for i in range(self.count()):
            if self.itemData(i).value == v:
                self.setCurrentIndex(i)
                return


class _TupleWidget(QWidget):
    def __init__(self, elem_type: type, parent=None):
        super().__init__(parent)
        self._elem_type = elem_type
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if elem_type is float:
            self._w0 = QDoubleSpinBox()
            self._w0.setRange(-1e9, 1e9)
            self._w1 = QDoubleSpinBox()
            self._w1.setRange(-1e9, 1e9)
        else:
            self._w0 = QSpinBox()
            self._w0.setRange(-1_000_000, 1_000_000)
            self._w1 = QSpinBox()
            self._w1.setRange(-1_000_000, 1_000_000)
        layout.addWidget(self._w0)
        layout.addWidget(QLabel(","))
        layout.addWidget(self._w1)

    def get_value(self) -> tuple:
        return (self._w0.value(), self._w1.value())

    def set_value(self, v: Any) -> None:
        if v and len(v) >= 2:
            self._w0.setValue(v[0])
            self._w1.setValue(v[1])


class _DictWidget(QPlainTextEdit):
    """YAML text editor for Dict[str, Any] or unknown fields."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(200)
        self.setPlaceholderText("YAML here, e.g.:\nkey: value\nother: 123")
        font = self.font()
        font.setFamily("Menlo")
        font.setStyleHint(font.Monospace)
        self.setFont(font)

    def get_value(self) -> Any:
        text = self.toPlainText().strip()
        if not text:
            return {}
        try:
            return yaml.safe_load(text)
        except Exception:
            return text  # return raw string; validation will catch it

    def set_value(self, v: Any) -> None:
        if v is None:
            self.setPlainText("")
        elif isinstance(v, (dict, list)):
            self.setPlainText(yaml.dump(v, default_flow_style=False).rstrip())
        else:
            self.setPlainText(str(v))


class _OptionalPrimitiveWidget(QWidget):
    """Checkbox + inner widget for Optional[int/float/str] primitives."""

    def __init__(self, inner: QWidget, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._cb = QCheckBox()
        self._cb.setToolTip("Check to set a value; uncheck to leave as None")
        self._inner = inner
        self._inner.setEnabled(False)
        self._cb.stateChanged.connect(lambda s: self._inner.setEnabled(s == Qt.Checked))
        layout.addWidget(self._cb)
        layout.addWidget(self._inner)

    def get_value(self) -> Any:
        if not self._cb.isChecked():
            return None
        return self._inner.get_value()

    def set_value(self, v: Any) -> None:
        if v is None:
            self._cb.setChecked(False)
            self._inner.setEnabled(False)
        else:
            self._cb.setChecked(True)
            self._inner.setEnabled(True)
            self._inner.set_value(v)


class _ListScalarWidget(QPlainTextEdit):
    """One item per line editor for List[scalar]."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setMaximumHeight(120)
        self.setPlaceholderText("One item per line")

    def get_value(self) -> list:
        """Return non-empty stripped lines as a list."""
        return [ln.strip() for ln in self.toPlainText().splitlines() if ln.strip()]

    def set_value(self, v: Any) -> None:
        if isinstance(v, (list, tuple)):
            self.setPlainText("\n".join(str(x) for x in v))
        elif v:
            self.setPlainText(str(v))


# ---------------------------------------------------------------------------
# Collapsible group box for nested models
# ---------------------------------------------------------------------------


class CollapsibleGroup(QWidget):
    """Group box with a toggle arrow and optional enable/disable checkbox.

    Parameters
    ----------
    title : str
        Group header label.
    model_class : Type[BaseModel]
        The nested model to render inside.
    optional : bool
        If True, shows a checkbox to enable/disable the whole group.
        When disabled, ``get_value()`` returns None.
    """

    def __init__(
        self,
        title: str,
        model_class: Type[BaseModel],
        optional: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._model_class = model_class
        self._optional = optional

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 0)
        outer.setSpacing(0)

        # Header row
        header = QHBoxLayout()
        self._toggle_btn = QToolButton()
        self._toggle_btn.setArrowType(Qt.RightArrow)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.clicked.connect(self._on_toggle)
        header.addWidget(self._toggle_btn)

        if optional:
            self._enable_cb = QCheckBox(title)
            self._enable_cb.stateChanged.connect(self._on_enable_changed)
            header.addWidget(self._enable_cb)
        else:
            lbl = QLabel(f"<b>{title}</b>")
            header.addWidget(lbl)

        header.addStretch()
        outer.addLayout(header)

        # Content area
        self._content = QWidget()
        self._content.setVisible(False)
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(16, 0, 0, 0)

        self._form = ModelFormWidget(model_class)
        content_layout.addWidget(self._form)
        outer.addWidget(self._content)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        outer.addWidget(line)

        if optional:
            self._on_enable_changed(Qt.Unchecked)

    def _on_toggle(self, checked: bool) -> None:
        self._toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)

    def _on_enable_changed(self, state: int) -> None:
        enabled = state == Qt.Checked
        self._form.setEnabled(enabled)
        # Auto-expand when enabled
        if enabled and not self._toggle_btn.isChecked():
            self._toggle_btn.setChecked(True)
            self._toggle_btn.setArrowType(Qt.DownArrow)
            self._content.setVisible(True)

    def is_enabled(self) -> bool:
        """Return True if the group is active (checkbox checked or non-optional)."""
        if self._optional:
            return self._enable_cb.isChecked()
        return True

    def get_value(self) -> Optional[Dict[str, Any]]:
        """Return sub-model dict, or None when the group is disabled."""
        if self._optional and not self._enable_cb.isChecked():
            return None
        return self._form.get_value()

    def set_value(self, v: Optional[Dict[str, Any]]) -> None:
        """Populate the nested form from *v*; pass None to disable an optional group."""
        if v is None:
            if self._optional:
                self._enable_cb.setChecked(False)
            return
        if self._optional:
            self._enable_cb.setChecked(True)
        if not self._toggle_btn.isChecked():
            self._toggle_btn.setChecked(True)
            self._toggle_btn.setArrowType(Qt.DownArrow)
            self._content.setVisible(True)
        self._form.set_value(v)


# ---------------------------------------------------------------------------
# List-of-models editor
# ---------------------------------------------------------------------------


class ListModelWidget(QWidget):
    """Add/remove editor for List[BaseModel] fields."""

    def __init__(self, item_model: Type[BaseModel], parent=None):
        super().__init__(parent)
        self._item_model = item_model
        self._items: List[ModelFormWidget] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add item")
        add_btn.clicked.connect(self._add_item)
        btn_row.addWidget(add_btn)
        btn_row.addStretch()
        self._layout.addLayout(btn_row)

        self._items_layout = QVBoxLayout()
        self._layout.addLayout(self._items_layout)

    def _add_item(self, data: Optional[dict] = None) -> None:
        idx = len(self._items)
        container = QGroupBox(f"Item {idx + 1}")
        container_layout = QVBoxLayout(container)

        form = ModelFormWidget(self._item_model)
        if data:
            form.set_value(data)
        container_layout.addWidget(form)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_item(container, form))
        container_layout.addWidget(remove_btn)

        self._items_layout.addWidget(container)
        self._items.append(form)

    def _remove_item(self, container: QGroupBox, form: ModelFormWidget) -> None:
        self._items.remove(form)
        container.setParent(None)

    def get_value(self) -> list:
        """Return all item form values as a list of dicts."""
        return [f.get_value() for f in self._items]

    def set_value(self, v: Any) -> None:
        """Rebuild the list from *v* (a list of dicts)."""
        if not isinstance(v, list):
            return
        while self._items_layout.count():
            item = self._items_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._items.clear()
        for data in v:
            self._add_item(data)


# ---------------------------------------------------------------------------
# Core form widget
# ---------------------------------------------------------------------------


class ModelFormWidget(QWidget):
    """Renders all fields of a Pydantic model class as a form.

    Parameters
    ----------
    model_class : Type[BaseModel]
        The model whose fields to render.
    data : dict, optional
        Initial values to populate into the form.
    """

    changed = pyqtSignal()  # emitted whenever any field changes

    def __init__(
        self,
        model_class: Type[BaseModel],
        data: Optional[Dict[str, Any]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._model_class = model_class
        self._field_widgets: Dict[str, Any] = {}
        self._literal_values: Dict[str, Any] = {}  # hidden fixed values

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.addLayout(form_layout)

        for desc in describe_model(model_class):
            widget = self._build_widget(desc)
            if widget is None:
                continue  # LITERAL — stored in _literal_values, not shown

            label_text = _safe_label(desc.name)
            if desc.is_required:
                label_text += " *"

            lbl = QLabel(label_text)
            if desc.description:
                lbl.setToolTip(desc.description)
                widget.setToolTip(desc.description) if hasattr(
                    widget, "setToolTip"
                ) else None

            if isinstance(widget, (CollapsibleGroup, ListModelWidget)):
                # Use addRow with a spanning label above
                form_layout.addRow(widget)
            else:
                form_layout.addRow(lbl, widget)

            self._field_widgets[desc.name] = widget

        layout.addStretch()

        if data:
            self.set_value(data)

    def _build_widget(self, desc: FieldDescriptor) -> Optional[QWidget]:
        """Return a widget for *desc*, or None for hidden fields."""
        d = desc.default

        if desc.kind == FieldKind.LITERAL:
            vals = desc.literal_values
            if len(vals) == 1:
                # Single-value literal = discriminator / constant — hide it
                self._literal_values[desc.name] = vals[0]
                return None
            else:
                # Multi-value literal = choice — render as combo box
                w = QComboBox()
                for v in vals:
                    w.addItem(str(v))
                # pick default
                if d is not None and str(d) in [str(v) for v in vals]:
                    w.setCurrentText(str(d))
                # Attach get_value / set_value dynamically
                w.get_value = lambda combo=w: combo.currentText()
                w.set_value = (
                    lambda v, combo=w: combo.setCurrentText(str(v))
                    if v is not None
                    else None
                )
                return w

        if desc.kind == FieldKind.BOOL:
            w = _BoolWidget()
            w.set_value(d if d is not None else False)
            return w

        if desc.kind == FieldKind.INT:
            inner = _IntWidget(desc.constraints)
            if d is not None:
                inner.set_value(d)
            if desc.is_optional:
                w = _OptionalPrimitiveWidget(inner)
                w.set_value(d)
                return w
            return inner

        if desc.kind == FieldKind.FLOAT:
            inner = _FloatWidget(desc.constraints)
            if d is not None:
                inner.set_value(d)
            if desc.is_optional:
                w = _OptionalPrimitiveWidget(inner)
                w.set_value(d)
                return w
            return inner

        if desc.kind == FieldKind.STR:
            is_path = any(x in desc.name.lower() for x in ("path", "file", "dir"))
            inner = _StrWidget(is_path=is_path)
            if d is not None:
                inner.set_value(d)
            if desc.is_optional:
                # _StrWidget already returns None for empty — no wrapper needed
                return inner
            return inner

        if desc.kind == FieldKind.ENUM:
            w = _EnumWidget(desc.enum_class)
            if d is not None:
                w.set_value(d.value if isinstance(d, Enum) else d)
            return w

        if desc.kind == FieldKind.TUPLE:
            w = _TupleWidget(desc.scalar_type)
            if d is not None:
                w.set_value(d)
            return w

        if desc.kind == FieldKind.DICT:
            w = _DictWidget()
            if d is not None:
                w.set_value(d)
            return w

        if desc.kind == FieldKind.LIST_SCALAR:
            w = _ListScalarWidget()
            if d is not None:
                w.set_value(d)
            return w

        if desc.kind == FieldKind.NESTED:
            w = CollapsibleGroup(
                title=_safe_label(desc.name),
                model_class=desc.inner_model,
                optional=False,
            )
            if d is not None:
                w.set_value(d if isinstance(d, dict) else d.model_dump())
            return w

        if desc.kind == FieldKind.OPTIONAL_MODEL:
            w = CollapsibleGroup(
                title=_safe_label(desc.name),
                model_class=desc.inner_model,
                optional=True,
            )
            if d is not None:
                w.set_value(d if isinstance(d, dict) else d.model_dump())
            return w

        if desc.kind == FieldKind.LIST_MODEL:
            w = ListModelWidget(desc.inner_model)
            if d:
                w.set_value(d)
            return w

        # UNKNOWN fallback
        w = _DictWidget()
        w.setPlaceholderText("Enter value as YAML")
        if d is not None:
            w.set_value(d)
        return w

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_value(self) -> Dict[str, Any]:
        """Return current form state as a plain dict."""
        result = dict(self._literal_values)  # include hidden literals
        for name, widget in self._field_widgets.items():
            v = widget.get_value()
            if v is not None:
                result[name] = v
        return result

    def set_value(self, data: Dict[str, Any]) -> None:
        """Populate all widgets from *data*."""
        for name, widget in self._field_widgets.items():
            if name in data:
                widget.set_value(data[name])

    def validate(self) -> Optional[str]:
        """Attempt Pydantic validation. Returns error string or None."""
        try:
            self._model_class.model_validate(self.get_value())
            return None
        except ValidationError as e:
            return str(e)
