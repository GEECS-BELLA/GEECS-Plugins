"""Collapsible section widget for Pydantic sub-model form rendering.

Renders all fields of a Pydantic ``BaseModel`` subclass as a collapsible
form section backed by a ``QWidget`` + custom header bar.  The header
contains a ``QToolButton`` (▶/▼) that toggles the content area, and
either a ``QCheckBox`` (when the model has an ``enabled: bool`` field) or
a plain ``QLabel`` for the section title.

When the ``enabled`` checkbox is unchecked, :meth:`SectionWidget.get_values`
returns ``None``, signalling to the parent that the section is disabled.

Nested ``BaseModel`` fields are rendered recursively as child
``SectionWidget`` instances rather than flat field widgets.

This module is consumed by ``config_editor_window.py`` to build the
full configuration form.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, get_args, get_origin

from pydantic import BaseModel, ValidationError
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .field_widgets import BaseFieldWidget, create_field_widget

logger = logging.getLogger(__name__)


def _is_basemodel_type(field_type: type) -> bool:
    """Check whether *field_type* is a concrete ``BaseModel`` subclass.

    Handles ``Optional[SomeModel]`` by unwrapping the union first, and
    also checks the raw type for plain ``BaseModel`` subclasses.

    Parameters
    ----------
    field_type : type
        The Python type annotation to inspect.

    Returns
    -------
    bool
        ``True`` if the (possibly unwrapped) type is a ``BaseModel``
        subclass.
    """
    # Direct subclass check
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        return True

    # Optional[SomeModel] → Union[SomeModel, None]
    from typing import Union

    origin = get_origin(field_type)
    if origin is Union:
        args = get_args(field_type)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = non_none[0]
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                return True

    return False


def _unwrap_model_type(field_type: type) -> Type[BaseModel]:
    """Return the ``BaseModel`` subclass from *field_type*.

    Works for both plain ``SomeModel`` and ``Optional[SomeModel]``.

    Parameters
    ----------
    field_type : type
        The type annotation (possibly ``Optional``).

    Returns
    -------
    Type[BaseModel]
        The concrete model class.

    Raises
    ------
    TypeError
        If the type cannot be unwrapped to a ``BaseModel``.
    """
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        return field_type

    from typing import Union

    origin = get_origin(field_type)
    if origin is Union:
        args = get_args(field_type)
        non_none = [a for a in args if a is not type(None)]
        if (
            len(non_none) == 1
            and isinstance(non_none[0], type)
            and issubclass(non_none[0], BaseModel)
        ):
            return non_none[0]

    raise TypeError(f"Cannot unwrap BaseModel from {field_type}")


class SectionWidget(QWidget):
    """Collapsible form section for a Pydantic sub-model.

    Renders each field of *model_class* as an appropriate widget inside a
    ``QFormLayout``.  Fields that are themselves ``BaseModel`` subclasses are
    rendered as nested ``SectionWidget`` instances.

    The widget uses a custom header bar containing:

    * A ``QToolButton`` (▶ / ▼) that collapses or expands the content area.
    * A ``QCheckBox`` (if the model declares an ``enabled: bool`` field) or a
      plain ``QLabel`` showing the section title.  The checkbox controls the
      *enabled* state: when unchecked, :meth:`get_values` returns ``None``.

    The content area starts **collapsed** by default so that forms with many
    sections do not overwhelm the user on first open.

    Parameters
    ----------
    section_name : str
        The config key name (e.g. ``"background"``, ``"roi"``).
    model_class : Type[BaseModel]
        The Pydantic model class whose fields define the form.
    parent : QWidget, optional
        Parent Qt widget.

    Signals
    -------
    sectionEnabledChanged(str, bool)
        Emitted when the enabled checkbox is toggled.  Arguments are
        ``(section_name, is_enabled)``.
    valueChanged()
        Emitted when any field value changes within this section.
    """

    sectionEnabledChanged = pyqtSignal(str, bool)
    valueChanged = pyqtSignal()

    def __init__(
        self,
        section_name: str,
        model_class: Type[BaseModel],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._section_name = section_name
        self._model_class = model_class

        # Detect whether the model has an ``enabled`` bool field.
        self._has_enabled_field: bool = "enabled" in model_class.model_fields

        # Internal storage
        self._field_widgets: Dict[str, BaseFieldWidget] = {}
        self._nested_sections: Dict[str, SectionWidget] = {}

        # Whether the section is currently enabled (mirrors the checkbox).
        self._enabled: bool = True

        # Build the UI
        self._setup_ui()
        self._build_fields()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the outer frame, header bar, and collapsible content area."""
        # Outer frame gives the section a visible border
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        frame = QFrame(self)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        outer_layout.addWidget(frame)

        # ── Header bar ──────────────────────────────────────────────────
        self._header = QFrame(frame)
        self._header.setObjectName("sectionHeader")
        self._header.setStyleSheet(
            "QFrame#sectionHeader { background-color: #d8d8d8; }"
        )
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(4, 2, 4, 2)
        header_layout.setSpacing(4)

        # Collapse / expand toggle button
        self._toggle_btn = QToolButton(self._header)
        self._toggle_btn.setText("▶")
        self._toggle_btn.setToolTip("Expand section")
        self._toggle_btn.setStyleSheet("QToolButton { border: none; font-size: 10px; }")
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)
        header_layout.addWidget(self._toggle_btn)

        title_text = _format_section_title(self._section_name)

        if self._has_enabled_field:
            # Checkbox doubles as the "enabled" control + section label
            self._enabled_cb = QCheckBox(title_text, self._header)
            self._enabled_cb.setChecked(True)
            self._enabled_cb.toggled.connect(self._on_enabled_toggled)
            header_layout.addWidget(self._enabled_cb)
        else:
            self._enabled_cb = None  # type: ignore[assignment]
            title_label = QLabel(title_text, self._header)
            title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            header_layout.addWidget(title_label)

        header_layout.addStretch()
        frame_layout.addWidget(self._header)

        # ── Content area ────────────────────────────────────────────────
        self._content_widget = QWidget(frame)
        self._form_layout = QFormLayout(self._content_widget)
        self._form_layout.setContentsMargins(8, 4, 8, 4)
        frame_layout.addWidget(self._content_widget)

        # Start collapsed
        self._content_widget.setVisible(False)

    def _build_fields(self) -> None:
        """Iterate over model fields and create widgets or nested sections.

        When the model has an ``enabled: bool`` field, that field is
        **skipped** because the header checkbox already represents it.
        """
        for field_name, field_info in self._model_class.model_fields.items():
            # Skip the ``enabled`` bool field — the header checkbox
            # already represents it.
            if field_name == "enabled" and self._has_enabled_field:
                continue

            field_type = self._model_class.__annotations__.get(field_name, str)

            if _is_basemodel_type(field_type):
                # Nested sub-model → recursive SectionWidget
                inner_model = _unwrap_model_type(field_type)
                nested = SectionWidget(field_name, inner_model, self)
                nested.valueChanged.connect(self.valueChanged.emit)
                nested.sectionEnabledChanged.connect(self.sectionEnabledChanged.emit)
                self._nested_sections[field_name] = nested
                self._form_layout.addRow(nested)
            else:
                # Scalar / list / dict field → field widget
                widget = create_field_widget(field_name, field_info, field_type, self)
                widget.valueChanged.connect(self.valueChanged.emit)
                self._field_widgets[field_name] = widget
                label = _format_section_title(field_name)
                self._form_layout.addRow(label, widget)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_toggle_clicked(self) -> None:
        """Toggle the visibility of the content area."""
        visible = not self._content_widget.isVisible()
        self._content_widget.setVisible(visible)
        if visible:
            self._toggle_btn.setText("▼")
            self._toggle_btn.setToolTip("Collapse section")
        else:
            self._toggle_btn.setText("▶")
            self._toggle_btn.setToolTip("Expand section")

    def _on_enabled_toggled(self, checked: bool) -> None:
        """Handle the enabled checkbox toggle.

        Parameters
        ----------
        checked : bool
            Whether the section is now enabled.
        """
        self._enabled = checked
        self.sectionEnabledChanged.emit(self._section_name, checked)
        self.valueChanged.emit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_values(self) -> Optional[Dict[str, Any]]:
        """Return the current field values as a dict, or ``None`` if disabled.

        The returned dictionary is suitable for passing to
        ``model_class.model_validate()``.

        When the model has an ``enabled`` field and the header checkbox is
        **unchecked**, ``None`` is returned.  For models without an
        ``enabled`` field the section is always considered active and a
        dict is always returned.

        When the checkbox is **checked** and the model declares an
        ``enabled`` field, ``"enabled": True`` is injected into the
        returned dict.

        Returns
        -------
            Field values keyed by Pydantic field name, or ``None`` when
            the section is disabled.
        """
        if self._has_enabled_field and not self._enabled:
            return None

        data: Dict[str, Any] = {}

        # Inject ``"enabled": True`` when the model declares the field.
        if self._has_enabled_field:
            data["enabled"] = True

        for name, widget in self._field_widgets.items():
            data[name] = widget.get_value()
        for name, section in self._nested_sections.items():
            data[name] = section.get_values()
        return data

    def set_values(self, data: Optional[Dict[str, Any]]) -> None:
        """Populate the section from a dictionary of field values.

        Parameters
        ----------
        data : dict or None
            If ``None``, the section is unchecked (disabled).  Otherwise
            each field widget is populated from the corresponding key.
            For models with an ``enabled`` field, the checkbox checked
            state is driven by ``data["enabled"]`` (defaulting to
            ``True`` when the key is absent).
        """
        if data is None:
            if self._has_enabled_field and self._enabled_cb is not None:
                self._enabled_cb.setChecked(False)
            self._enabled = False
            return

        if self._has_enabled_field and self._enabled_cb is not None:
            enabled_val = bool(data.get("enabled", True))
            self._enabled_cb.setChecked(enabled_val)
            self._enabled = enabled_val
        else:
            self._enabled = True

        for name, widget in self._field_widgets.items():
            if name in data:
                widget.set_value(data[name])
        for name, section in self._nested_sections.items():
            section_data = data.get(name)
            section.set_values(section_data)

    def is_enabled(self) -> bool:
        """Return whether the section is currently enabled.

        Returns
        -------
            ``True`` if the section is enabled (checkbox checked, or the
            model has no ``enabled`` field).
        """
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Check or uncheck the enabled state of the section.

        Parameters
        ----------
        enabled : bool
            Whether to enable the section.
        """
        if self._has_enabled_field and self._enabled_cb is not None:
            self._enabled_cb.setChecked(enabled)
        else:
            self._enabled = enabled

    def validate(self) -> List[Tuple[str, str]]:
        """Validate current values through Pydantic.

        Calls ``model_class.model_validate()`` on the current widget
        values and returns any validation errors as a list of
        ``(field_name, error_message)`` tuples.

        Returns
        -------
            Validation errors.  Empty list means the data is valid.
        """
        data = self.get_values()
        if data is None:
            # Disabled section is always valid
            return []

        errors: List[Tuple[str, str]] = []
        try:
            self._model_class.model_validate(data)
        except ValidationError as exc:
            for err in exc.errors():
                # Build a dotted field path from the error location
                loc_parts = [str(p) for p in err.get("loc", [])]
                field_path = ".".join(loc_parts) if loc_parts else "unknown"
                message = err.get("msg", "Validation error")
                errors.append((field_path, message))
        return errors

    def show_errors(self, errors: List[Tuple[str, str]]) -> None:
        """Display validation errors on the appropriate field widgets.

        Parameters
        ----------
        errors : list of (str, str)
            Each tuple is ``(field_name, error_message)``.  The
            *field_name* may be a dotted path for nested sections
            (e.g. ``"dynamic_computation.percentile"``).
        """
        for field_path, message in errors:
            parts = field_path.split(".", 1)
            top_field = parts[0]

            if len(parts) > 1 and top_field in self._nested_sections:
                # Delegate to nested section
                self._nested_sections[top_field].show_errors([(parts[1], message)])
            elif top_field in self._field_widgets:
                self._field_widgets[top_field].set_error(message)

    def clear_errors(self) -> None:
        """Clear all error indicators on field widgets and nested sections."""
        for widget in self._field_widgets.values():
            widget.set_error(None)
        for section in self._nested_sections.values():
            section.clear_errors()


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _format_section_title(name: str) -> str:
    """Convert a snake_case name to a Title Case label.

    Args:
        name: A snake_case identifier (e.g. ``"crosshair_masking"``).

    Returns
    -------
        Title-cased string (e.g. ``"Crosshair Masking"``).
    """
    return name.replace("_", " ").title()
