"""Collapsible section widget for Pydantic sub-model form rendering.

Renders all fields of a Pydantic ``BaseModel`` subclass as a checkable
``QGroupBox`` with a ``QFormLayout``.  When the group box is unchecked
the section is considered disabled (value is ``None``).

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
from PyQt5.QtWidgets import QFormLayout, QGroupBox, QWidget

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


class SectionWidget(QGroupBox):
    """Collapsible form section for a Pydantic sub-model.

    Renders each field of *model_class* as an appropriate widget inside
    a ``QFormLayout``.  Fields that are themselves ``BaseModel``
    subclasses are rendered as nested ``SectionWidget`` instances.

    The ``QGroupBox`` is **checkable**: when unchecked the section is
    disabled and :meth:`get_values` returns ``None``.

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
        Emitted when the section checkbox is toggled.  Arguments are
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
        super().__init__(_format_section_title(section_name), parent)
        self._section_name = section_name
        self._model_class = model_class

        # Detect whether the model has an ``enabled`` bool field.
        # When it does, the QGroupBox checkbox replaces the separate
        # ``BoolFieldWidget`` that would otherwise be rendered.
        self._has_enabled_field: bool = "enabled" in model_class.model_fields

        # Make the group box checkable (unchecked → section disabled)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggled)

        # Internal storage
        self._field_widgets: Dict[str, BaseFieldWidget] = {}
        self._nested_sections: Dict[str, SectionWidget] = {}

        # Layout
        self._form_layout = QFormLayout()
        self.setLayout(self._form_layout)

        # Build widgets for each field in the model
        self._build_fields()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_fields(self) -> None:
        """Iterate over model fields and create widgets or nested sections.

        When the model has an ``enabled: bool`` field, that field is
        **skipped** because the ``QGroupBox`` checkbox already serves
        the same purpose.
        """
        for field_name, field_info in self._model_class.model_fields.items():
            # Skip the ``enabled`` bool field — the QGroupBox checkbox
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

    def _on_toggled(self, checked: bool) -> None:
        """Handle the group-box checkbox toggle.

        Parameters
        ----------
        checked : bool
            Whether the section is now enabled.
        """
        self.sectionEnabledChanged.emit(self._section_name, checked)
        self.valueChanged.emit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_values(self) -> Optional[Dict[str, Any]]:
        """Return the current field values as a dict, or ``None`` if disabled.

        The returned dictionary is suitable for passing to
        ``model_class.model_validate()``.

        For models that have an ``enabled`` field, the dict always
        includes ``"enabled": True/False`` derived from the QGroupBox
        checkbox state, and the full field values are always returned
        (never ``None``).  For models *without* an ``enabled`` field,
        ``None`` is returned when the section is unchecked.

        Returns
        -------
        dict or None
            Field values keyed by Pydantic field name, or ``None`` when
            the section checkbox is unchecked and the model has no
            ``enabled`` field.
        """
        if not self.isChecked() and not self._has_enabled_field:
            return None

        data: Dict[str, Any] = {}

        # Inject the ``enabled`` flag from the QGroupBox checkbox.
        if self._has_enabled_field:
            data["enabled"] = self.isChecked()

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

            For models with an ``enabled`` field, the QGroupBox checked
            state is driven by ``data["enabled"]`` (defaulting to
            ``True`` when the key is absent).
        """
        if data is None:
            self.setChecked(False)
            return

        # Drive the QGroupBox checkbox from the ``enabled`` key when
        # the model declares one.
        if self._has_enabled_field:
            self.setChecked(data.get("enabled", True))
        else:
            self.setChecked(True)

        for name, widget in self._field_widgets.items():
            if name in data:
                widget.set_value(data[name])
        for name, section in self._nested_sections.items():
            section_data = data.get(name)
            section.set_values(section_data)

    def is_enabled(self) -> bool:
        """Return whether the section checkbox is checked.

        Returns
        -------
        bool
            ``True`` if the section is enabled.
        """
        return self.isChecked()

    def set_enabled(self, enabled: bool) -> None:
        """Check or uncheck the section.

        Parameters
        ----------
        enabled : bool
            Whether to enable the section.
        """
        self.setChecked(enabled)

    def validate(self) -> List[Tuple[str, str]]:
        """Validate current values through Pydantic.

        Calls ``model_class.model_validate()`` on the current widget
        values and returns any validation errors as a list of
        ``(field_name, error_message)`` tuples.

        Returns
        -------
        list of (str, str)
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
