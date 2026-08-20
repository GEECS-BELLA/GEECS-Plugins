"""Widget tooltips derived from pydantic ``Field(description=...)`` texts.

The geecs-schemas models carry an operator-facing description on every
field; the editors' form widgets show those descriptions as tooltips so the
GUI text and the schema documentation are one source of truth (issue #497
phase 1).  A widget whose meaning matches a schema field must get its
tooltip through :func:`apply_schema_tooltips` — never a hand-written copy
that can drift.  Hand-written tooltips are reserved for pure GUI concepts
with no schema counterpart (buttons, lists, status chips).
"""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel


def schema_tooltip(model_cls: Type[BaseModel], field_name: str) -> str:
    """Return the tooltip text for one schema field.

    Parameters
    ----------
    model_cls : type
        The pydantic model that declares the field.
    field_name : str
        The field's Python name (not its alias).

    Returns
    -------
    str
        The field's ``description``.

    Raises
    ------
    KeyError
        When the model has no such field (a typo here, or a renamed field).
    LookupError
        When the field exists but carries no description — add one to the
        schema (the single source of truth) instead of hardcoding GUI text.
    """
    description = model_cls.model_fields[field_name].description
    if not description:
        raise LookupError(
            f"{model_cls.__name__}.{field_name} has no Field description — "
            "add one in geecs-schemas rather than hardcoding tooltip text."
        )
    return description


def apply_schema_tooltips(model_cls: Type[BaseModel], widgets: dict) -> None:
    """Set each widget's tooltip from its schema field's description.

    Parameters
    ----------
    model_cls : type
        The pydantic model the widgets edit.
    widgets : dict
        ``{field_name: widget}`` — or ``{field_name: [widget, ...]}`` when
        several widgets edit one field (e.g. a Device/Variable edit pair
        that together form a ``Device:Variable`` target).

    Raises
    ------
    KeyError, LookupError
        Propagated from :func:`schema_tooltip` — a bad mapping fails loudly
        at editor construction, which the editor tests exercise.
    """
    for field_name, widget in widgets.items():
        text = schema_tooltip(model_cls, field_name)
        targets = widget if isinstance(widget, (list, tuple)) else (widget,)
        for target in targets:
            target.setToolTip(text)
