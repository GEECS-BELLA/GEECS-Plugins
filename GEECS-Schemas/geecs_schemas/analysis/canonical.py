"""The canonical on-disk form of an analysis document.

One rule for every writer (the config editor's store, a notebook): a document is written with the fields that were set, minus
``None`` values whose field default is already ``None``, and with
``schema_version`` first so a reader sees the format before the content.
Pydantic-only; YAML serialisation is the caller's.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

__all__ = ["canonical_document"]


def canonical_document(model: BaseModel) -> dict[str, Any]:
    """Return *model* as the mapping that should be written to disk.

    Parameters
    ----------
    model : pydantic.BaseModel
        A validated document (``AnalysisDiagnostic``, ``AnalysisGroup``).

    Returns
    -------
    dict
        JSON-mode dump of the set fields, default-``None`` noise removed,
        ``schema_version`` first when the model carries one.
    """
    dumped = model.model_dump(mode="json", exclude_unset=True)
    pruned = _prune_default_nones(model, dumped)
    version = getattr(model, "schema_version", None)
    if version is not None:
        pruned = {
            "schema_version": version,
            **{k: v for k, v in pruned.items() if k != "schema_version"},
        }
    return pruned


def _prune_default_nones(model: BaseModel, dumped: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in dumped.items():
        field = type(model).model_fields.get(name)
        if value is None and field is not None and field.default is None:
            continue
        child = getattr(model, name, None)
        if isinstance(child, BaseModel) and isinstance(value, dict):
            value = _prune_default_nones(child, value)
        out[name] = value
    return out
