"""Pydantic model introspection for GUI field generation.

Walks model_fields of a Pydantic v2 BaseModel and returns FieldDescriptor
objects that the widget layer turns into PyQt5 controls.  No Qt imports here.
"""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


class FieldKind(str, Enum):
    """Logical type category for a model field."""

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"  # includes Path
    ENUM = "enum"
    LITERAL = "literal"  # Literal["value"] — hidden in GUI
    NESTED = "nested"  # required BaseModel sub-field
    OPTIONAL_MODEL = "optional_model"  # Optional[BaseModel] — collapsible group
    LIST_MODEL = "list_model"  # List[BaseModel]
    LIST_SCALAR = "list_scalar"  # List[primitive]
    TUPLE = "tuple"  # Tuple[int, int] or Tuple[float, float]
    DICT = "dict"  # Dict[str, Any] — raw YAML editor
    UNKNOWN = "unknown"  # fall back to YAML text box


@dataclass
class FieldDescriptor:
    """All metadata needed to build a widget for one model field."""

    name: str
    kind: FieldKind
    annotation: Any  # raw annotation from model_fields
    default: Any = None
    description: str = ""
    is_required: bool = False
    is_optional: bool = False  # True if None is a valid value
    enum_class: Optional[Type[Enum]] = None
    inner_model: Optional[Type[BaseModel]] = (
        None  # for NESTED / OPTIONAL_MODEL / LIST_MODEL
    )
    scalar_type: type = str  # for LIST_SCALAR / TUPLE elements
    literal_values: List[Any] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)  # ge, le, gt, lt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def describe_model(model_class: Type[BaseModel]) -> List[FieldDescriptor]:
    """Return a descriptor for every field in *model_class*."""
    descriptors = []
    for name, fi in model_class.model_fields.items():
        descriptors.append(_describe_field(name, fi))
    return descriptors


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Serialise a model instance to a plain dict (exclude-None, enum values)."""
    return model.model_dump(mode="python", exclude_none=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _describe_field(name: str, fi: FieldInfo) -> FieldDescriptor:
    annotation = fi.annotation
    default = fi.default if fi.default is not PydanticUndefined else None
    description = fi.description or ""
    is_required = fi.is_required()

    # Collect pydantic numeric constraints
    constraints = {}
    if fi.metadata:
        for m in fi.metadata:
            for attr in ("ge", "le", "gt", "lt", "multiple_of"):
                if hasattr(m, attr) and getattr(m, attr) is not None:
                    constraints[attr] = getattr(m, attr)

    # Unwrap the annotation
    kind, extra = _classify(annotation)

    desc = FieldDescriptor(
        name=name,
        kind=kind,
        annotation=annotation,
        default=default,
        description=description,
        is_required=is_required,
        constraints=constraints,
    )

    # Fill kind-specific extras
    if kind == FieldKind.ENUM:
        desc.enum_class = extra
    elif kind in (FieldKind.NESTED, FieldKind.OPTIONAL_MODEL, FieldKind.LIST_MODEL):
        desc.inner_model = extra
        desc.is_optional = kind == FieldKind.OPTIONAL_MODEL
    elif kind == FieldKind.LIST_SCALAR:
        desc.scalar_type = extra or str
    elif kind == FieldKind.TUPLE:
        desc.scalar_type = extra or int
    elif kind == FieldKind.LITERAL:
        desc.literal_values = list(get_args(annotation))
    elif kind in (FieldKind.BOOL, FieldKind.INT, FieldKind.FLOAT, FieldKind.STR):
        pass

    # Optional primitives
    if kind not in (FieldKind.NESTED,) and _is_optional(annotation):
        desc.is_optional = True

    return desc


def _classify(annotation: Any) -> Tuple[FieldKind, Any]:
    """Return (FieldKind, extra_info) for a raw annotation."""
    # None / NoneType
    if annotation is type(None):
        return FieldKind.UNKNOWN, None

    # Strip Optional (Union[X, None])
    inner = _unwrap_optional(annotation)
    was_optional = inner is not annotation
    ann = inner  # work on the unwrapped version

    origin = get_origin(ann)
    args = get_args(ann)

    # Literal
    if _is_literal(ann):
        return FieldKind.LITERAL, None

    # bool must be before int (bool is subclass of int)
    if ann is bool:
        return FieldKind.BOOL, None

    if ann in (int,):
        return FieldKind.INT, None

    if ann in (float,):
        return FieldKind.FLOAT, None

    if ann in (str, Path) or (isinstance(ann, type) and issubclass(ann, Path)):
        return FieldKind.STR, None

    # Enum
    if isinstance(ann, type) and issubclass(ann, Enum) and ann is not Enum:
        return FieldKind.ENUM, ann

    # Pydantic BaseModel
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        if was_optional:
            return FieldKind.OPTIONAL_MODEL, ann
        return FieldKind.NESTED, ann

    # List
    if origin is list:
        if args:
            item = args[0]
            if isinstance(item, type) and issubclass(item, BaseModel):
                return FieldKind.LIST_MODEL, item
            return FieldKind.LIST_SCALAR, item
        return FieldKind.LIST_SCALAR, str

    # Tuple
    if origin is tuple:
        elem_type = args[0] if args else int
        return FieldKind.TUPLE, elem_type

    # Dict
    if origin is dict:
        return FieldKind.DICT, None

    # Union[str, Path] → treat as STR
    if _is_union_type(ann):
        non_none = [a for a in args if a is not type(None)]
        if all(
            a in (str, Path) or (isinstance(a, type) and issubclass(a, Path))
            for a in non_none
        ):
            return FieldKind.STR, None

    return FieldKind.UNKNOWN, None


def _is_union_type(annotation: Any) -> bool:
    """Return True if annotation is any form of Union."""
    origin = get_origin(annotation)
    if origin is typing.Union:
        return True
    if hasattr(types, "UnionType") and isinstance(annotation, types.UnionType):
        return True
    return False


def _unwrap_optional(annotation: Any) -> Any:
    """Return inner type if annotation is Optional[X], else return annotation."""
    if _is_union_type(annotation):
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _is_optional(annotation: Any) -> bool:
    return _unwrap_optional(annotation) is not annotation


def _is_literal(annotation: Any) -> bool:
    return get_origin(annotation) is typing.Literal
