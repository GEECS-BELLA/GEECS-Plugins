"""effective_vartype — the one DB-type rule (moved from geecs_ca_gateway.config)."""

from __future__ import annotations

from geecs_core.db.variable_types import (
    CHOICE_TYPE_DESCRIPTORS,
    SKIP_VARTYPES,
    VARTYPE_TO_DTYPE,
    effective_vartype,
    is_scalar_vartype,
)


def test_bare_descriptor_in_choices_is_authoritative() -> None:
    # variabletype='choice' + choices='image' is an image, not a one-option enum
    assert effective_vartype("choice", "image") == "image"
    assert effective_vartype("choice", "1darray") == "1darray"
    assert effective_vartype(None, "numeric") == "numeric"
    assert effective_vartype("", "path") == "path"
    assert effective_vartype("string", "path") == "path"


def test_an_option_list_is_always_an_enum() -> None:
    # 18 Undulator rows: variabletype='numeric' with a filter-wheel style list —
    # the options name configurations, not values (Sam, 2026-09-09)
    assert effective_vartype("numeric", "1,2,3,4,5,6") == "choice"
    assert effective_vartype("choice", "on,off") == "choice"
    assert effective_vartype("string", "a,b") == "choice"


def test_blank_variabletype_falls_back_on_choices_shape() -> None:
    assert effective_vartype(None, "on,off") == "choice"
    assert effective_vartype(None, "Error,Moving,OK") == "choice"
    assert effective_vartype(None, None) == "numeric"
    assert effective_vartype("  ", "") == "numeric"


def test_normalisation() -> None:
    assert effective_vartype(" Numeric ", None) == "numeric"
    assert effective_vartype(None, " STRING ") == "string"


def test_tables_are_consistent() -> None:
    assert SKIP_VARTYPES <= CHOICE_TYPE_DESCRIPTORS
    assert set(VARTYPE_TO_DTYPE) == {"numeric", "string", "path", "choice"}
    assert not (set(VARTYPE_TO_DTYPE) & SKIP_VARTYPES)
    assert is_scalar_vartype("numeric") and not is_scalar_vartype("image")
