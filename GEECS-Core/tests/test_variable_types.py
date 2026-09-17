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


def test_variabletype_wins_over_an_option_list() -> None:
    # The rule the gateway serves with today.  18 Undulator rows carry
    # variabletype='numeric' + a filter-wheel list and SHOULD be 'choice';
    # that is a DB fix (variabletype), not a rule change — see the module doc.
    assert effective_vartype("numeric", "1,2,3,4,5,6") == "numeric"
    assert effective_vartype("choice", "on,off") == "choice"


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


def test_scalar_attribute_variables_numeric_subscribed_minus_ladder(caplog) -> None:
    """The PVA plugin's per-frame scalar filter (GEECS-Core 0.6.0)."""
    import logging

    from geecs_core.db.variable_types import (
        TIMESTAMP_LADDER,
        scalar_attribute_variables,
    )

    rows = [
        {"name": "MaxCounts", "variabletype": "numeric", "choices": None},
        {"name": "exposure", "variabletype": None, "choices": "numeric"},
        {
            "name": "trigger",
            "variabletype": "",
            "choices": "on,off",
        },  # enum: text label
        {"name": "localsavingpath", "variabletype": "string", "choices": None},
        {"name": "image", "variabletype": "image", "choices": None},
        {"name": "acq_timestamp", "variabletype": "numeric", "choices": None},
        {"name": "Mean Counts", "variabletype": "numeric", "choices": None},
        {"name": "mean_counts", "variabletype": "numeric", "choices": None},
    ]
    subscribed = [
        "acq_timestamp",
        "MaxCounts",
        "trigger",
        "localsavingpath",
        "image",
        "ghost",
        "exposure",
        "Mean Counts",
        "mean_counts",
        "MaxCounts",
    ]
    assert TIMESTAMP_LADDER == ("acq_timestamp", "systimestamp")
    # Without a normalizer: numeric, subscribed, not the ladder, DB order, deduped.
    assert scalar_attribute_variables(rows, subscribed) == [
        "MaxCounts",
        "exposure",
        "Mean Counts",
        "mean_counts",
    ]
    # With the naming contract: the second name onto one dataset is dropped, warned.
    with caplog.at_level(logging.WARNING):
        out = scalar_attribute_variables(
            rows, subscribed, normalize=lambda s: s.lower().replace(" ", "_")
        )
    assert out == ["MaxCounts", "exposure", "Mean Counts"]
    assert "mean_counts" in caplog.text and "normalizes to" in caplog.text
    assert scalar_attribute_variables([], subscribed) == []
    # Case: the DB spells the subscribed name differently from the metadata
    # row (the namespace lower-matches too); the metadata spelling is kept.
    assert scalar_attribute_variables(
        rows, ["maxcounts", "EXPOSURE", "Acq_Timestamp"], normalize=str.lower
    ) == ["MaxCounts", "exposure"]
