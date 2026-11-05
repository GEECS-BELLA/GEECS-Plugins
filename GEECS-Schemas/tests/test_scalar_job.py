"""ScalarJob: the scalar-columns document, and how the loader tells it from the others."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY
from geecs_schemas.analysis import (
    HimgJob,
    ScalarJob,
    canonical_document,
    load_analysis_document,
)

JOB = {
    "kind": "scalar_columns",
    "columns": [
        {"name": "x (mm)", "expr": "x * 0.0244", "inputs": {"x": "Cam centroid_x"}},
        {
            "name": "ratio",
            "expr": "a * b / c",
            "inputs": {"a": "A", "b": "B", "c": "C"},
        },
    ],
}


def column(**over):
    return {**JOB, "columns": [{**JOB["columns"][0], **over}]}


def test_registered_and_dispatched_as_its_own_kind():
    assert SCHEMA_REGISTRY["scalar_job"] is ScalarJob
    job = load_analysis_document(JOB)
    assert type(job) is ScalarJob
    assert (
        type(load_analysis_document({"kind": "himg_compact", "device": "D"})) is HimgJob
    )


def test_host_facing_names():
    job = load_analysis_document(JOB)
    assert job.data_folder == ""
    assert job.destructive is False
    assert job.scan.priority == 1000
    job._source_id = "TopView_mm"
    assert job.effective_output_name == job.device == "TopView_mm"
    named = ScalarJob.model_validate({**JOB, "output_name": "mm"})
    assert named.effective_output_name == "mm"


@pytest.mark.parametrize(
    "expr",
    [
        "x * 0.0244",
        "sqrt(x**2 + 1) / pi",
        "-x % 3",
        "arctan2(x, 2) + abs(x)",
        "hypot(x, x)",
    ],
)
def test_accepted_formulas(expr):
    load_analysis_document(column(expr=expr))


@pytest.mark.parametrize(
    ("expr", "why"),
    [
        ("x.__class__", "disallowed|unknown"),
        ("__import__('os')", "only whitelisted"),
        ("y * 2", "unknown name 'y'"),
        ("x *", "syntax|invalid"),
        ("x > 1", "disallowed"),
        ("'a'", "not a number"),
    ],
)
def test_refused_formulas_fail_at_load(expr, why):
    with pytest.raises(ValidationError, match=why):
        load_analysis_document(column(expr=expr))


def test_symbols_must_be_names_and_not_functions():
    with pytest.raises(ValidationError, match="not a valid name"):
        load_analysis_document(column(expr="1", inputs={"a b": "C"}))
    with pytest.raises(ValidationError, match="function or constant"):
        load_analysis_document(column(expr="sqrt", inputs={"sqrt": "C"}))


def test_duplicate_column_names_refused():
    doc = {**JOB, "columns": [JOB["columns"][0], JOB["columns"][0]]}
    with pytest.raises(ValidationError, match="defined twice"):
        load_analysis_document(doc)


def test_a_misspelt_kind_names_the_scalar_kind():
    with pytest.raises(ValidationError, match="scalar_columns"):
        load_analysis_document({**JOB, "kind": "scalar_column"})


def test_the_canonical_form_round_trips():
    job = ScalarJob.model_validate(JOB)
    written = canonical_document(job)
    assert load_analysis_document(written) == job
