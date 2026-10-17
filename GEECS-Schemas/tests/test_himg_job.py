"""HimgJob: the ``.himg`` job document, and how the loader tells it from a recipe."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY
from geecs_schemas.analysis import (
    AnalysisRecipe,
    HimgJob,
    canonical_document,
    load_analysis_document,
)

JOB = {"schema_version": 1, "kind": "himg_compact", "device": "U_HasoLift"}
V3 = {
    "schema_version": 3,
    "device": "Cam",
    "input": {"kind": "camera"},
    "measure": {"kind": "beam"},
}


def test_registered_as_its_own_document_kind():
    assert SCHEMA_REGISTRY["himg_job"] is HimgJob


def test_each_document_of_the_tree_loads_as_itself():
    assert type(load_analysis_document(JOB)) is HimgJob
    assert type(load_analysis_document(V3)) is AnalysisRecipe


@pytest.mark.parametrize("kind", ["himg_to_stack", "himg_compact", "himg_restore"])
def test_the_compaction_alone_is_destructive(kind):
    job = load_analysis_document({**JOB, "kind": kind})
    assert job.destructive is (kind == "himg_compact")
    assert job.data_folder == job.device == "U_HasoLift"
    assert job.scan.priority == 100


def test_names_default_like_the_analysis_documents():
    job = HimgJob.model_validate({**JOB, "output_name": "HasoLift_compact"})
    assert job.effective_output_name == "HasoLift_compact"
    assert HimgJob.model_validate(JOB).effective_output_name == "U_HasoLift"


def test_the_kind_is_read_first_so_a_typo_names_the_job():
    with pytest.raises(ValidationError, match="himg_to_stack"):
        load_analysis_document({**JOB, "kind": "himg_compactt"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_analysis_document({**JOB, "file_tail": ".himg"})


def test_the_canonical_form_round_trips():
    job = HimgJob.model_validate({**JOB, "scan": {"priority": 6}})
    written = canonical_document(job)
    assert list(written)[:2] == ["schema_version", "kind"]
    assert load_analysis_document(written) == job
