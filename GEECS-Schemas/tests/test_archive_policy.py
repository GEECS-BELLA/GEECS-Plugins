"""Model tests for the Archiver Appliance curation overlay."""

import pytest
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY, ArchivePolicy, SamplingOverride


def test_defaults_are_the_derived_set_with_the_default_policy():
    policy = ArchivePolicy()
    assert policy.schema_version == 1
    assert policy.exclude == []
    assert policy.include_setpoints and policy.include_status and policy.include_derived
    assert policy.default_sampling_period == 1.0
    assert policy.default_sampling_method == "MONITOR"
    assert policy.sampling_overrides == []


def test_full_document_round_trips():
    doc = {
        "schema_version": 1,
        "exclude": ["undulator:uc_*:image_size*"],
        "include_setpoints": False,
        "sampling_overrides": [
            {"match": "undulator:u_vacuumgauge:*", "policy": "Slow"},
            {
                "match": "undulator:u_s1h:*",
                "sampling_period": 0.5,
                "sampling_method": "SCAN",
            },
        ],
    }
    policy = ArchivePolicy.model_validate(doc)
    assert policy.model_dump(exclude_defaults=True) == {
        k: v for k, v in doc.items() if k != "schema_version"
    }
    assert policy.sampling_overrides[1].sampling_method == "SCAN"


def test_override_must_set_something():
    with pytest.raises(ValidationError, match="sets nothing"):
        SamplingOverride(match="undulator:*")


def test_override_rejects_non_positive_period_and_unknown_method():
    with pytest.raises(ValidationError):
        SamplingOverride(match="x", sampling_period=0)
    with pytest.raises(ValidationError):
        SamplingOverride(match="x", sampling_method="RANDOM")


def test_unknown_keys_are_refused():
    with pytest.raises(ValidationError):
        ArchivePolicy.model_validate({"include": ["anything"]})


def test_registered_as_a_config_kind():
    assert SCHEMA_REGISTRY["archive_policy"] is ArchivePolicy
