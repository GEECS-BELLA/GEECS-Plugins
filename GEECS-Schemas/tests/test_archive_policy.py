"""Model tests for the Archiver Appliance curation overlay."""

import pytest
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY, ArchivePolicy, SamplingOverride


def test_defaults_are_the_derived_set_sampled_each_second():
    policy = ArchivePolicy()
    assert policy.schema_version == 1
    assert policy.exclude == [] and policy.include == []
    assert policy.include_setpoints and policy.include_status and policy.include_derived
    assert policy.default_sampling_period == 1.0
    assert policy.default_sampling_method == "MONITOR"
    assert policy.sampling_overrides == []


def test_full_document_round_trips():
    doc = {
        "schema_version": 1,
        "exclude": ["undulator:uc_*:image_size*"],
        "include": ["undulator:cagateway:devices_connected"],
        "include_setpoints": False,
        "sampling_overrides": [
            {"match": "undulator:u_vacuumgauge:*", "sampling_period": 10.0},
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


def test_override_rejects_non_positive_period_unknown_method_and_named_policies():
    with pytest.raises(ValidationError):
        SamplingOverride(match="x", sampling_period=0)
    with pytest.raises(ValidationError):
        SamplingOverride(match="x", sampling_method="RANDOM")
    with pytest.raises(
        ValidationError
    ):  # named policies were removed: the file is the one table
        SamplingOverride.model_validate({"match": "x", "policy": "Slow"})


def test_unknown_keys_are_refused():
    with pytest.raises(ValidationError):
        ArchivePolicy.model_validate({"includes": ["anything"]})


def test_from_path_yaml_json_and_empty(tmp_path):
    y = tmp_path / "p.yaml"
    y.write_text("schema_version: 1\ninclude_setpoints: false\n")
    assert ArchivePolicy.from_path(y).include_setpoints is False
    j = tmp_path / "p.json"
    j.write_text('{"exclude": ["a:*"]}')
    assert ArchivePolicy.from_path(j).exclude == ["a:*"]
    e = tmp_path / "empty.yaml"
    e.write_text("")
    assert ArchivePolicy.from_path(e) == ArchivePolicy()


def test_registered_as_a_config_kind():
    assert SCHEMA_REGISTRY["archive_policy"] is ArchivePolicy
