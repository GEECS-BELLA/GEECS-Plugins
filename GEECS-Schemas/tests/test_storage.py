"""The registry's file / not-a-file table and the tiers it is read with."""

import typing

import pytest

from geecs_schemas import (
    KIND_STORAGE,
    SCHEMA_REGISTRY,
    TIERS,
    ArchivePolicy,
    FileStorage,
    NotAFile,
    Tier,
)
from geecs_schemas.schema_export import EXPORTED_SCHEMAS

FILE_KINDS = sorted(k for k, s in KIND_STORAGE.items() if isinstance(s, FileStorage))
NOT_FILES = sorted(k for k, s in KIND_STORAGE.items() if isinstance(s, NotAFile))


def test_the_table_names_exactly_the_registry_kinds():
    """Every registered kind is in the table once, and the table names nothing else."""
    assert set(KIND_STORAGE) == set(SCHEMA_REGISTRY)
    assert len(KIND_STORAGE) == len(SCHEMA_REGISTRY)


def test_the_non_file_kinds():
    """The kinds that are not configs-repo files, each with its reason."""
    assert NOT_FILES == [
        "action_plan",
        "family_file",
        "journal_entry",
        "log_entry",
        "scan_request",
    ]
    for kind in NOT_FILES:
        assert KIND_STORAGE[kind].reason.strip()


def test_tiers_vocabulary_is_the_literal():
    """``TIERS`` lists the ``Tier`` literal's values, least restricted first."""
    assert TIERS == typing.get_args(Tier)
    assert TIERS[0] == "operator"


@pytest.mark.parametrize("kind", FILE_KINDS)
def test_every_file_kind_has_a_tier_and_a_place(kind):
    """A file kind declares its tier, and the table its root, folder and pattern."""
    model = SCHEMA_REGISTRY[kind]
    storage = KIND_STORAGE[kind]
    assert model.tier in TIERS
    assert storage.pattern.endswith(".yaml")
    assert "/" not in storage.pattern and ".." not in storage.folder
    assert storage.root in ("experiment", "analysis")


@pytest.mark.parametrize("kind", sorted(SCHEMA_REGISTRY))
def test_the_tier_reaches_the_json_schema(kind):
    """The ClassVar and ``model_json_schema()`` agree; a kind with no tier has no key."""
    model = SCHEMA_REGISTRY[kind]
    schema = model.model_json_schema()
    if model.tier is None:
        assert "tier" not in schema
    else:
        assert schema["tier"] == model.tier


def test_non_file_kinds_carry_no_tier_but_family_files_are_code():
    """A database row or a parameter is edited by nobody; family files are code."""
    for kind in NOT_FILES:
        expected = "code" if kind == "family_file" else None
        assert SCHEMA_REGISTRY[kind].tier == expected


def test_the_maintainer_documents():
    """The brief's maintainer tier: the policy and the device registries."""
    maintainer = sorted(
        k for k in FILE_KINDS if SCHEMA_REGISTRY[k].tier == "maintainer"
    )
    assert maintainer == [
        "derived_channels",
        "native_devices",
        "operator_policy",
        "panel_devices",
    ]


def test_the_archive_policy_membership_rules_are_the_maintainers():
    """Include and exclude (and the derived-rule switches) are stricter than the document."""
    properties = ArchivePolicy.model_json_schema()["properties"]
    maintainer = {
        name for name, p in properties.items() if p.get("tier") == "maintainer"
    }
    assert maintainer == {
        "exclude",
        "include",
        "include_setpoints",
        "include_status",
        "include_derived",
    }
    assert ArchivePolicy.tier == "operator"
    assert "tier" not in properties["sampling_overrides"]


def test_every_file_kind_is_published():
    """``schema_export`` publishes every file kind under its registry name."""
    for kind in FILE_KINDS:
        assert EXPORTED_SCHEMAS[kind] is SCHEMA_REGISTRY[kind]
