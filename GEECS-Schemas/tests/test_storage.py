"""The registry's file / not-a-file table and the tiers it is read with."""

import importlib
import typing
from collections import Counter

import pytest

from geecs_schemas import (
    KIND_STORAGE,
    SCHEMA_REGISTRY,
    STORAGE_ROOT_PATHS,
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
    assert storage.root in STORAGE_ROOT_PATHS
    assert storage.suffixes[0] == ".yaml"
    assert set(storage.suffixes) <= {".yaml", ".yml"}
    assert storage.path_pattern().startswith(STORAGE_ROOT_PATHS[storage.root] + "/")


def test_the_named_kinds_read_by_the_resolver_accept_yml():
    """The resolver's named configs, the analysis trees and setups also read ``.yml``."""
    accepting = sorted(k for k in FILE_KINDS if ".yml" in KIND_STORAGE[k].suffixes)
    assert accepting == [
        "analysis_group",
        "analysis_recipe",
        "himg_job",
        "optimizer_config",
        "preset",
        "setup",
        "trigger_profile",
    ]


def test_kinds_sharing_a_folder_name_the_loader_that_tells_them_apart():
    """Two kinds in one folder each carry a dispatch loader, and it imports."""
    places = Counter(
        (s.root, s.folder) for s in KIND_STORAGE.values() if isinstance(s, FileStorage)
    )
    for kind in FILE_KINDS:
        storage = KIND_STORAGE[kind]
        shared = places[(storage.root, storage.folder)] > 1 and storage.folder
        if not shared:
            continue
        assert storage.dispatch, kind
        module, _, name = storage.dispatch.rpartition(".")
        assert callable(getattr(importlib.import_module(module), name))
    assert KIND_STORAGE["analysis_recipe"].dispatch == KIND_STORAGE["himg_job"].dispatch


def test_path_pattern():
    assert KIND_STORAGE["preset"].path_pattern() == (
        "scanner_configs/experiments/<experiment>/presets/<name>.yaml"
    )
    assert KIND_STORAGE["home_layout"].path_pattern() == (
        "scanner_configs/experiments/<experiment>/home.yaml"
    )
    assert KIND_STORAGE["analysis_group"].path_pattern() == (
        "scan_analysis_configs/groups/**/<name>.yaml"
    )


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
