"""Diagnostic read boundary shared by numerical and live consumers."""

import pytest
import yaml

from geecs_data_utils.analysis_configs import (
    deep_merge as _deep_merge,
    discover_diagnostics,
    read_diagnostic,
    read_yaml_mapping,
)


class TestDeepMerge:
    """Recursive dict merge: nested mappings merge key-by-key, scalars replace."""

    def test_disjoint_keys_union(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_overlay_scalar_replaces_base_scalar(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_dicts_merge_key_by_key(self):
        base = {"scan": {"mode": "per_shot", "priority": 100}}
        overlay = {"scan": {"mode": "per_bin"}}
        assert _deep_merge(base, overlay) == {
            "scan": {"mode": "per_bin", "priority": 100}
        }

    def test_three_level_nesting(self):
        base = {"image": {"background": {"method": "constant", "value": 0.0}}}
        overlay = {"image": {"background": {"value": 12.5}}}
        assert _deep_merge(base, overlay) == {
            "image": {"background": {"method": "constant", "value": 12.5}}
        }

    def test_lists_replace_wholesale(self):
        assert _deep_merge({"steps": [1, 2, 3]}, {"steps": [9]}) == {"steps": [9]}

    def test_none_replaces_base(self):
        assert _deep_merge({"a": 1}, {"a": None}) == {"a": None}

    def test_returns_new_dict_does_not_mutate_inputs(self):
        base = {"scan": {"mode": "per_shot"}}
        overlay = {"scan": {"mode": "per_bin"}}
        _deep_merge(base, overlay)
        assert base == {"scan": {"mode": "per_shot"}}
        assert overlay == {"scan": {"mode": "per_bin"}}

    def test_overlay_can_introduce_new_nested_key(self):
        base = {"scan": {"mode": "per_shot"}}
        overlay = {"scan": {"gdoc_slot": 2}}
        assert _deep_merge(base, overlay) == {
            "scan": {"mode": "per_shot", "gdoc_slot": 2}
        }


def test_discovery_and_load_are_fresh_and_leave_tree_untouched(tmp_path):
    folder = tmp_path / "analyzers" / "experiment"
    folder.mkdir(parents=True)
    path = folder / "profile.yml"
    path.write_text(yaml.safe_dump({"name": "camera", "image": {"pipeline": ["roi"]}}))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    found, data = read_diagnostic(
        "profile", config_dir=tmp_path, overrides={"image": {"pipeline": []}}
    )
    assert found == path and data["image"]["pipeline"] == []
    assert discover_diagnostics(tmp_path) == {"profile": path}
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    path.write_text("name: changed\n")
    assert read_diagnostic("profile", config_dir=tmp_path)[1] == {"name": "changed"}


def test_overrides_and_base_do_not_alias_nested_inputs():
    base = {"keep": [1], "nested": {"a": [2]}}
    overrides = {"nested": {"b": [3]}}
    merged = _deep_merge(base, overrides)
    merged["keep"].append(4)
    merged["nested"]["a"].append(5)
    merged["nested"]["b"].append(6)
    assert base == {"keep": [1], "nested": {"a": [2]}}
    assert overrides == {"nested": {"b": [3]}}


def test_duplicate_stems_are_ambiguous_across_extensions_and_subfolders(tmp_path):
    for name in ["one/profile.yaml", "two/profile.yml"]:
        path = tmp_path / "analyzers" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("name: camera\n")
    with pytest.raises(ValueError, match="Duplicate diagnostic ID"):
        read_diagnostic("profile", config_dir=tmp_path)


def test_explicit_path_needs_no_config_root(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text("name: camera\n")
    assert read_diagnostic(path) == (path, {"name": "camera"})
    with pytest.raises(ValueError, match="config_dir is required"):
        read_diagnostic("profile")


def test_missing_paths_never_create_directories(tmp_path):
    missing = tmp_path / "scans" / "Scan001"
    with pytest.raises(FileNotFoundError):
        read_diagnostic(missing / "profile.yaml")
    with pytest.raises(FileNotFoundError):
        discover_diagnostics(missing)
    assert not missing.exists()


def test_empty_and_non_mapping_yaml_are_explicit(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert read_yaml_mapping(path) == {}
    path.write_text("- not\n- a mapping\n")
    with pytest.raises(ValueError, match="Expected a YAML mapping"):
        read_yaml_mapping(path)
