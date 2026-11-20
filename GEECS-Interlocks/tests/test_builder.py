"""The builder's draft: grouping, editing, validation, the markers, saving."""

import pytest
import yaml
from conftest import interlock_set, threshold
from pydantic import ValidationError

from geecs_interlocks.builder import Draft, to_yaml
from geecs_interlocks.catalog import StaticCatalog, named_in
from geecs_interlocks.config import load_interlocks

SAT = {
    "kind": "threshold",
    "device": "Cam",
    "variable": "Max",
    "operator": ">=",
    "limit": 4000,
}
EXT = {"kind": "external", "port": 5050}


@pytest.fixture
def draft():
    return Draft(interlock_set(threshold()))


class TestGrouping:
    def test_add_to_an_interlock_joins_its_group(self, draft):
        assert draft.add_check(SAT, to="vac") == ("vac", 1)
        assert len(draft.interlocks[0]["conditions"]) == 2
        assert draft.document()

    def test_add_to_a_new_interlock(self, draft):
        assert draft.add_check(SAT, new_name=" cam ") == ("cam", 0)
        assert draft.names() == ["vac", "cam"]

    @pytest.mark.parametrize("name", ["", "vac"])
    def test_a_new_interlock_needs_a_fresh_name(self, draft, name):
        with pytest.raises(ValueError):
            draft.add_check(SAT, new_name=name)

    def test_move_regroups_and_drops_an_emptied_interlock(self, draft):
        draft.add_check(SAT, new_name="cam")
        assert draft.move_check("cam", 0, "vac") == ("vac", 1)
        assert draft.names() == ["vac"]

    def test_move_to_an_unknown_interlock_changes_nothing(self, draft):
        before = draft.interlocks
        with pytest.raises(KeyError):
            draft.move_check("vac", 0, "nope")
        assert draft.interlocks == before

    def test_ungroup_splits_in_place(self, draft):
        draft.add_check(SAT, to="vac")
        draft.add_check(EXT, new_name="after")
        assert draft.ungroup("vac") == ["vac 1", "vac 2"]
        assert draft.names() == ["vac 1", "vac 2", "after"]

    def test_removing_the_last_check_removes_the_interlock(self, draft):
        draft.remove_check("vac", 0)
        assert draft.names() == []


class TestEditing:
    def test_replace_check(self, draft):
        draft.replace_check("vac", 0, {**draft.check("vac", 0), "limit": 20})
        assert draft.document().interlocks[0].conditions[0].limit == 20

    def test_enable_rename_settle_and_order(self, draft):
        draft.add_check(SAT, new_name="cam")
        draft.set_enabled("vac", False)
        draft.rename("cam", "camera")
        draft.set_settle("camera", 2.5)
        draft.move("camera", -1)
        document = draft.document()
        assert [i.name for i in document.interlocks] == ["camera", "vac"]
        assert document.interlocks[0].settle_s == 2.5
        assert not document.interlocks[1].enabled

    def test_rename_refuses_a_taken_name(self, draft):
        draft.add_check(SAT, new_name="cam")
        with pytest.raises(ValueError):
            draft.rename("cam", "vac")

    def test_an_invalid_draft_does_not_validate(self, draft):
        draft.replace_check("vac", 0, {**draft.check("vac", 0), "operator": "=="})
        draft.replace_check("vac", 0, {**draft.check("vac", 0), "hysteresis": 1})
        with pytest.raises(ValidationError):
            draft.document()

    def test_dirty_and_revert(self, draft):
        assert not draft.dirty
        draft.add_check(SAT, to="vac")
        assert draft.dirty
        draft.revert()
        assert not draft.dirty and len(draft.interlocks[0]["conditions"]) == 1


class TestMarkers:
    def test_watched_names_each_variable_s_interlocks(self, draft):
        draft.add_check(SAT, to="vac")
        draft.add_check({**SAT, "limit": 100}, new_name="cam")
        watched = draft.watched()
        assert watched["Gauge"] == {"P": ["vac"]}
        assert watched["Cam"] == {"Max": ["vac", "cam"]}

    def test_disabled_interlocks_still_count(self, draft):
        draft.set_enabled("vac", False)
        assert "Gauge" in draft.watched()

    def test_ports(self, draft):
        draft.add_check(EXT, new_name="stage")
        assert draft.ports() == {5050: ["stage"]}


class TestSaving:
    def test_save_round_trips_and_leaves_defaults_out(self, draft, tmp_path):
        draft.add_check(SAT, to="vac")
        path = tmp_path / "interlocks" / "main.yaml"
        saved = draft.save(path)
        assert load_interlocks(path)[0] == saved
        text = path.read_text()
        assert text.startswith("schema_version: 1")
        assert "hysteresis" not in text and "enabled" not in text
        assert not draft.dirty
        assert not list(path.parent.glob(".*tmp"))

    def test_an_invalid_draft_writes_nothing(self, draft, tmp_path):
        path = tmp_path / "main.yaml"
        path.write_text("keep me")
        draft.replace_check("vac", 0, {"kind": "nonsense"})
        with pytest.raises(ValidationError):
            draft.save(path)
        assert path.read_text() == "keep me"

    def test_diff_against_the_file(self, draft, tmp_path):
        path = tmp_path / "main.yaml"
        draft.save(path)
        draft.add_check(SAT, to="vac")
        diff = draft.diff(path)
        assert "+    variable: Max" in diff and diff.startswith("---")

    def test_to_yaml_is_loadable(self):
        doc = interlock_set(threshold())
        assert yaml.safe_load(to_yaml(doc))["interlocks"][0]["name"] == "vac"


class TestCatalog:
    def test_static_catalog_and_merge(self):
        catalog = StaticCatalog({"B": ["y", "x"], "A": []}).merged(
            {"A": ["z"], "C": ["w"]}
        )
        assert catalog.devices() == ["A", "B", "C"]
        assert catalog.variables("B") == ["x", "y"]
        assert catalog.variables("A") == ["z"]
        assert catalog.variables("nope") == []

    def test_named_in_covers_every_device_kind(self):
        doc = interlock_set(
            {
                "name": "a",
                "enabled": False,
                "conditions": [
                    SAT,
                    {
                        "kind": "alignment",
                        "device": "Cam",
                        "variable": "X",
                        "target_device": "Ref",
                        "target_variable": "X0",
                        "tolerance": 1,
                    },
                    {"kind": "config_change", "device": "W", "variables": ["P"]},
                    {"kind": "stale", "device": "G"},
                    EXT,
                ],
            }
        )
        assert named_in(doc) == {
            "Cam": {"Max", "X"},
            "Ref": {"X0"},
            "W": {"P"},
            "G": set(),
        }
