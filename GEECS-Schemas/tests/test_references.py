"""Which documents name which: the table a delete's check and the configs repo's gate read."""

from __future__ import annotations

from pathlib import Path

import pytest

from geecs_schemas import (
    KIND_STORAGE,
    REFERS_TO,
    SCHEMA_REGISTRY,
    FileStorage,
    references,
)


def _doc(kind: str, raw: dict):
    return SCHEMA_REGISTRY[kind].model_validate(raw)


def test_the_table_names_file_kinds_only() -> None:
    for referrer, targets in REFERS_TO.items():
        assert isinstance(KIND_STORAGE[referrer], FileStorage), referrer
        assert all(isinstance(KIND_STORAGE[t], FileStorage) for t in targets), referrer


@pytest.mark.parametrize(
    "kind,raw,named",
    [
        (
            "preset",
            {
                "name": "p",
                "trigger_profile": "laser_off",
                "devices": [],
                "plan": {"name": "count", "kwargs": {}},
            },
            [("trigger_profile", "laser_off")],
        ),  # fmt: skip
        (
            "experiment_defaults",
            {"trigger_profile": "standard"},
            [("trigger_profile", "standard")],
        ),  # fmt: skip
        (
            "setup",
            {
                "name": "a",
                "steps": [
                    {"do": "apply", "setup": "b"},
                    {"do": "wait", "seconds": 1},
                    {"do": "apply", "setup": "b"},
                ],
            },
            [("setup", "b")],
        ),  # fmt: skip
        (
            "home_layout",
            {"areas": [{"title": "Laser", "screens": ["amp", "stretcher"]}]},
            [("screen", "amp"), ("screen", "stretcher")],
        ),  # fmt: skip
        (
            "analysis_group",
            {
                "name": "g",
                "analyzers": ["UC_Top", {"ref": "UC_Side", "enabled": False}],
            },
            [("analysis_recipe", "UC_Top"), ("analysis_recipe", "UC_Side")],
        ),  # fmt: skip
    ],
)
def test_each_referrer_names_what_it_reads(kind: str, raw: dict, named: list) -> None:
    assert references(_doc(kind, raw)) == named


def test_a_screen_names_the_setups_its_buttons_apply() -> None:
    screen = _doc("screen", {"title": "S", "groups": [{"title": "G", "at": [1, 1], "widgets": [{"kind": "button", "label": "Go", "bind": {"action": "high_power"}}]}]})  # fmt: skip
    assert references(screen) == [("setup", "high_power")]


def test_an_optimize_preset_names_its_optimizer_config_and_that_its_diagnostics() -> (
    None
):
    import yaml

    preset = _doc("preset", {"name": "o", "devices": [], "plan": {"name": "optimize", "kwargs": {"optimizer_config": "bax_alignment_S1V"}}})  # fmt: skip
    assert references(preset) == [("optimizer_config", "bax_alignment_S1V")]
    fixture = (
        Path(__file__).parent / "fixtures/optimizer_configs/bax_alignment_S1V.yaml"
    )
    config = _doc("optimizer_config", yaml.safe_load(fixture.read_text()))
    named = references(config)
    assert named and all(kind == "analysis_recipe" for kind, _ in named)


def test_a_preset_without_a_plan_names_only_its_profile() -> None:
    preset = _doc("preset", {"name": "p", "trigger_profile": "t", "devices": []})
    assert references(preset) == [("trigger_profile", "t")]


def test_a_document_naming_nothing_names_nothing() -> None:
    assert references(_doc("experiment_defaults", {})) == []
    assert references(_doc("setup", {"name": "a", "steps": [{"do": "wait", "seconds": 1}]})) == []  # fmt: skip
