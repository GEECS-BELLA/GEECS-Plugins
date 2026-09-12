"""Model tests for ActionPlan and ActionPlanLibrary."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from geecs_schemas import ActionPlan, ActionPlanLibrary, CheckStep, SetStep

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"


def make_plan():
    return ActionPlan.model_validate(
        {
            "steps": [
                {
                    "do": "set",
                    "device": "U_HP_Daq",
                    "variable": "AnalogOutput.Channel 1",
                    "value": 0,
                },
                {"do": "wait", "seconds": 3},
                {
                    "do": "check",
                    "device": "U_148_PLC",
                    "variable": "DI.Ch17",
                    "expected": "off",
                },
                {"do": "run", "plan": "close_gaia_internal_shutters"},
            ]
        }
    )


class TestActionPlan:
    def test_round_trip(self):
        plan = make_plan()
        again = ActionPlan.model_validate(plan.model_dump(mode="json"))
        assert again == plan

    def test_step_types_resolve(self):
        plan = make_plan()
        assert isinstance(plan.steps[0], SetStep)
        assert isinstance(plan.steps[2], CheckStep)

    def test_set_defaults_to_wait_for_execution(self):
        plan = make_plan()
        assert plan.steps[0].wait_for_execution is True

    def test_value_types_preserved(self):
        plan = make_plan()
        assert plan.steps[0].value == 0  # int survives
        assert plan.steps[2].expected == "off"  # str survives

    def test_unknown_step_kind_fails_loudly(self):
        with pytest.raises(ValidationError, match="do"):
            ActionPlan.model_validate({"steps": [{"do": "sleep", "seconds": 1}]})

    def test_unknown_step_field_fails_loudly(self):
        with pytest.raises(ValidationError, match="secconds"):
            ActionPlan.model_validate({"steps": [{"do": "wait", "secconds": 1}]})

    def test_nonpositive_wait_rejected(self):
        with pytest.raises(ValidationError):
            ActionPlan.model_validate({"steps": [{"do": "wait", "seconds": 0}]})

    def test_empty_plan_rejected(self):
        with pytest.raises(ValidationError):
            ActionPlan.model_validate({"steps": []})


class TestActionPlanLibrary:
    def test_nested_references_validated(self):
        library = ActionPlanLibrary.model_validate(
            {
                "plans": {
                    "inner": {"steps": [{"do": "wait", "seconds": 1}]},
                    "outer": {"steps": [{"do": "run", "plan": "inner"}]},
                }
            }
        )
        assert set(library.plans) == {"inner", "outer"}

    def test_dangling_reference_fails_loudly(self):
        with pytest.raises(ValidationError, match="unknown"):
            ActionPlanLibrary.model_validate(
                {"plans": {"outer": {"steps": [{"do": "run", "plan": "ghost"}]}}}
            )


class TestCorpusDocument:
    """The Undulator library as deployed — an ``ActionPlanLibrary`` document.

    The fixture is the regenerated ``action_library/actions.yaml`` (0.22.0:
    the legacy ``actions:`` dialect and its converter are gone); the golden
    pins one plan's exact shape so a schema change that alters what a
    deployed file means shows up here.
    """

    def test_undulator_library_validates_with_nested_references(self):
        document = yaml.safe_load(
            (FIXTURES / "actions/actions_undulator.yaml").read_text()
        )
        assert "actions" not in document  # new schema only
        library = ActionPlanLibrary.model_validate(document)
        outer = library.plans["experiment_CLOSEOUT"]
        assert all(step.do == "run" for step in outer.steps)
        assert all(step.plan in library.plans for step in outer.steps)

    def test_amp4_dump_hp_matches_golden(self):
        document = yaml.safe_load(
            (FIXTURES / "actions/actions_undulator.yaml").read_text()
        )
        library = ActionPlanLibrary.model_validate(document)
        expected = json.loads((GOLDEN / "amp4_dump_hp_plan.json").read_text())
        assert library.plans["Amp4_DUMP_HP"].model_dump(mode="json") == expected, (
            "The fixture's Amp4_DUMP_HP no longer matches the golden — if the "
            "change is intentional, regenerate with tests/generate_golden.py."
        )

    def test_legacy_dialect_is_not_a_library(self):
        """No converter: the old shape fails validation (unknown key, no plans)."""
        with pytest.raises(ValidationError):
            ActionPlanLibrary.model_validate(
                {"actions": {"x": {"steps": [{"action": "wait", "wait": 1}]}}}
            )
