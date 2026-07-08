"""Model tests for ActionPlan and ActionPlanLibrary."""

import pytest
from pydantic import ValidationError

from geecs_schemas import ActionPlan, ActionPlanLibrary, CheckStep, SetStep


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
