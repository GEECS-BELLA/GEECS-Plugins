"""Model tests for ScanRequest and its sub-models."""

import pytest
from pydantic import ValidationError

from geecs_schemas import (
    AcquisitionMode,
    OptimizationSpec,
    PositionList,
    PositionRange,
    ScanRequest,
    ScanRequestMode,
)


def make_step_request(**overrides):
    base = {
        "mode": "step",
        "axes": [
            {
                "variable": "jet_z",
                "positions": {"start": 4.0, "end": 6.0, "step": 0.5},
            }
        ],
        "shots_per_step": 10,
        "save_set": "undulator_baseline",
        "trigger_profile": "htu_shot_control",
    }
    base.update(overrides)
    return ScanRequest.model_validate(base)


def make_optimization_block():
    return {
        "variables": {"U_Hexapod:ypos": [17, 19]},
        "objectives": {"f": "MINIMIZE"},
        "evaluator": {
            "module": "geecs_scanner.optimization.evaluators.x",
            "class": "MaxCountsEvaluator",
            "kwargs": {"analyzers": ["UC_TC_Output"]},
        },
        "generator": {"name": "bayes_default"},
    }


class TestScanRequest:
    def test_step_scan_round_trip(self):
        request = make_step_request()
        assert request.mode is ScanRequestMode.STEP
        assert request.acquisition is AcquisitionMode.STRICT  # engine default
        again = ScanRequest.model_validate(request.model_dump(mode="json"))
        assert again == request

    def test_vision_yaml_sketch_validates(self):
        # The vision doc §4.1 sketch, updated for the axes-list shape (the
        # single variable/positions pair became a one-entry axes list when
        # grid scans were added — maintainer scope addition, 2026-07-07).
        request = ScanRequest.model_validate(
            {
                "schema_version": 1,
                "mode": "step",
                "axes": [
                    {
                        "variable": "jet_z",
                        "positions": {"start": 4.0, "end": 6.0, "step": 0.5},
                    }
                ],
                "shots_per_step": 10,
                "acquisition": "free_run",
                "save_set": "undulator_baseline",
                "trigger_profile": "htu_laser_off",
                "actions": {"setup": ["pre_scan_ebeam"], "closeout": []},
                "description": "jet z scan with probe",
            }
        )
        assert request.actions.setup == ["pre_scan_ebeam"]

    def test_two_axis_grid_round_trip(self):
        request = make_step_request(
            axes=[
                {
                    "variable": "jet_z",
                    "positions": {"start": 4.0, "end": 6.0, "step": 0.5},
                },
                {
                    "variable": "gas_pressure",
                    "positions": {"values": [1.5, 2.0, 2.5]},
                },
            ]
        )
        assert [axis.variable for axis in request.axes] == [
            "jet_z",
            "gas_pressure",
        ]
        again = ScanRequest.model_validate(request.model_dump(mode="json"))
        assert again == request

    def test_grid_shape_and_step_count(self):
        request = make_step_request(
            axes=[
                {
                    "variable": "jet_z",
                    "positions": {"start": 4.0, "end": 6.0, "step": 0.5},
                },  # 5 positions — outermost (slowest) loop
                {
                    "variable": "gas_pressure",
                    "positions": {"values": [1.5, 2.0, 2.5]},
                },  # 3 positions — innermost (fastest) loop
            ]
        )
        assert request.grid_shape() == (5, 3)
        assert request.n_steps() == 15

    def test_noscan_is_one_bin(self):
        request = ScanRequest.model_validate({"mode": "noscan"})
        assert request.grid_shape() == ()
        assert request.n_steps() == 1

    def test_duplicate_axis_variable_rejected(self):
        with pytest.raises(ValidationError, match="more than once"):
            make_step_request(
                axes=[
                    {"variable": "jet_z", "positions": {"values": [1.0]}},
                    {"variable": "jet_z", "positions": {"values": [2.0]}},
                ]
            )

    def test_per_step_actions_slot(self):
        # The architecture's acceptance test: actions between scan steps are
        # a slot on the existing model, not a new dialect.
        request = make_step_request(actions={"per_step": ["insert_plunger_check"]})
        assert request.actions.per_step == ["insert_plunger_check"]

    def test_unknown_field_fails_loudly(self):
        with pytest.raises(ValidationError, match="shots_per_stpe"):
            make_step_request(shots_per_stpe=10)

    def test_step_requires_axes(self):
        with pytest.raises(ValidationError, match="axes"):
            ScanRequest.model_validate({"mode": "step"})
        with pytest.raises(ValidationError, match="axes"):
            ScanRequest.model_validate({"mode": "step", "axes": []})

    def test_noscan_forbids_axes(self):
        with pytest.raises(ValidationError, match="only applies to 'step'"):
            ScanRequest.model_validate(
                {
                    "mode": "noscan",
                    "axes": [{"variable": "jet_z", "positions": {"values": [1.0]}}],
                }
            )

    def test_optimize_forbids_axes(self):
        with pytest.raises(ValidationError, match="optimization' block"):
            ScanRequest.model_validate(
                {
                    "mode": "optimize",
                    "optimization": make_optimization_block(),
                    "axes": [{"variable": "jet_z", "positions": {"values": [1.0]}}],
                }
            )

    def test_noscan_minimal(self):
        request = ScanRequest.model_validate({"mode": "noscan", "shots_per_step": 100})
        assert request.axes == []
        assert not request.background

    def test_optimize_requires_block(self):
        with pytest.raises(ValidationError, match="optimization"):
            ScanRequest.model_validate({"mode": "optimize"})

    def test_optimize_round_trip(self):
        request = ScanRequest.model_validate(
            {"mode": "optimize", "optimization": make_optimization_block()}
        )
        assert request.optimization.generator.name == "bayes_default"

    def test_optimization_block_forbidden_elsewhere(self):
        with pytest.raises(ValidationError, match="only allowed"):
            ScanRequest.model_validate(
                {
                    "mode": "noscan",
                    "optimization": make_optimization_block(),
                }
            )

    def test_trigger_variant_needs_profile(self):
        with pytest.raises(ValidationError, match="trigger_variant"):
            ScanRequest.model_validate(
                {"mode": "noscan", "trigger_variant": "laser_off"}
            )

    def test_explicit_position_list(self):
        request = make_step_request(
            axes=[{"variable": "jet_z", "positions": {"values": [0.0, 0.5, 2.0]}}]
        )
        positions = request.axes[0].positions
        assert isinstance(positions, PositionList)
        assert positions.to_values() == [0.0, 0.5, 2.0]


class TestPositions:
    def test_range_expansion(self):
        values = PositionRange(start=4.0, end=6.0, step=0.5).to_values()
        assert values == [4.0, 4.5, 5.0, 5.5, 6.0]

    def test_descending_range_ignores_step_sign(self):
        # Legacy presets store descending sweeps with a positive step size.
        values = PositionRange(start=-18.0, end=-26.0, step=0.5).to_values()
        assert values[0] == -18.0 and values[-1] == -26.0
        assert len(values) == 17

    def test_zero_step_rejected(self):
        with pytest.raises(ValidationError, match="step"):
            PositionRange(start=0, end=1, step=0)

    def test_empty_value_list_rejected(self):
        with pytest.raises(ValidationError):
            PositionList(values=[])


class TestOptimizationSpec:
    def test_direction_normalized(self):
        spec = OptimizationSpec.model_validate(
            {**make_optimization_block(), "objectives": {"f": "minimize"}}
        )
        assert spec.objectives == {"f": "MINIMIZE"}

    def test_bad_direction_rejected(self):
        with pytest.raises(ValidationError, match="MINIMIZE"):
            OptimizationSpec.model_validate(
                {**make_optimization_block(), "objectives": {"f": "downhill"}}
            )

    def test_bax_shape_no_objectives(self):
        spec = OptimizationSpec.model_validate(
            {
                **make_optimization_block(),
                "objectives": {},
                "observables": ["x_CoM"],
                "generator": {
                    "name": "multipoint_bax_alignment_l2",
                    "options": {"control_names": ["U_S1H:Current"]},
                },
            }
        )
        assert spec.observables == ["x_CoM"]

    def test_bad_constraint_bound_rejected(self):
        with pytest.raises(ValidationError, match="LESS_THAN"):
            OptimizationSpec.model_validate(
                {
                    **make_optimization_block(),
                    "constraints": {"charge": ["ABOVE", 5.0]},
                }
            )

    def test_evaluator_class_alias(self):
        spec = OptimizationSpec.model_validate(make_optimization_block())
        assert spec.evaluator.class_name == "MaxCountsEvaluator"
        dumped = spec.model_dump(by_alias=True)
        assert dumped["evaluator"]["class"] == "MaxCountsEvaluator"
