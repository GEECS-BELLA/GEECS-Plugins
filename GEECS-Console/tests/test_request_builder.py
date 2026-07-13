"""form state -> exact ScanRequest, validated against the real schema."""

import pytest
from pydantic import ValidationError

from geecs_console.request_builder import (
    MAXIMUM_SCAN_SIZE,
    ConsoleFormError,
    ConsoleFormState,
    ConsoleMode,
    FormAxis,
    build_scan_request,
    estimate_total_shots,
    form_state_from_request,
)
from geecs_schemas import (
    AcquisitionMode,
    PositionList,
    PositionRange,
    ScanRequest,
    ScanRequestMode,
)


def roundtrip(request: ScanRequest) -> ScanRequest:
    """Re-validate through the schema's own loader (serialization parity)."""
    return ScanRequest.model_validate(request.model_dump(mode="json"))


def _optimization_spec():
    """A small valid OptimizationSpec for the optimize-mode tests."""
    from geecs_schemas import EvaluatorSpec, GeneratorSpec, OptimizationSpec

    return OptimizationSpec(
        variables={"jet_x": (0.0, 1.0)},
        objectives={"counts": "MAXIMIZE"},
        evaluator=EvaluatorSpec(module="m", class_name="C"),
        generator=GeneratorSpec(name="random"),
    )


class TestNoscan:
    def test_noscan_maps_to_schema_noscan(self):
        form = ConsoleFormState(
            mode=ConsoleMode.NOSCAN, shots_per_step=100, save_sets=["Amp4In"]
        )
        request = build_scan_request(form)
        assert request.mode is ScanRequestMode.NOSCAN
        assert request.axes == []
        assert request.shots_per_step == 100
        assert request.save_sets == ["Amp4In"]
        assert request.background is False
        assert request.n_steps() == 1
        roundtrip(request)

    def test_acquisition_defaults_to_free_run(self):
        request = build_scan_request(ConsoleFormState(mode=ConsoleMode.NOSCAN))
        assert request.acquisition is AcquisitionMode.FREE_RUN

    def test_strict_acquisition_passes_through(self):
        form = ConsoleFormState(
            mode=ConsoleMode.NOSCAN, acquisition=AcquisitionMode.STRICT
        )
        assert build_scan_request(form).acquisition is AcquisitionMode.STRICT


class TestBackground:
    def test_background_is_noscan_with_flag(self):
        form = ConsoleFormState(mode=ConsoleMode.BACKGROUND, shots_per_step=50)
        request = build_scan_request(form)
        assert request.mode is ScanRequestMode.NOSCAN
        assert request.background is True
        assert request.axes == []
        roundtrip(request)

    def test_explicit_background_flag_survives_other_modes(self):
        form = ConsoleFormState(
            mode=ConsoleMode.ONE_D,
            axes=[FormAxis(variable="jet_x", start=0.0, stop=1.0, step=0.5)],
            background=True,
        )
        assert build_scan_request(form).background is True


class TestOneD:
    def default_form(self, **overrides) -> ConsoleFormState:
        kwargs = dict(
            mode=ConsoleMode.ONE_D,
            axes=[FormAxis(variable="jet_x", start=0.0, stop=1.0, step=0.25)],
            shots_per_step=10,
            save_sets=["Amp4In"],
        )
        kwargs.update(overrides)
        return ConsoleFormState(**kwargs)

    def test_start_stop_step_becomes_position_range(self):
        request = build_scan_request(self.default_form())
        assert request.mode is ScanRequestMode.STEP
        assert len(request.axes) == 1
        axis = request.axes[0]
        assert axis.variable == "jet_x"
        assert isinstance(axis.positions, PositionRange)
        assert axis.positions.to_values() == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert request.n_steps() == 5
        roundtrip(request)

    def test_explicit_values_become_position_list(self):
        form = self.default_form(
            axes=[FormAxis(variable="jet_x", values=[0.0, 0.5, 2.0, 8.0])]
        )
        axis = build_scan_request(form).axes[0]
        assert isinstance(axis.positions, PositionList)
        assert axis.positions.values == [0.0, 0.5, 2.0, 8.0]

    def test_shots_per_step_and_description_land(self):
        form = self.default_form(shots_per_step=7, description="align the jet")
        request = build_scan_request(form)
        assert request.shots_per_step == 7
        assert request.description == "align the jet"

    def test_save_sets_list_passthrough(self):
        form = self.default_form(save_sets=["Amp4In", "EBeamDiags"])
        assert build_scan_request(form).save_sets == ["Amp4In", "EBeamDiags"]

    def test_trigger_profile_and_variant(self):
        form = self.default_form(
            trigger_profile="HTU-Standard", trigger_variant="laser_off"
        )
        request = build_scan_request(form)
        assert request.trigger_profile == "HTU-Standard"
        assert request.trigger_variant == "laser_off"
        roundtrip(request)

    def test_variant_without_profile_rejected_by_schema(self):
        form = self.default_form(trigger_variant="laser_off")
        with pytest.raises(ValidationError, match="trigger_profile"):
            build_scan_request(form)

    def test_zero_step_raises_form_error(self):
        with pytest.raises(ConsoleFormError, match="step"):
            build_scan_request(
                self.default_form(
                    axes=[FormAxis(variable="jet_x", start=0.0, stop=1.0, step=0.0)]
                )
            )

    def test_one_d_requires_exactly_one_axis(self):
        with pytest.raises(ConsoleFormError, match="exactly one axis"):
            build_scan_request(self.default_form(axes=[]))


class TestGrid:
    def test_two_axes_form_an_outer_product(self):
        form = ConsoleFormState(
            mode=ConsoleMode.GRID,
            axes=[
                FormAxis(variable="jet_x", start=0.0, stop=1.0, step=0.5),
                FormAxis(variable="jet_z", start=-1.0, stop=1.0, step=1.0),
            ],
            shots_per_step=3,
        )
        request = build_scan_request(form)
        assert request.mode is ScanRequestMode.STEP
        assert [axis.variable for axis in request.axes] == ["jet_x", "jet_z"]
        assert request.grid_shape() == (3, 3)
        assert request.n_steps() == 9
        assert estimate_total_shots(form) == 27
        roundtrip(request)

    def test_grid_requires_two_axes(self):
        form = ConsoleFormState(
            mode=ConsoleMode.GRID,
            axes=[FormAxis(variable="jet_x", start=0.0, stop=1.0, step=0.5)],
        )
        with pytest.raises(ConsoleFormError, match="two axes"):
            build_scan_request(form)

    def test_duplicate_axis_variable_rejected_by_schema(self):
        form = ConsoleFormState(
            mode=ConsoleMode.GRID,
            axes=[
                FormAxis(variable="jet_x", start=0.0, stop=1.0, step=0.5),
                FormAxis(variable="jet_x", start=0.0, stop=1.0, step=0.5),
            ],
        )
        with pytest.raises(ValidationError, match="more than once"):
            build_scan_request(form)


class TestGuardsAndModes:
    def test_maximum_scan_size_guard_raises(self):
        form = ConsoleFormState(
            mode=ConsoleMode.ONE_D,
            axes=[FormAxis(variable="jet_x", start=0.0, stop=2_000_000.0, step=1.0)],
            shots_per_step=1,
        )
        assert estimate_total_shots(form) == 2_000_001 > MAXIMUM_SCAN_SIZE
        with pytest.raises(ConsoleFormError, match="runaway"):
            build_scan_request(form)

    def test_guard_boundary_is_inclusive(self):
        form = ConsoleFormState(
            mode=ConsoleMode.ONE_D,
            axes=[FormAxis(variable="jet_x", start=1.0, stop=1_000_000.0, step=1.0)],
            shots_per_step=1,
        )
        assert estimate_total_shots(form) == MAXIMUM_SCAN_SIZE
        build_scan_request(form)  # exactly at the limit is allowed

    def test_optimization_without_spec_refused(self):
        with pytest.raises(ConsoleFormError, match="optimization config"):
            build_scan_request(ConsoleFormState(mode=ConsoleMode.OPTIMIZATION))

    def test_optimization_with_spec_builds_optimize_request(self):
        request = build_scan_request(
            ConsoleFormState(
                mode=ConsoleMode.OPTIMIZATION,
                shots_per_step=5,
                save_sets=["Amp4In"],
                optimization=_optimization_spec(),
            )
        )
        assert request.mode is ScanRequestMode.OPTIMIZE
        assert request.axes == []
        assert request.optimization == _optimization_spec()
        assert request.shots_per_step == 5
        roundtrip(request)

    def test_stray_spec_on_non_optimize_mode_rejected_by_schema(self):
        form = ConsoleFormState(
            mode=ConsoleMode.NOSCAN, optimization=_optimization_spec()
        )
        with pytest.raises(ValidationError, match="optimize"):
            build_scan_request(form)

    def test_axis_needs_one_positions_shape(self):
        with pytest.raises(ValidationError, match="start, stop, and step"):
            FormAxis(variable="jet_x", start=0.0)
        with pytest.raises(ValidationError, match="not both"):
            FormAxis(variable="jet_x", start=0.0, stop=1.0, step=0.5, values=[1.0])


class TestFormStateFromRequest:
    """The Apply inverse: request -> form state, round-trip exact."""

    def round_trip(self, form: ConsoleFormState) -> ConsoleFormState:
        return form_state_from_request(build_scan_request(form))

    def test_noscan_round_trips_exactly(self):
        form = ConsoleFormState(
            mode=ConsoleMode.NOSCAN,
            shots_per_step=100,
            save_sets=["Amp4In"],
            description="stats run",
        )
        assert self.round_trip(form) == form

    def test_background_round_trips_to_background_mode(self):
        form = ConsoleFormState(mode=ConsoleMode.BACKGROUND, shots_per_step=50)
        assert self.round_trip(form) == form

    def test_one_d_round_trips_exactly(self):
        form = ConsoleFormState(
            mode=ConsoleMode.ONE_D,
            axes=[FormAxis(variable="jet_x", start=0.0, stop=1.0, step=0.25)],
            shots_per_step=10,
            save_sets=["Amp4In", "EBeamDiags"],
            trigger_profile="HTU-Standard",
            trigger_variant="laser_off",
            acquisition=AcquisitionMode.STRICT,
            description="align the jet",
        )
        assert self.round_trip(form) == form

    def test_grid_round_trips_exactly(self):
        form = ConsoleFormState(
            mode=ConsoleMode.GRID,
            axes=[
                FormAxis(variable="jet_x", start=0.0, stop=1.0, step=0.5),
                FormAxis(variable="jet_z", start=-1.0, stop=1.0, step=1.0),
            ],
            shots_per_step=3,
        )
        assert self.round_trip(form) == form

    def test_position_list_round_trips_exactly(self):
        form = ConsoleFormState(
            mode=ConsoleMode.ONE_D,
            axes=[FormAxis(variable="jet_x", values=[0.0, 0.5, 2.0, 8.0])],
        )
        assert self.round_trip(form) == form

    def test_step_scan_keeps_the_background_flag(self):
        form = ConsoleFormState(
            mode=ConsoleMode.ONE_D,
            axes=[FormAxis(variable="jet_x", start=0.0, stop=1.0, step=0.5)],
            background=True,
        )
        assert self.round_trip(form) == form

    def test_yaml_round_trip_survives_serialization(self):
        """Preset parity: form -> request -> YAML dict -> request -> form."""
        form = ConsoleFormState(
            mode=ConsoleMode.ONE_D,
            axes=[FormAxis(variable="jet_x", start=0.0, stop=1.0, step=0.25)],
            shots_per_step=10,
            save_sets=["Amp4In"],
            trigger_profile="HTU-Standard",
        )
        request = build_scan_request(form)
        reloaded = ScanRequest.model_validate(request.model_dump(mode="json"))
        assert form_state_from_request(reloaded) == form

    def test_optimize_round_trips_exactly(self):
        form = ConsoleFormState(
            mode=ConsoleMode.OPTIMIZATION,
            shots_per_step=5,
            save_sets=["Amp4In"],
            description="tune the jet",
            optimization=_optimization_spec(),
        )
        assert self.round_trip(form) == form

    def test_optimize_round_trip_survives_serialization(self):
        """Preset parity: optimize form -> request -> YAML dict -> form."""
        form = ConsoleFormState(
            mode=ConsoleMode.OPTIMIZATION, optimization=_optimization_spec()
        )
        reloaded = roundtrip(build_scan_request(form))
        assert form_state_from_request(reloaded) == form

    def test_action_bindings_raise_console_form_error(self):
        from geecs_schemas import ActionBindings

        request = ScanRequest(
            mode=ScanRequestMode.NOSCAN,
            actions=ActionBindings(setup=["warmup"]),
        )
        with pytest.raises(ConsoleFormError, match="action bindings"):
            form_state_from_request(request)
