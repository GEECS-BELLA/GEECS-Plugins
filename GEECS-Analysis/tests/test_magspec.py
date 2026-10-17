"""The ``magspec`` measure: the legacy calibration bit for bit; tables and the per-shot field as inputs."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from geecs_data_utils.frames import Frame, ShotMeta
from geecs_data_utils.io import read_table
from geecs_data_utils.tables import Table
from geecs_schemas.analysis import AnalysisRecipe
from pydantic import ValidationError

from geecs_analysis.algorithms import magspec_calibration as calibration
from geecs_analysis.engine.compiled import analyze_compiled
from geecs_analysis.engine.stream import ShotGroup, run_units
from geecs_analysis.measures.magspec import MagspecSpec
from geecs_analysis.pipeline import measure_tables, shot_scalar_columns
from geecs_analysis.recipe import RecipeError, compile_recipe
from geecs_analysis.registry import MeasureSpec, measure

#: The s-file column of the per-shot field in these tests (a test name; the
#: legacy analyzer hardcoded the facility's, which stays in its code).
COLUMN = "Dipole Field"
PIPELINE = {
    "pipeline": ["background", "roi", "filtering"],
    "background": {"method": "constant", "constant_level": 5},
    "roi": {"x_min": 10, "x_max": 138, "y_min": 4, "y_max": 60},
    "filtering": {"median_kernel_size": 3},
}


def write_tsv(path: Path, columns: list[str], rows: list[list]) -> Path:
    lines = ["\t".join(columns), *("\t".join(str(c) for c in row) for row in rows)]
    path.write_text("\n".join(lines) + "\n")
    return path


def synthetic_tables(directory: Path, *, lanex_by_setting: bool = True) -> dict:
    """Three surveys in the real tables' column vocabulary, cameras 1 (front) and 3 (side)."""
    cam_columns = [
        "cam number", "FOV [mm]", "Left edge [mm]", "ROI Y offset", "ROI height",
        "ROI X offset", "ROI width", "Y center pixel", "Y Start", "Y End",
        "X Start", "X End", "rot [deg]", "set", "screen", "name", "sensitivity",
    ]  # fmt: skip
    cam_rows = [
        [1, 100.0, 150.0, 0, 64, 0, 128, 32, 1, 64, 1, 128, 0, 1, "back", "Cam1", 0.5],
        [
            3,
            120.0,
            260.0,
            0,
            64,
            0,
            200,
            32,
            1,
            64,
            37,
            164,
            0,
            2,
            "front",
            "Cam3",
            0.25,
        ],
    ]
    # Front-screen rows listed with descending screen position (the real
    # table's order), so the port's argsort is exercised; then the side rows.
    front = np.linspace(0.170, 0.030, 24)
    side = np.linspace(0.150, 0.260, 20)
    trj_columns = [
        "momentum [MeV/c]",
        "side logic",
        "front screen [m]",
        "side screen [m]",
    ]
    trj_rows = [[8.0 / s, 0, s, s] for s in front] + [[3.0 / s, 1, s, s] for s in side]
    lanex_columns = [
        "FOV slope",
        "FOV offset",
        "sensitivity 2",
        "sensitivity 1",
        "sensitivity 0",
    ]
    lanex_rows = [[0.7, 20.0, 1.5e-5, 1.2e-3, 0.015], [0.9, 25.0, 2.0e-5, 1.0e-3, 0.02]]
    if lanex_by_setting:
        # Setting 2 listed first: the lookup is by number, not by position.
        lanex_columns = ["setting number", *lanex_columns]
        lanex_rows = [[2, *lanex_rows[1]], [1, *lanex_rows[0]]]
    return {
        "camera": write_tsv(directory / "cam.txt", cam_columns, cam_rows),
        "trajectory": write_tsv(directory / "trj.txt", trj_columns, trj_rows),
        "lanex": write_tsv(directory / "lanex.txt", lanex_columns, lanex_rows),
    }


#: The converter's v3 form of ``PIPELINE``.
STEPS = [
    {"step": "background_constant", "level": 5},
    {"step": "roi", "bounds": [[4, 60], [10, 138]]},
    {"step": "median"},
]


def dnn(
    paths,
    *,
    camera=1,
    field=0.3,
    lanex=True,
    energy_range=(1.0, 1500.0),
    steps=None,
    field_column=None,
) -> AnalysisRecipe:
    """The v3 dnn_axis recipe, its tables bound by absolute path."""
    keys = ["camera", "trajectory", *(["lanex"] if lanex else [])]
    measure_ = {
        "kind": "magspec",
        "calibration": {
            "kind": "dnn_axis",
            "camera_number": camera,
            "magnetic_field_t": field,
        },
        **{f"{key}_table": f"{key}_table" for key in keys},
        "energy_range": list(energy_range),
        "num_energy_points": 300,
    }
    if field_column is not None:
        measure_["magnetic_field_column"] = field_column
    return AnalysisRecipe.model_validate(
        {
            "device": "TestCam",
            "input": {"kind": "camera"},
            "inputs": {f"{key}_table": {"table": str(paths[key])} for key in keys},
            "steps": STEPS if steps is None else steps,
            "measure": measure_,
        }
    )


def poly(**measure_) -> AnalysisRecipe:
    """The v3 polynomial recipe; ``measure_`` overrides measure fields."""
    return AnalysisRecipe.model_validate(
        {
            "device": "TestCam",
            "input": {"kind": "camera"},
            "steps": STEPS,
            "measure": {
                "kind": "magspec",
                "calibration": {"kind": "polynomial", "coeffs": [20.0, 0.5, 2e-3]},
                "energy_range": [30, 90],
                "num_energy_points": 200,
                **measure_,
            },
        }
    )


def frame(seed: int) -> np.ndarray:
    """A dispersed beam: two vertical stripes on noise, 64 x 160 uint16."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:160]
    img = 3000 * np.exp(-0.5 * (((x - 70) / 9) ** 2 + ((y - 30) / 7) ** 2))
    img += 1200 * np.exp(-0.5 * (((x - 110) / 5) ** 2 + ((y - 34) / 9) ** 2))
    img += 30 + rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def bound_tables(compiled) -> dict[str, Table]:
    return {request.key: read_table(request.path) for request in compiled.tables}


def assert_same_result(check, name: str, core) -> None:
    """Scalars, the resampled frame, its energy axis and the spectrum, vs ``name``."""
    assert set(core.scalars) == MagspecSpec.scalar_docs.keys()
    stored = check(
        name,
        actual={
            **{f"scalar_{k}": v for k, v in core.scalars.items()},
            "frame": core.frame.data,
            "energy_axis": core.frame.axes[1].values,
        },
    )
    assert (core.frame.axes[1].unit, core.frame.axes[1].label) == ("MeV", "Energy")
    spectrum = core.extras["spectrum"]
    np.testing.assert_allclose(
        spectrum.data, np.sum(stored["frame"], axis=0) / 1000.0, rtol=1e-6
    )
    assert spectrum.axes[0] is core.frame.axes[1]
    assert (spectrum.unit, spectrum.label) == ("pC/MeV", "Charge Density")


# ------------------------------------------------------------ polynomial / array


@pytest.mark.parametrize("seed", range(3))
def test_polynomial_route_equals_the_legacy_analyzer(seed, expected):
    raw = frame(seed)
    core = analyze_compiled(raw, compile_recipe(poly(), allow_file_backgrounds=True))
    assert_same_result(
        expected,
        f"polynomial_{seed}",
        core,
    )
    assert core.frame.unit == "counts/MeV" and core.overlays == ()


def test_array_calibration_is_inline_only(expected):
    values = [20.0 + 0.5 * i for i in range(128)]
    raw = frame(0)
    compiled = compile_recipe(
        poly(calibration={"kind": "array", "values": values}),
        allow_file_backgrounds=True,
    )
    assert_same_result(
        expected,
        "array",
        analyze_compiled(raw, compiled),
    )


# ------------------------------------------------------------ dnn_axis + tables


@pytest.mark.parametrize("camera", [1, 3])
@pytest.mark.parametrize("lanex", [True, False])
@pytest.mark.parametrize("energy_range", [(1.0, 1500.0), (60.0, 75.0)])
def test_dnn_route_equals_the_legacy_analyzer_on_the_same_tables(
    tmp_path, camera, lanex, energy_range, expected, request
):
    doc = dnn(
        synthetic_tables(tmp_path),
        camera=camera,
        lanex=lanex,
        energy_range=energy_range,
    )
    with pytest.raises(RecipeError, match="source host"):
        compile_recipe(doc)
    compiled = compile_recipe(doc, allow_file_backgrounds=True)
    assert [t.key for t in compiled.tables] == (
        ["camera_table", "trajectory_table", "lanex_table"]
        if lanex
        else ["camera_table", "trajectory_table"]
    )
    raw = frame(camera)
    core = analyze_compiled(raw, compiled, inputs=bound_tables(compiled))
    assert_same_result(
        expected,
        f"dnn_{request.node.callspec.id}",
        core,
    )
    assert core.frame.unit == ("fC/MeV" if lanex else "counts/MeV")
    if energy_range[0] < 2:
        # The request is clipped to the calibration's reach, silently.
        axis = core.frame.axes[1].values
        assert axis[0] > 1.0 and axis[-1] < 1500.0
        assert core.notes == ()


def test_lanex_row_falls_back_to_position_without_a_setting_column(tmp_path, expected):
    doc = dnn(synthetic_tables(tmp_path, lanex_by_setting=False), camera=3)
    compiled = compile_recipe(doc, allow_file_backgrounds=True)
    raw = frame(3)
    assert_same_result(
        expected,
        "lanex_by_position",
        analyze_compiled(raw, compiled, inputs=bound_tables(compiled)),
    )


@pytest.mark.parametrize("camera", [1, 3])
def test_cells_the_legacy_reader_never_converted_may_be_blank(
    tmp_path, camera, expected
):
    """Only the camera's screen rows are converted, as the legacy reader did.

    A survey may leave the other screen's rows of the screen and momentum
    columns blank, and ``side logic`` after the split row; the legacy
    reader never read them.
    """
    paths = synthetic_tables(tmp_path)
    table = read_table(paths["trajectory"])
    logic = table.columns.index("side logic")
    split = next(i for i, row in enumerate(table.rows) if float(row[logic]) == 1.0)
    front = camera in (1, 2)
    own = set(range(split) if front else range(split, len(table)))
    screen = table.columns.index("front screen [m]" if front else "side screen [m]")
    momentum = table.columns.index("momentum [MeV/c]")
    rows = [list(row) for row in table.rows]
    for index, row in enumerate(rows):
        if index not in own:
            row[screen] = row[momentum] = ""
        if index > split:
            row[logic] = ""
    write_tsv(paths["trajectory"], list(table.columns), rows)
    doc = dnn(paths, camera=camera)
    compiled = compile_recipe(doc, allow_file_backgrounds=True)
    raw = frame(camera)
    assert_same_result(
        expected,
        f"blank_cells_{camera}",
        analyze_compiled(raw, compiled, inputs=bound_tables(compiled)),
    )


def test_the_per_shot_field_equals_the_legacy_override_and_never_leaks(
    tmp_path, expected
):
    doc = dnn(synthetic_tables(tmp_path), field=0.3, field_column=COLUMN)
    compiled = compile_recipe(doc, allow_file_backgrounds=True)
    assert shot_scalar_columns(compiled.analysis.measure) == (COLUMN,)
    tables = bound_tables(compiled)
    raw = frame(5)
    with_field = analyze_compiled(
        raw,
        compiled,
        shot=ShotMeta("TestCam", 1, scalars={COLUMN: 0.25}),
        inputs=tables,
    )
    assert_same_result(
        expected,
        "per_shot_field",
        with_field,
    )
    # A second shot without the column: the recipe's field, as a fresh
    # legacy analyzer would give — not the previous shot's reading.
    without = analyze_compiled(
        raw, compiled, shot=ShotMeta("TestCam", 2), inputs=tables
    )
    assert_same_result(
        expected,
        "recipe_field",
        without,
    )
    assert without.scalars["peak_energy_MeV"] != with_field.scalars["peak_energy_MeV"]


def test_the_field_rides_through_the_runner_on_shot_metadata(tmp_path):
    doc = dnn(synthetic_tables(tmp_path), field_column=COLUMN)
    compiled = compile_recipe(doc, allow_file_backgrounds=True)
    tables = bound_tables(compiled)
    raw = frame(6)
    shot = ShotMeta("TestCam", 1, 10.0, {COLUMN: 0.22})
    (unit,) = run_units(
        compiled,
        [ShotGroup(1, (1,))],
        {1: raw}.__getitem__,
        inputs=tables,
        shot_metadata={1: shot},
    )
    direct = analyze_compiled(raw, compiled, shot=shot, inputs=tables)
    assert unit.error is None and dict(unit.measurement.scalars) == dict(direct.scalars)
    assert unit.measurement.frame.shot == shot


# ------------------------------------------------------------ the real tables


def real_tables() -> dict[str, Path]:
    root = Path(__file__).parent / "data" / "HTT-MagSpce-Calibrations"
    return {
        "camera": root / "260224DnnVarianCam.txt",
        "trajectory": root / "170925DnnVarianTrj.txt",
        "lanex": root / "250101lanexCalib.txt",
    }


@pytest.mark.parametrize("camera", [1, 2, 3, 4])
def test_the_real_calibration_equals_the_legacy_axis_and_charge_factor(
    camera, expected
):
    paths = real_tables()
    tables = {key: read_table(path) for key, path in paths.items()}
    row = calibration.camera_row(tables["camera"], camera)
    geometry = calibration.camera_geometry(row)
    axis = calibration.dnn_energy_axis(
        geometry,
        calibration.trajectory(tables["trajectory"], camera),
        0.2008,
        geometry.width,
    )
    # The HTT documents' request, clipped to the camera's own reach.
    image = np.random.default_rng(camera).normal(100, 10, (4, geometry.width))
    resampled, grid = calibration.resample_columns(image, axis, 1000, 1.2, 1501.3)
    assert (grid[0], grid[-1]) == (axis.min(), axis.max())

    expected(
        f"real_calibration_{camera}",
        actual={
            "axis": axis,
            "charge_factor": calibration.charge_factor(row, tables["lanex"]),
            "resampled": resampled,
            "grid": grid,
        },
    )


# ------------------------------------------------------------ refusals


@pytest.mark.parametrize(
    "patch, needle",
    [
        ({"camera_table": None}, "camera_table and trajectory_table"),
        ({"calibration": {"kind": "polynomial", "coeffs": [1.0]}}, "binds no tables"),
        ({"energy_range": [90.0, 30.0]}, "max must exceed min"),
    ],
)
def test_spec_refusals(patch, needle):
    base = {
        "calibration": {"kind": "dnn_axis", "camera_number": 1},
        "camera_table": "cam",
        "trajectory_table": "trj",
        "energy_range": [30.0, 90.0],
    }
    with pytest.raises(ValidationError, match=needle):
        MagspecSpec.model_validate({**base, **patch})


def test_registration_requires_optional_string_fields():
    class Bogus(MeasureSpec):
        kind: str = "bogus"

    for keyword in ({"tables": ("kind",)}, {"shot_scalars": ("missing",)}):
        with pytest.raises(ValueError, match="optional string fields"):
            measure(Bogus, ndim={2}, **keyword)


def recipe(**patch) -> AnalysisRecipe:
    document = {
        "device": "TestCam",
        "input": {"kind": "camera"},
        "inputs": {
            "cam": {"table": "calibrations/HTT/cam.txt"},
            "trj": {"table": "trj.txt"},
        },
        "measure": {
            "kind": "magspec",
            "calibration": {"kind": "dnn_axis", "camera_number": 1},
            "camera_table": "cam",
            "trajectory_table": "trj",
            "energy_range": [30, 90],
        },
    }
    return AnalysisRecipe.model_validate({**document, **patch})


def test_a_v3_recipe_binds_its_tables_by_key():
    compiled = compile_recipe(recipe(), allow_file_backgrounds=True)
    assert [(t.key, t.path) for t in compiled.tables] == [
        ("cam", "calibrations/HTT/cam.txt"),
        ("trj", "trj.txt"),
    ]
    assert compiled.file_backgrounds == () and compiled.scan_backgrounds == ()
    assert measure_tables(compiled.analysis.measure) == {
        "camera_table": "cam",
        "trajectory_table": "trj",
    }
    assert pickle.loads(pickle.dumps(compiled)) == compiled
    with pytest.raises(RecipeError, match="source host"):
        compile_recipe(recipe())


@pytest.mark.parametrize(
    "patch, needle",
    [
        (
            {"inputs": {"cam": {"path": "cam.txt"}, "trj": {"table": "trj.txt"}}},
            "declared as a frame but bound as a table",
        ),
        (
            {
                "inputs": {
                    "cam": {"table": "cam.txt"},
                    "trj": {"table": "t"},
                    "dark": {"table": "d.txt"},
                },
                "steps": [{"step": "background_frame", "source": "dark"}],
            },
            "declared as a table but bound as a frame",
        ),
        ({"inputs": {"cam": {"table": "cam.txt"}}}, "does not declare"),
        (
            {
                "inputs": {"cam": {"table": "cam.txt"}},
                "measure": {
                    "kind": "magspec",
                    "calibration": {"kind": "dnn_axis", "camera_number": 1},
                    "camera_table": "cam",
                    "trajectory_table": "cam",
                    "energy_range": [30, 90],
                },
            },
            "two table fields",
        ),
        (
            {
                "inputs": {
                    "cam": {"table": "c"},
                    "trj": {"table": "t"},
                    "spare": {"table": "s"},
                }
            },
            "no step or measure uses",
        ),
    ],
)
def test_table_binding_errors_are_recipe_errors(patch, needle):
    with pytest.raises(RecipeError, match=needle):
        compile_recipe(recipe(**patch), allow_file_backgrounds=True)


def test_a_per_shot_column_needs_per_shot_units(tmp_path):
    bound = recipe(
        measure={
            "kind": "magspec",
            "calibration": {"kind": "dnn_axis", "camera_number": 1},
            "camera_table": "cam",
            "trajectory_table": "trj",
            "magnetic_field_column": COLUMN,
            "energy_range": [30, 90],
        },
        scan={"average_frames_first": True},
    )
    # "measure …" so the editor places it on the measure section.
    with pytest.raises(RecipeError, match="^measure .*average_frames_first"):
        compile_recipe(bound, allow_file_backgrounds=True)


def test_execution_refuses_a_missing_or_mistyped_table(tmp_path):
    compiled = compile_recipe(
        dnn(synthetic_tables(tmp_path)), allow_file_backgrounds=True
    )
    tables = bound_tables(compiled)
    with pytest.raises(ValueError, match="Missing table input: lanex_table"):
        analyze_compiled(
            frame(0),
            compiled,
            inputs={k: v for k, v in tables.items() if k != "lanex_table"},
        )
    with pytest.raises(TypeError, match="Table input 'camera_table' must be a Table"):
        analyze_compiled(
            frame(0),
            compiled,
            inputs={**tables, "camera_table": Frame.from_array(np.ones((2, 2)))},
        )


@pytest.mark.parametrize(
    "change, needle",
    [
        (
            {
                "steps": [
                    *STEPS[:1],
                    {"step": "roi", "bounds": [[4, 60], [10, 100]]},
                    *STEPS[2:],
                ]
            },
            "doesn't match image width",
        ),
        ({"camera": 2}, "Camera number 2 not found"),
        ({"field": 0.3, "no_split": True}, "side logic"),
        ({"far": True}, "non-finite momentum"),
    ],
)
def test_calibration_refusals_match_the_legacy_ones(tmp_path, change, needle):
    paths = synthetic_tables(tmp_path)
    if change.pop("no_split", False):
        text = paths["trajectory"].read_text().replace("\t1\t", "\t0\t")
        paths["trajectory"].write_text(text)
    if change.pop("far", False):
        text = paths["camera"].read_text().replace("\t150.0\t", "\t900.0\t")
        paths["camera"].write_text(text)
    doc = dnn(paths, **change)
    compiled = compile_recipe(doc, allow_file_backgrounds=True)
    with pytest.raises(ValueError, match=needle):
        analyze_compiled(frame(0), compiled, inputs=bound_tables(compiled))


# ------------------------------------------------------------ conversion


# ------------------------------------------------------- host-facing declarations


def test_sibling_outputs_are_declared_for_the_host():
    from geecs_analysis.registry import SiblingOutput, measure_definition

    definition = measure_definition(MagspecSpec(**polynomial_measure()))
    assert [
        (o.suffix, o.source, o.format, o.scale) for o in definition.sibling_outputs
    ] == [
        ("-interp", "frame", "png16", 1000.0),
        ("-interpSpec", "spectrum", "tsv", 1.0),
    ]
    assert definition.per_shot_axes
    for bad in ("", "..", "a/b", "a\\b"):
        with pytest.raises(ValueError, match="one folder name"):
            SiblingOutput(bad, "frame", "png16")
    with pytest.raises(ValueError, match="format"):
        SiblingOutput("-x", "frame", "jpeg")

    class Twice(MeasureSpec):
        kind: str = "twice"

    with pytest.raises(ValueError, match="distinct suffixes"):
        measure(
            Twice,
            ndim={2},
            sibling_outputs=(SiblingOutput("-a", "frame", "png16"),) * 2,
        )


def polynomial_measure() -> dict:
    return {
        "calibration": {"kind": "polynomial", "coeffs": [20.0, 0.5, 2e-3]},
        "energy_range": [30, 90],
    }


@pytest.mark.parametrize("mode", ["noscan", "bin"])
def test_an_average_folds_moving_energy_grids_index_wise(tmp_path, mode, expected):
    """A per-shot field moves the grid; the average is the legacy nanmean of frames and axes."""

    from geecs_analysis.engine.average import RunningAverage

    doc = dnn(synthetic_tables(tmp_path), field_column=COLUMN)
    compiled = compile_recipe(doc, allow_file_backgrounds=True)
    tables = bound_tables(compiled)
    running = RunningAverage(compiled, mode=mode)
    grids = []
    fields = (0.28, 0.30, 0.33)
    for n, field in enumerate(fields, start=1):
        shot = ShotMeta("TestCam", n, scalars={COLUMN: field})
        core = analyze_compiled(frame(n), compiled, shot=shot, inputs=tables)
        running.add(core)
        grids.append(core.frame.axes[1].values)
    assert not np.array_equal(grids[0], grids[2]), "the field moves the grid"
    average = running.result()

    expected(
        f"average_{mode}",
        actual={
            "frame": average.frame.data,
            "energy_axis": average.frame.axes[1].values,
        },
    )


def test_a_steady_grid_keeps_the_first_axes_and_other_measures_still_refuse():
    from geecs_data_utils.frames import Axis

    from geecs_analysis.engine.average import RunningAverage

    compiled = compile_recipe(poly(), allow_file_backgrounds=True)
    first = analyze_compiled(frame(1), compiled)
    running = RunningAverage(compiled, mode="bin")
    for n in (1, 2, 3):
        running.add(analyze_compiled(frame(n), compiled))
    for kept, axis in zip(running.result().frame.axes, first.frame.axes, strict=True):
        assert np.array_equal(kept.values, axis.values)
    beam = compile_recipe(
        AnalysisRecipe.model_validate(
            {
                "device": "TestCam",
                "input": {"kind": "camera"},
                "measure": {"kind": "beam"},
            }
        )
    )
    result = analyze_compiled(frame(1), beam)
    y, x = result.frame.axes
    moved = result.frame.data, (y, Axis(x.values + 1.0, unit=x.unit, label=x.label))
    running = RunningAverage(beam, mode="bin")
    running.add(result)
    with pytest.raises(ValueError, match="matching axes"):
        running.add(
            result.__class__(
                scalars={}, frame=Frame.from_array(moved[0], axes=moved[1])
            )
        )
