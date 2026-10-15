"""The ``magspec`` measure: the legacy calibration bit for bit; tables and the per-shot field as inputs."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pytest
from geecs_data_utils.frames import Frame, ShotMeta
from geecs_data_utils.io import read_table
from geecs_data_utils.tables import Table
from geecs_schemas.analysis import AnalysisDiagnostic, AnalysisRecipe
from geecs_schemas.analysis.canonical import canonical_document
from pydantic import ValidationError

from geecs_analysis.algorithms import magspec_calibration as calibration
from geecs_analysis.compat.convert import to_v3
from geecs_analysis.compat.v2 import UnsupportedRecipe, analyze_v2, compile_v2
from geecs_analysis.compat.v2_run import ShotGroup, run_units
from geecs_analysis.measures.magspec import MagspecSpec
from geecs_analysis.pipeline import measure_tables, shot_scalar_columns
from geecs_analysis.recipe import RecipeError, compile_recipe
from geecs_analysis.registry import MeasureSpec, measure

#: The s-file column of the per-shot field in these tests (a test name; the
#: legacy analyzer hardcoded the facility's, which stays in its code).
COLUMN = "Dipole Field"
LEGACY_COLUMN = "HTT-MagTeslameter-DTM141 Field"
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


def diagnostic(
    paths,
    *,
    camera=1,
    field=0.3,
    lanex=True,
    energy_range=(1.0, 1500.0),
    image=None,
    scan=None,
):
    cal = {
        "kind": "dnn_axis",
        "camera_calibration_file": str(paths["camera"]),
        "trajectory_calibration_file": str(paths["trajectory"]),
        "camera_number": camera,
        "magnetic_field_t": field,
    }
    if lanex:
        cal["lanex_calibration_file"] = str(paths["lanex"])
    return AnalysisDiagnostic.model_validate(
        {
            "name": "TestCam",
            "analyzer": {
                "kind": "magspec",
                "calibration": cal,
                "energy_range": list(energy_range),
                "num_energy_points": 300,
            },
            "image": {"type": "camera", **(PIPELINE if image is None else image)},
            "scan": scan or {},
        }
    )


def polynomial(**analyzer) -> AnalysisDiagnostic:
    return AnalysisDiagnostic.model_validate(
        {
            "name": "TestCam",
            "analyzer": {
                "kind": "magspec",
                "calibration": {"kind": "polynomial", "coeffs": [20.0, 0.5, 2e-3]},
                "energy_range": [30, 90],
                "num_energy_points": 200,
                **analyzer,
            },
            "image": {"type": "camera", **PIPELINE},
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


def legacy(doc: AnalysisDiagnostic, raw: np.ndarray, aux: dict | None = None):
    from image_analysis.analyzers.magspec_manual_calib_analyzer import (
        MagSpecManualCalibAnalyzer,
    )

    return MagSpecManualCalibAnalyzer(doc.image, spec=doc.analyzer).analyze_image(
        raw, aux
    )


def bound_tables(compiled) -> dict[str, Table]:
    return {request.key: read_table(request.path) for request in compiled.tables}


def assert_same_result(core, old) -> None:
    """Scalars, the resampled frame, its energy axis and the spectrum, 0 ulp."""
    assert set(core.scalars) == set(old.scalars) == MagspecSpec.scalar_docs.keys()
    for key, value in old.scalars.items():
        if math.isnan(value):
            assert math.isnan(core.scalars[key]), key
        else:
            assert core.scalars[key] == value, key
    np.testing.assert_array_equal(core.frame.data, old.processed_image)
    np.testing.assert_array_equal(
        core.frame.axes[1].values, old.render_data["energy_axis"]
    )
    assert (core.frame.axes[1].unit, core.frame.axes[1].label) == ("MeV", "Energy")
    spectrum = core.extras["spectrum"]
    np.testing.assert_array_equal(
        spectrum.data, np.sum(old.processed_image, axis=0) / 1000.0
    )
    assert spectrum.axes[0] is core.frame.axes[1]
    assert (spectrum.unit, spectrum.label) == ("pC/MeV", "Charge Density")


# ------------------------------------------------------------ polynomial / array


@pytest.mark.parametrize("seed", range(3))
def test_polynomial_route_equals_the_legacy_analyzer(seed):
    doc = polynomial()
    raw = frame(seed)
    # The whole kind waits for a host that serves it, whatever the calibration.
    with pytest.raises(UnsupportedRecipe, match="source host"):
        compile_v2(doc)
    core = analyze_v2(raw, compile_v2(doc, allow_table_inputs=True))
    assert_same_result(core, legacy(doc, raw))
    assert core.frame.unit == "counts/MeV" and core.overlays == ()


def test_array_calibration_is_inline_only():
    values = [20.0 + 0.5 * i for i in range(128)]
    doc = polynomial(calibration={"kind": "array", "values": values})
    raw = frame(0)
    compiled = compile_v2(doc, allow_table_inputs=True)
    assert_same_result(analyze_v2(raw, compiled), legacy(doc, raw))
    with pytest.raises(UnsupportedRecipe, match="file"):
        compile_v2(
            polynomial(calibration={"kind": "array", "file": "axis.npy"}),
            allow_table_inputs=True,
        )


# ------------------------------------------------------------ dnn_axis + tables


@pytest.mark.parametrize("camera", [1, 3])
@pytest.mark.parametrize("lanex", [True, False])
@pytest.mark.parametrize("energy_range", [(1.0, 1500.0), (60.0, 75.0)])
def test_dnn_route_equals_the_legacy_analyzer_on_the_same_tables(
    tmp_path, camera, lanex, energy_range
):
    doc = diagnostic(
        synthetic_tables(tmp_path),
        camera=camera,
        lanex=lanex,
        energy_range=energy_range,
    )
    with pytest.raises(UnsupportedRecipe, match="tables"):
        compile_v2(doc)
    compiled = compile_v2(doc, allow_table_inputs=True)
    assert [t.key for t in compiled.tables] == (
        ["camera_table", "trajectory_table", "lanex_table"]
        if lanex
        else ["camera_table", "trajectory_table"]
    )
    raw = frame(camera)
    core = analyze_v2(raw, compiled, inputs=bound_tables(compiled))
    assert_same_result(core, legacy(doc, raw))
    assert core.frame.unit == ("fC/MeV" if lanex else "counts/MeV")
    if energy_range[0] < 2:
        # The request is clipped to the calibration's reach, silently.
        axis = core.frame.axes[1].values
        assert axis[0] > 1.0 and axis[-1] < 1500.0
        assert core.notes == ()


def test_lanex_row_falls_back_to_position_without_a_setting_column(tmp_path):
    doc = diagnostic(synthetic_tables(tmp_path, lanex_by_setting=False), camera=3)
    compiled = compile_v2(doc, allow_table_inputs=True)
    raw = frame(3)
    assert_same_result(
        analyze_v2(raw, compiled, inputs=bound_tables(compiled)), legacy(doc, raw)
    )


def test_the_per_shot_field_equals_the_legacy_override_and_never_leaks(tmp_path):
    doc = diagnostic(synthetic_tables(tmp_path), field=0.3)
    compiled = compile_recipe(
        to_v3(doc, magnetic_field_column=COLUMN).recipe, allow_file_backgrounds=True
    )
    assert shot_scalar_columns(compiled.analysis.measure) == (COLUMN,)
    tables = bound_tables(compiled)
    raw = frame(5)
    with_field = analyze_v2(
        raw,
        compiled,
        shot=ShotMeta("TestCam", 1, scalars={COLUMN: 0.25}),
        inputs=tables,
    )
    assert_same_result(with_field, legacy(doc, raw, {LEGACY_COLUMN: 0.25}))
    # A second shot without the column: the recipe's field, as a fresh
    # legacy analyzer would give — not the previous shot's reading.
    without = analyze_v2(raw, compiled, shot=ShotMeta("TestCam", 2), inputs=tables)
    assert_same_result(without, legacy(doc, raw))
    assert without.scalars["peak_energy_MeV"] != with_field.scalars["peak_energy_MeV"]
    from image_analysis.analyzers.magspec_manual_calib_analyzer import (
        MagSpecManualCalibAnalyzer,
    )

    leaky = MagSpecManualCalibAnalyzer(doc.image, spec=doc.analyzer)
    leaky.analyze_image(raw, {LEGACY_COLUMN: 0.25})
    leaked = leaky.analyze_image(raw, None)
    assert leaked.scalars["peak_energy_MeV"] == with_field.scalars["peak_energy_MeV"]
    assert leaked.scalars["peak_energy_MeV"] != without.scalars["peak_energy_MeV"]


def test_the_field_rides_through_the_runner_on_shot_metadata(tmp_path):
    doc = diagnostic(synthetic_tables(tmp_path))
    compiled = compile_recipe(
        to_v3(doc, magnetic_field_column=COLUMN).recipe, allow_file_backgrounds=True
    )
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
    direct = analyze_v2(raw, compiled, shot=shot, inputs=tables)
    assert unit.error is None and dict(unit.measurement.scalars) == dict(direct.scalars)
    assert unit.measurement.frame.shot == shot


# ------------------------------------------------------------ the real tables


def real_tables() -> dict[str, Path]:
    import image_analysis

    root = Path(image_analysis.__file__).parent / "data" / "HTT-MagSpce-Calibrations"
    return {
        "camera": root / "260224DnnVarianCam.txt",
        "trajectory": root / "170925DnnVarianTrj.txt",
        "lanex": root / "250101lanexCalib.txt",
    }


@pytest.mark.parametrize("camera", [1, 2, 3, 4])
def test_the_real_calibration_equals_the_legacy_axis_and_charge_factor(camera):
    from image_analysis.analyzers.magspec_manual_calib_analyzer import (
        DnnAxisCalibration,
    )

    paths = real_tables()
    oracle = DnnAxisCalibration(
        camera_calibration_file=str(paths["camera"]),
        trajectory_calibration_file=str(paths["trajectory"]),
        lanex_calibration_file=str(paths["lanex"]),
        camera_number=camera,
        magnetic_field_t=0.2008,
    )
    tables = {key: read_table(path) for key, path in paths.items()}
    row = calibration.camera_row(tables["camera"], camera)
    geometry = calibration.camera_geometry(row)
    axis = calibration.dnn_energy_axis(
        geometry,
        calibration.trajectory(tables["trajectory"], camera),
        0.2008,
        geometry.width,
    )
    np.testing.assert_array_equal(axis, oracle.build_axis(geometry.width))
    assert calibration.charge_factor(row, tables["lanex"]) == oracle.get_charge_factor()
    # The HTT documents' request, clipped to the camera's own reach.
    image = np.random.default_rng(camera).normal(100, 10, (4, geometry.width))
    resampled, grid = calibration.resample_columns(image, axis, 1000, 1.2, 1501.3)
    assert (grid[0], grid[-1]) == (axis.min(), axis.max())
    from image_analysis.algorithms.axis_interpolation import interpolate_image_axis

    old, old_grid = interpolate_image_axis(image, axis, 1, 1000, 1.2, 1501.3)
    np.testing.assert_array_equal(resampled, old)
    np.testing.assert_array_equal(grid, old_grid)


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
    with pytest.raises(RecipeError, match="average_frames_first"):
        compile_recipe(bound, allow_file_backgrounds=True)
    # The converter leaves a per_bin document's column unset and says why:
    # the legacy per-bin override used the bin's mean field.
    doc = diagnostic(synthetic_tables(tmp_path), scan={"mode": "per_bin"})
    conversion = to_v3(doc, magnetic_field_column=COLUMN)
    written = canonical_document(conversion.recipe)["measure"]
    assert "magnetic_field_column" not in written
    assert any("bin's mean field" in note for note in conversion.notes)
    assert compile_recipe(conversion.recipe, allow_file_backgrounds=True)


def test_execution_refuses_a_missing_or_mistyped_table(tmp_path):
    compiled = compile_v2(
        diagnostic(synthetic_tables(tmp_path)), allow_table_inputs=True
    )
    tables = bound_tables(compiled)
    with pytest.raises(ValueError, match="Missing table input: lanex_table"):
        analyze_v2(
            frame(0),
            compiled,
            inputs={k: v for k, v in tables.items() if k != "lanex_table"},
        )
    with pytest.raises(TypeError, match="Table input 'camera_table' must be a Table"):
        analyze_v2(
            frame(0),
            compiled,
            inputs={**tables, "camera_table": Frame.from_array(np.ones((2, 2)))},
        )


@pytest.mark.parametrize(
    "change, needle",
    [
        (
            {"image": {**PIPELINE, "roi": {**PIPELINE["roi"], "x_max": 100}}},
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
    doc = diagnostic(paths, **change)
    compiled = compile_v2(doc, allow_table_inputs=True)
    with pytest.raises(ValueError, match=needle):
        analyze_v2(frame(0), compiled, inputs=bound_tables(compiled))
    with pytest.raises(ValueError, match=needle):
        legacy(doc, frame(0))


# ------------------------------------------------------------ conversion


def test_to_v3_moves_the_tables_under_a_namespace_and_names_the_field_column(tmp_path):
    doc = diagnostic(synthetic_tables(tmp_path))
    plain = to_v3(doc)
    assert plain.recipe.inputs["camera_table"].table == str(tmp_path / "cam.txt")
    assert any("not carried" in note for note in plain.notes)
    assert "magnetic_field_column" not in canonical_document(plain.recipe)["measure"]

    conversion = to_v3(doc, table_namespace="HTT", magnetic_field_column=COLUMN)
    inputs = conversion.recipe.inputs
    assert {k: v.table for k, v in inputs.items()} == {
        "camera_table": "calibrations/HTT/cam.txt",
        "trajectory_table": "calibrations/HTT/trj.txt",
        "lanex_table": "calibrations/HTT/lanex.txt",
    }
    assert sum("calibrations/HTT/" in note for note in conversion.notes) == 3
    written = canonical_document(conversion.recipe)
    assert written["measure"]["calibration"] == {
        "kind": "dnn_axis",
        "camera_number": 1,
        "magnetic_field_t": 0.3,
    }
    assert written["measure"]["magnetic_field_column"] == COLUMN
    reread = AnalysisRecipe.model_validate(written)
    compiled = compile_recipe(reread, allow_file_backgrounds=True)
    assert compiled == compile_recipe(conversion.recipe, allow_file_backgrounds=True)
    assert [t.path for t in compiled.tables] == [
        "calibrations/HTT/cam.txt",
        "calibrations/HTT/trj.txt",
        "calibrations/HTT/lanex.txt",
    ]
    assert compiled.analysis.measure.magnetic_field_column == COLUMN
    # A polynomial document ignores the column: nothing per shot to read.
    assert (
        "magnetic_field_column"
        not in canonical_document(
            to_v3(polynomial(), magnetic_field_column=COLUMN).recipe
        )["measure"]
    )
