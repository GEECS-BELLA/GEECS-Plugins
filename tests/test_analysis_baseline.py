"""Migration-baseline checks: changed numerics must not silently compare equal."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

SPEC = importlib.util.spec_from_file_location(
    "analysis_baseline",
    Path(__file__).resolve().parents[1] / "scripts/analysis_baseline.py",
)
baseline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = baseline
SPEC.loader.exec_module(baseline)


def snapshot():
    return baseline.Snapshot(
        recipe='{"kind":"beam"}',
        inputs=("shot:hash",),
        mode="per_shot",
        results=(baseline.Result(np.arange(6.0).reshape(2, 3), {"total": 15.0}),),
    )


def test_round_trip_has_no_pickle_and_cannot_overwrite_reference(tmp_path):
    original = snapshot()
    output = tmp_path / "reference.npz"
    baseline.save(original, output)
    assert baseline.compare(original, baseline.load(output)) == []
    with np.load(output, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        baseline.save(original, output)
    assert output.read_bytes() == before


@pytest.mark.parametrize("existing", [False, True])
def test_cannot_create_or_write_inside_raw_scan(tmp_path, existing):
    raw_scan = tmp_path / "scans" / "Scan001"
    if existing:
        raw_scan.mkdir(parents=True)
    with pytest.raises(ValueError, match="outside scans"):
        baseline.save(snapshot(), raw_scan / "baseline.npz")
    assert raw_scan.exists() is existing
    if existing:
        assert list(raw_scan.iterdir()) == []


def test_scan_guard_resolves_symlinks(tmp_path):
    raw_scan = tmp_path / "scans" / "Scan001"
    raw_scan.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(raw_scan, target_is_directory=True)
    with pytest.raises(ValueError, match="outside scans"):
        baseline.save(snapshot(), alias / "baseline.npz")


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"recipe": "different"}, "workload recipe differs"),
        ({"inputs": ("shot:new-hash",)}, "workload inputs differs"),
        ({"mode": "per_bin"}, "workload mode differs"),
        ({"results": ()}, "empty baseline cannot establish parity"),
    ],
)
def test_mismatched_workloads_do_not_compare_equal(changes, message):
    original = snapshot()
    assert message in baseline.compare(original, replace(original, **changes))


def test_changed_pixels_scalar_names_and_values_are_reported():
    original = snapshot()
    altered = replace(
        original,
        results=(
            baseline.Result(original.results[0].data + 1, {"total": 16, "extra": 1}),
        ),
    )
    report = "\n".join(baseline.compare(original, altered))
    assert "result 0 data: values differ" in report
    assert "extra=['extra']" in report
    assert "scalar total: values differ" in report
    assert "missing=['total']" in "\n".join(
        baseline.compare(
            original,
            replace(original, results=(baseline.Result(np.zeros((3, 2)), {}),)),
        )
    )


def test_shape_change_cannot_broadcast_to_a_pass():
    original = snapshot()
    altered = replace(
        original, results=(baseline.Result(np.arange(3.0), {"total": 15}),)
    )
    assert "shape" in baseline.compare(original, altered)[0]


@pytest.mark.parametrize("candidate_dtype", [np.int64, np.float64])
def test_exact_comparison_preserves_large_integer_differences(candidate_dtype):
    original = replace(
        snapshot(), results=(baseline.Result(np.array([2**53 + 1]), {}),)
    )
    altered = replace(
        original,
        results=(baseline.Result(np.array([2**53], dtype=candidate_dtype), {}),),
    )
    assert baseline.compare(original, altered)


def test_empty_baselines_cannot_pass_or_be_saved(tmp_path):
    empty = replace(snapshot(), inputs=(), results=())
    assert baseline.compare(empty, empty)
    with pytest.raises(ValueError, match="empty"):
        baseline.save(empty, tmp_path / "empty.npz")


@pytest.mark.parametrize("kind", ["beam", "line"])
def test_external_backgrounds_are_rejected_before_reading(tmp_path, kind):
    config, _ = write_case(tmp_path, kind)
    doc = yaml.safe_load(config.read_text())
    doc["image"]["background"] = {"method": "from_file", "file_path": "missing.npy"}
    doc["image"]["pipeline"] = ["background"]
    config.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="dependencies"):
        baseline.capture(config, [tmp_path / "missing-shot"])


def test_external_vignette_is_rejected_before_reading(tmp_path):
    config, _ = write_case(tmp_path, "beam")
    doc = yaml.safe_load(config.read_text())
    doc["image"]["vignette"] = {"method": "map_file", "map_file_path": "missing.npy"}
    doc["image"]["pipeline"] = ["vignette"]
    config.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="dependencies"):
        baseline.capture(config, [tmp_path / "missing-shot"])


def test_explicit_tolerance_and_nonfinite_values():
    original = snapshot()
    altered = replace(
        original,
        results=(
            baseline.Result(original.results[0].data + 1e-8, {"total": 15 + 1e-8}),
        ),
    )
    assert baseline.compare(original, altered)
    assert baseline.compare(original, altered, atol=1e-7) == []
    for value in (np.nan, np.inf, -np.inf):
        bad = replace(
            original, results=(baseline.Result(np.array([value]), {"x": value}),)
        )
        assert len(baseline.compare(bad, bad)) == 2
    for tolerance in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="Tolerances"):
            baseline.compare(original, original, atol=tolerance)


def write_case(tmp_path, kind):
    doc = {
        "schema_version": 2,
        "name": "Fixture",
        "analyzer": {"kind": kind},
        "image": {"type": "camera" if kind == "beam" else "line"},
    }
    axis = np.linspace(-5, 5, 64)
    if kind == "beam":
        from PIL import Image

        frame = (1000 * np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2))).astype(
            "uint16"
        )
        path = tmp_path / "shot.png"
        Image.fromarray(frame).save(path)
    else:
        frame = np.column_stack([axis, 1000 * np.exp(-(axis**2))])
        path = tmp_path / "shot.npy"
        np.save(path, frame)
        doc["image"]["data_loading"] = {"data_type": "npy"}
    config = tmp_path / "diagnostic.yaml"
    config.write_text(yaml.safe_dump(doc))
    return config, path


@pytest.mark.parametrize("kind", ["beam", "line"])
def test_capture_matches_file_analysis_and_leaves_inputs_untouched(tmp_path, kind):
    from image_analysis.config import create_image_analyzer, load_diagnostic

    config, path = write_case(tmp_path, kind)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    actual = baseline.capture(config, [path])
    expected = create_image_analyzer(load_diagnostic(config)).analyze_image_file(path)
    assert actual.results[0].scalars == expected.scalars
    np.testing.assert_array_equal(actual.results[0].data, expected.get_primary_data())
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert baseline.compare(actual, baseline.capture(config, [path])) == []


def test_per_bin_averages_inputs_before_analysis(tmp_path):
    from image_analysis.config import create_image_analyzer, load_diagnostic
    from image_analysis.ephemeral import run_document_ephemeral
    from PIL import Image

    config, first = write_case(tmp_path, "beam")
    second = tmp_path / "second.png"
    Image.fromarray(np.full((64, 64), 10, dtype=np.uint16)).save(second)
    diag = load_diagnostic(config)
    loader = create_image_analyzer(diag)
    mean = np.mean([loader.load_image(first), loader.load_image(second)], axis=0)
    (expected,) = run_document_ephemeral(diag, [mean])
    actual = baseline.capture(config, [first, second], "per_bin")
    assert len(actual.results) == 1
    assert actual.results[0].scalars == expected.scalars
    np.testing.assert_array_equal(actual.results[0].data, expected.get_primary_data())


def test_capture_rejects_unsupported_workloads_before_loading(tmp_path):
    config, path = write_case(tmp_path, "line")
    with pytest.raises(ValueError, match="camera-frame"):
        baseline.capture(config, [path], "per_bin")
    with pytest.raises(ValueError, match="At least one"):
        baseline.capture(config, [])
    config.write_text(
        "schema_version: 2\nname: FROG\nanalyzer: {kind: frog_retrieval}\nimage: {type: camera}\n"
    )
    with pytest.raises(ValueError, match="Only beam and line"):
        baseline.capture(config, [tmp_path / "missing.png"])


def test_cli_comparison_exit_code(tmp_path, capsys):
    old, new = tmp_path / "old.npz", tmp_path / "new.npz"
    baseline.save(snapshot(), old)
    baseline.save(replace(snapshot(), inputs=("different",)), new)
    assert baseline.main(["compare", str(old), str(old)]) == 0
    assert baseline.main(["compare", str(old), str(new)]) == 1
    assert "workload inputs differs" in capsys.readouterr().out
