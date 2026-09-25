"""Scan orchestration preserves bin membership and bare core measurements."""

import numpy as np
import pandas as pd
import pytest
from geecs_analysis.compat.v2 import UnsupportedRecipe
from geecs_schemas.analysis import AnalysisDiagnostic

from scan_analysis.core_scan import group_shots, prepare_scan


def document(mode="per_shot"):
    return AnalysisDiagnostic.model_validate(
        {
            "name": "Camera",
            "output_name": "Variant",
            "metric_suffix": "_roi",
            "analyzer": {"kind": "beam"},
            "image": {"type": "camera", "pipeline": []},
            "scan": {"mode": mode, "file_tail": ".npy"},
        }
    )


def inputs(tmp_path):
    device = tmp_path / "Camera"
    device.mkdir()
    y, x = np.mgrid[:20, :20]
    for shot in (1, 3):
        data = np.exp(-((x - 7 * shot / 2) ** 2 + (y - 10) ** 2) / 10)
        np.save(device / f"Scan001_Camera_{shot:03d}.npy", data)
    return pd.DataFrame({"Shotnumber": [3, 2, 1, 4], "Bin #": [2, 1, 1, 9]})


def test_groups_keep_row_order_and_full_membership_only_for_mapped_bins():
    rows = pd.DataFrame(
        {"Shotnumber": [3.0, 2, 1, 4, 5], "Bin #": [2, 1, 1, 9, np.nan]}
    )
    shot = group_shots(rows, [1, 3, 5], "per_shot")
    assert [(g.key, g.shots) for g in shot] == [(3, (3,)), (1, (1,)), (5, (5,))]
    bins = group_shots(rows, [1, 3, 5], "per_bin")
    assert [(g.key, g.shots) for g in bins] == [(2, (3,)), (1, (2, 1))]
    no_bins = group_shots(rows.drop(columns="Bin #"), [1, 3], "per_bin")
    assert [(g.key, g.shots) for g in no_bins] == [(0, (3, 1))]
    assert group_shots(rows, [], "per_bin") == ()


@pytest.mark.parametrize("shots", [[1, 1], [0], [True], [1.2], [np.nan], ["1"]])
def test_invalid_shot_identity_is_rejected(shots):
    with pytest.raises(ValueError, match="Shotnumber"):
        group_shots(pd.DataFrame({"Shotnumber": shots}), [1], "per_shot")


def test_fractional_bin_identity_is_not_silently_truncated():
    with pytest.raises(ValueError, match="Bin #"):
        group_shots(pd.DataFrame({"Shotnumber": [1], "Bin #": [1.5]}), [1], "per_bin")


@pytest.mark.parametrize("mode", ["per_shot", "per_bin"])
def test_execution_matches_legacy_arrays_and_scalar_naming_without_writes(
    tmp_path, mode
):
    from image_analysis.ephemeral import run_document_ephemeral

    rows = inputs(tmp_path)
    original_rows = rows.copy(deep=True)
    doc = document(mode)
    original_doc = doc.model_dump_json()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    scan = prepare_scan(doc, tmp_path, rows)
    outcomes = list(scan.run())
    assert [result.group.key for result in outcomes] == (
        [3, 1] if mode == "per_shot" else [2, 1]
    )
    for outcome in outcomes:
        raw = [scan.source.load(n) for n in outcome.loaded_shots]
        (old,) = run_document_ephemeral(
            doc, [np.mean(raw, axis=0)] if mode == "per_bin" else raw
        )
        assert outcome.error is None
        new = outcome.measurement
        np.testing.assert_array_equal(new.frame.data, old.processed_image)
        assert all(np.isfinite(v) for v in old.scalars.values())
        assert dict(new.scalars) == old.scalars
        records = scan.scalar_records(outcome)
        assert [r["Shotnumber"] for r in records] == list(outcome.group.shots)
        assert records[0]["Variant_x_CoM_roi"] == old.scalars["x_CoM"]
        assert "x_CoM" in new.scalars  # Projection never prefixes the core value.
    if mode == "per_bin":
        assert outcomes[1].loaded_shots == (1,)
        assert [f.shot for f in outcomes[1].load_failures] == [2]
        assert outcomes[1].group.shots == (2, 1)
    assert doc.model_dump_json() == original_doc
    pd.testing.assert_frame_equal(rows, original_rows)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert not (tmp_path / "analysis").exists()


def test_prepared_run_is_lazy_and_survives_caller_edits(tmp_path, monkeypatch):
    from scan_analysis import core_source

    rows = inputs(tmp_path)
    doc = document()
    calls = []
    reader = core_source.read_imaq_image

    def counted(path):
        calls.append(path)
        return reader(path)

    monkeypatch.setattr(core_source, "read_imaq_image", counted)
    scan = prepare_scan(doc, tmp_path, rows)
    doc.output_name = "Changed"
    doc.scan.mode = "per_bin"
    rows.loc[:, "Shotnumber"] = 99
    iterator = scan.run()
    assert not calls
    first = next(iterator)
    assert len(calls) == 1
    assert first.group.shots == (3,)
    assert "Variant_x_CoM_roi" in scan.scalar_records(first)[0]
    assert next(iterator).group.shots == (1,)


def test_failed_analysis_has_no_scalar_updates_and_later_shots_continue(tmp_path):
    rows = inputs(tmp_path)
    np.save(tmp_path / "Camera" / "Scan001_Camera_003.npy", np.ones((2, 2, 2)))
    scan = prepare_scan(document(), tmp_path, rows)
    failed, success = list(scan.run())
    assert failed.error and failed.measurement is None
    assert not scan.scalar_records(failed)
    assert success.error is None


def test_explicit_empty_output_name_retains_unprefixed_scalar_opt_in(tmp_path):
    rows = inputs(tmp_path)
    doc = document()
    doc.output_name = ""
    scan = prepare_scan(doc, tmp_path, rows)
    result = next(scan.run())
    record = scan.scalar_records(result)[0]
    assert "x_CoM_roi" in record
    assert "_x_CoM_roi" not in record


def test_unsupported_recipe_fails_before_background_or_shot_reads(
    tmp_path, monkeypatch
):
    from scan_analysis import core_inputs, core_source

    doc = document()
    data = doc.model_dump(mode="json")
    data["image"]["pipeline"] = ["transforms"]
    data["image"]["transforms"] = {"flip_horizontal": True}
    doc = AnalysisDiagnostic.model_validate(data)

    def forbidden(*args, **kwargs):
        raise AssertionError("Input I/O preceded capability validation")

    monkeypatch.setattr(core_inputs, "read_imaq_image", forbidden)
    monkeypatch.setattr(core_source, "map_shot_files", forbidden)
    with pytest.raises(UnsupportedRecipe):
        prepare_scan(doc, tmp_path, pd.DataFrame({"Shotnumber": [1]}))


def test_missing_scan_is_not_created(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_scan(document(), tmp_path / "scans" / "Scan999", pd.DataFrame())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["per_shot", "per_bin"])
def test_native_line_execution_keeps_scaled_axis_and_finite_metrics(tmp_path, mode):
    from image_analysis.ephemeral import run_document_ephemeral

    doc = AnalysisDiagnostic.model_validate(
        {
            "name": "Spectrum",
            "analyzer": {"kind": "line"},
            "image": {
                "type": "line",
                "data_loading": {"data_type": "npy"},
                "pipeline": [],
            },
            "scan": {"mode": mode, "file_tail": ".npy"},
        }
    )
    directory = tmp_path / "Spectrum"
    directory.mkdir()
    raw = []
    for shot in (1, 2):
        x = np.linspace(10, 20, 40, dtype=np.float32)
        data = np.column_stack((x, np.exp(-((x - 14 - shot) ** 2))))
        raw.append(data)
        np.save(directory / f"Scan001_Spectrum_{shot:03d}.npy", data)
    scan = prepare_scan(doc, tmp_path, pd.DataFrame({"Shotnumber": [1, 2]}))
    expected = run_document_ephemeral(
        doc, [np.mean(raw, axis=0)] if mode == "per_bin" else raw
    )
    for result, old in zip(scan.run(), expected, strict=True):
        assert result.error is None
        np.testing.assert_array_equal(
            result.measurement.frame.as_trace(), old.line_data
        )
        assert all(np.isfinite(v) for v in old.scalars.values())
        assert dict(result.measurement.scalars) == old.scalars
