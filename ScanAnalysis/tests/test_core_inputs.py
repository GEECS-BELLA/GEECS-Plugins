"""File resolution preserves legacy numerical and failure contracts without writes."""

import subprocess
import sys

import numpy as np
import pytest
from geecs_analysis.compat.v2 import UnsupportedRecipe, analyze_v2
from geecs_schemas.analysis import AnalysisDiagnostic

from scan_analysis import core_inputs


def document(path):
    return AnalysisDiagnostic.model_validate(
        {
            "name": "Camera",
            "analyzer": {"kind": "beam"},
            "image": {
                "type": "camera",
                "pipeline": ["roi", "background", "background"],
                "roi": {"x_min": 1, "x_max": 9, "y_min": 1, "y_max": 9},
                "background": {
                    "method": "from_file",
                    "file_path": str(path),
                    "constant_level": 13,
                    "additional_constant": -2,
                },
            },
        }
    )


@pytest.mark.parametrize("state", ["loaded", "missing", "corrupt"])
def test_file_backgrounds_and_fallbacks_match_old_backend(tmp_path, state, caplog):
    from image_analysis.ephemeral import run_document_ephemeral

    path = tmp_path / "dark.npy"
    if state == "loaded":
        np.save(path, np.arange(64).reshape(8, 8) / 10)
    elif state == "corrupt":
        path.write_bytes(b"not an array")
    doc = document(path)
    before = doc.model_dump_json()
    files = {p: p.read_bytes() for p in tmp_path.iterdir()}
    raw = np.random.default_rng(7).integers(100, 200, (10, 10), dtype=np.uint16)
    (old,) = run_document_ephemeral(doc, [raw])
    prepared = core_inputs.prepare_v2(doc)
    result = analyze_v2(raw, prepared.recipe, inputs=prepared.inputs)
    np.testing.assert_array_equal(result.frame.data, old.processed_image)
    assert result.scalars.keys() == old.scalars.keys()
    assert all(np.isfinite(v) for v in old.scalars.values())
    assert dict(result.scalars) == old.scalars
    assert doc.model_dump_json() == before
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == files
    if state != "loaded":
        assert not prepared.inputs
        assert "Falling back to constant_level=13" in caplog.text


def test_device_directory_resolution_and_single_load_are_explicit(
    tmp_path, monkeypatch
):
    device = tmp_path / "scans" / "Scan001" / "Camera"
    device.mkdir(parents=True)
    np.save(device / "dark.npy", np.full((8, 8), 3))
    doc = document("{scan_dir}/dark.npy")
    calls = []
    reader = core_inputs.read_imaq_image

    def counted(path):
        calls.append(path)
        return reader(path)

    monkeypatch.setattr(core_inputs, "read_imaq_image", counted)
    prepared = core_inputs.prepare_v2(doc, data_dir=device)
    # Repeated steps and multiple frames reuse the loaded snapshot.
    for _ in range(2):
        result = analyze_v2(
            np.full((10, 10), 100), prepared.recipe, inputs=prepared.inputs
        )
        np.testing.assert_array_equal(result.frame.data, np.full((8, 8), 98))
    assert calls == [device / "dark.npy"]
    assert doc.image.background.file_path == "{scan_dir}/dark.npy"
    with pytest.raises(TypeError):
        prepared.inputs["camera_background"] = None


def test_context_free_preview_keeps_placeholder_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    doc = document("{scan_dir}/dark.npy")
    prepared = core_inputs.prepare_v2(doc)
    assert not prepared.inputs
    result = analyze_v2(np.full((10, 10), 100), prepared.recipe)
    np.testing.assert_array_equal(result.frame.data, np.full((8, 8), 78))
    assert not list(tmp_path.iterdir())


def test_missing_scan_tree_is_not_created_by_background_resolution(tmp_path):
    data_dir = tmp_path / "scans" / "Scan999" / "Camera"
    prepared = core_inputs.prepare_v2(
        document("{scan_dir}/dark.npy"), data_dir=data_dir
    )
    assert not prepared.inputs
    assert not (tmp_path / "scans").exists()


def test_loaded_shape_mismatch_raises_without_constant_fallback(tmp_path, caplog):
    path = tmp_path / "dark.npy"
    np.save(path, np.ones((1, 8)))
    prepared = core_inputs.prepare_v2(document(path))
    with pytest.raises(ValueError, match="shape"):
        analyze_v2(np.ones((10, 10)), prepared.recipe, inputs=prepared.inputs)
    assert "Falling back" not in caplog.text


def test_compile_refusal_happens_before_reading_any_file(monkeypatch):
    data = document("unused.npy").model_dump(mode="json")
    data["image"]["pipeline"].append("transforms")
    data["image"]["transforms"] = {"flip_horizontal": True}
    doc = AnalysisDiagnostic.model_validate(data)

    def forbid_read(*args):
        raise AssertionError("read before capability validation")

    monkeypatch.setattr(core_inputs, "read_imaq_image", forbid_read)
    with pytest.raises(UnsupportedRecipe):
        core_inputs.prepare_v2(doc)


def test_preparation_has_no_legacy_analyzer_imports(tmp_path):
    path = tmp_path / "dark.npy"
    np.save(path, np.ones((8, 8)))
    code = f"""
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('image_analysis'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from geecs_schemas.analysis import AnalysisDiagnostic
from scan_analysis.core_inputs import prepare_v2
assert prepare_v2(AnalysisDiagnostic.model_validate_json({document(path).model_dump_json()!r})).inputs
"""
    subprocess.run([sys.executable, "-c", code], check=True)
