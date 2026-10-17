"""File resolution preserves legacy numerical and failure contracts without writes."""

import numpy as np
import pytest
from geecs_analysis.engine.compiled import analyze_compiled
from geecs_analysis.recipe import RecipeError
from geecs_schemas.analysis import AnalysisRecipe

from scan_analysis import core_inputs


def recipe(path) -> AnalysisRecipe:
    frame = {
        "step": "background_frame",
        "source": "camera_background",
        "alignment": "samples",
    }
    constant = {"step": "background_constant", "level": -2}
    return AnalysisRecipe.model_validate(
        {
            "device": "Camera",
            "input": {"kind": "camera"},
            "inputs": {
                "camera_background": {"path": str(path), "fallback_level": 13.0}
            },
            "steps": [
                {"step": "roi", "bounds": [[1, 9], [1, 9]]},
                frame,
                constant,
                frame,
                constant,
            ],
            "measure": {"kind": "beam"},
            "figure": {"imshow": {"vmin": 0}, "fig": {"dpi": 150}},
            "summaries": [
                {"kind": "image_grid", "panel_size": [6.0, 6.0]},
                {"kind": "average"},
            ],
        }
    )


@pytest.mark.parametrize("state", ["loaded", "missing", "corrupt"])
def test_file_backgrounds_and_fallbacks_match_old_backend(
    tmp_path, state, caplog, measurement_golden
):
    path = tmp_path / "dark.npy"
    if state == "loaded":
        np.save(path, np.arange(64).reshape(8, 8) / 10)
    elif state == "corrupt":
        path.write_bytes(b"not an array")
    doc = recipe(path)
    before = doc.model_dump_json()
    files = {p: p.read_bytes() for p in tmp_path.iterdir()}
    raw = np.random.default_rng(7).integers(100, 200, (10, 10), dtype=np.uint16)

    prepared = core_inputs.prepare_recipe(doc)
    result = analyze_compiled(raw, prepared.recipe, inputs=prepared.inputs)
    assert all(np.isfinite(v) for v in result.scalars.values())
    measurement_golden(
        f"inputs/file_background_{state}",
        result.frame.data,
        dict(result.scalars),
    )
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
    doc = recipe("{scan_dir}/dark.npy")
    calls = []
    reader = core_inputs.read_imaq_image

    def counted(path):
        calls.append(path)
        return reader(path)

    monkeypatch.setattr(core_inputs, "read_imaq_image", counted)
    prepared = core_inputs.prepare_recipe(doc, data_dir=device)
    # Repeated steps and multiple frames reuse the loaded snapshot.
    for _ in range(2):
        result = analyze_compiled(
            np.full((10, 10), 100), prepared.recipe, inputs=prepared.inputs
        )
        np.testing.assert_array_equal(result.frame.data, np.full((8, 8), 98))
    assert calls == [device / "dark.npy"]
    assert doc.inputs["camera_background"].path == "{scan_dir}/dark.npy"
    with pytest.raises(TypeError):
        prepared.inputs["camera_background"] = None


def test_context_free_preview_keeps_placeholder_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    doc = recipe("{scan_dir}/dark.npy")
    prepared = core_inputs.prepare_recipe(doc)
    assert not prepared.inputs
    result = analyze_compiled(np.full((10, 10), 100), prepared.recipe)
    np.testing.assert_array_equal(result.frame.data, np.full((8, 8), 78))
    assert not list(tmp_path.iterdir())


def test_missing_scan_tree_is_not_created_by_background_resolution(tmp_path):
    data_dir = tmp_path / "scans" / "Scan999" / "Camera"
    prepared = core_inputs.prepare_recipe(
        recipe("{scan_dir}/dark.npy"), data_dir=data_dir
    )
    assert not prepared.inputs
    assert not (tmp_path / "scans").exists()


def test_loaded_shape_mismatch_raises_without_constant_fallback(tmp_path, caplog):
    path = tmp_path / "dark.npy"
    np.save(path, np.ones((1, 8)))
    prepared = core_inputs.prepare_recipe(recipe(path))
    with pytest.raises(ValueError, match="shape"):
        analyze_compiled(np.ones((10, 10)), prepared.recipe, inputs=prepared.inputs)
    assert "Falling back" not in caplog.text


def test_compile_refusal_happens_before_reading_any_file(tmp_path, monkeypatch):
    data = recipe(tmp_path / "dark.npy").model_dump(mode="json")
    data["steps"].append({"step": "no_such_step"})
    doc = AnalysisRecipe.model_validate(data)

    def forbid_read(*args):
        raise AssertionError("read before capability validation")

    monkeypatch.setattr(core_inputs, "read_imaq_image", forbid_read)
    with pytest.raises(RecipeError, match="does not bind"):
        core_inputs.prepare_recipe(doc)
