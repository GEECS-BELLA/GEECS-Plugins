"""Every portal draw is the core's; refusals are 400s, and the display contract."""

import numpy as np
import pytest
import yaml

from geecs_portal.resources import figure_png

processing = pytest.importorskip("geecs_portal.processing")
AnalysisRecipe = pytest.importorskip("geecs_schemas.analysis").AnalysisRecipe


@pytest.fixture
def recipe_tree(tmp_path):
    folder = tmp_path / "analyzers"
    folder.mkdir()
    path = folder / "beam.yaml"
    document = {
        "schema_version": 3,
        "device": "Camera",
        "input": {"kind": "camera"},
        "steps": [
            {"step": "roi", "bounds": [[3, 17], [2, 15]]},
            {"step": "background_constant", "level": 5},
        ],
        "measure": {"kind": "beam"},
    }
    path.write_text(yaml.safe_dump(document))
    return tmp_path, path, document


def test_processing_and_unsaved_preview_run_on_the_core(recipe_tree, monkeypatch):
    root, path, document = recipe_tree
    before = path.read_bytes()
    data = np.arange(400, dtype=np.uint16).reshape(20, 20)
    original = data.copy()
    assert processing.list_diagnostics(config_dir=root) == ["beam"]
    (array,) = processing.process_images("beam", [data], config_dir=root)
    np.testing.assert_array_equal(array, data[3:17, 2:15].astype(float) - 5)
    document["steps"][0]["bounds"][1][0] = 4
    (figure,) = processing.render_document_ephemeral(
        AnalysisRecipe.model_validate(document),
        [data],
        cmap="viridis",
        vmin=10,
        vmax=200,
    )
    artist = figure.axes[0].images[0]
    np.testing.assert_array_equal(
        artist.get_array(), data[3:17, 4:15].astype(float) - 5
    )
    assert tuple(artist.get_extent()) == (3.5, 14.5, 16.5, 2.5)
    assert artist.get_clim() == (10, 200)
    assert artist.get_cmap().name == "viridis"
    assert len(figure.axes[0].lines) == 3  # two projections and the centroid
    assert figure_png(figure).startswith(b"\x89PNG")
    assert path.read_bytes() == before
    np.testing.assert_array_equal(data, original)


def test_a_document_the_core_declines_is_refused_not_rerouted(recipe_tree, monkeypatch):
    """A compilation-time refusal is the portal's 400, in every view (#1039)."""
    root, path, document = recipe_tree
    document["inputs"] = {"bg": {"from_scan": {"statistic": "median"}}}
    document["steps"].append({"step": "background_frame", "source": "bg"})
    document["summaries"] = [{"kind": "average"}]
    path.write_text(yaml.safe_dump(document))
    array = np.arange(400, dtype=float).reshape(20, 20)
    doc = AnalysisRecipe.model_validate(document)
    with pytest.raises(ValueError, match="cannot be drawn here"):
        processing.process_images("beam", [array], config_dir=root)
    with pytest.raises(ValueError, match="cannot be drawn here"):
        processing.render_document_ephemeral(doc, [array])
    with pytest.raises(ValueError, match="cannot be drawn here"):
        processing.render_document_as_run(doc, [array])
    with pytest.raises(ValueError, match="no summary preview"):
        processing.render_summary_as_run(doc, [array], [None], "shot", 0)


def test_a_core_failure_propagates_without_a_second_backend(recipe_tree, monkeypatch):
    root, path, document = recipe_tree
    array = np.arange(400, dtype=float).reshape(20, 20)

    def fail_core(*args, **kwargs):
        raise RuntimeError("core failure")

    import scan_analysis.core_preview as core_preview

    # every view analyses through ScanAnalysis' preview seam
    monkeypatch.setattr(core_preview, "analyze_v2", fail_core)
    with pytest.raises(RuntimeError, match="core failure"):
        processing.process_images("beam", [array], config_dir=root)
    with pytest.raises(RuntimeError, match="core failure"):
        processing.render_document_ephemeral(
            AnalysisRecipe.model_validate(document), [array]
        )


def test_display_window_and_bin_average_have_no_shot_overlays():
    array = np.arange(100, dtype=float).reshape(10, 10)
    array[0, :2] = [np.nan, np.inf]
    figure = processing.render_frame_figure(array, window=(10, 90), cmap="gray")
    expected = np.percentile(array[np.isfinite(array)], (10, 90))
    assert figure.axes[0].images[0].get_clim() == tuple(expected)
    assert not figure.axes[0].lines
    assert len(figure.axes) == 2  # image and colorbar


def test_bad_core_render_is_a_render_error_without_fallback(recipe_tree, monkeypatch):
    _, _, document = recipe_tree
    with pytest.raises(processing.RenderError):
        processing.render_document_ephemeral(
            AnalysisRecipe.model_validate(document),
            [np.ones((20, 20))],
            cmap="missing-colormap",
        )


def test_trace_preview_retains_physical_axis_and_never_mutates_input():
    document = AnalysisRecipe.model_validate(
        {
            "schema_version": 3,
            "device": "Spectrum",
            "input": {
                "kind": "line",
                "loading": {"data_type": "tsv"},
                "x_scale": 1000,
                "x_unit": "MeV",
            },
            "measure": {"kind": "line"},
        }
    )
    data = np.column_stack(
        (np.linspace(0.05, 0.15, 40), np.sin(np.linspace(0, np.pi, 40)))
    )
    before = data.copy()
    (figure,) = processing.render_document_ephemeral(document, [data])
    np.testing.assert_allclose(
        figure.axes[0].lines[0].get_xdata(), data[:, 0] * 1000, rtol=1e-7
    )
    assert "MeV" in figure.axes[0].get_xlabel()
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize("available", [True, False])
def test_file_background_processing_and_unsaved_preview_use_core(
    recipe_tree, monkeypatch, available
):
    root, path, document = recipe_tree
    background_path = root / "dark.npy"
    if available:
        np.save(background_path, np.full((20, 20), 7))
    document["inputs"] = {"dark": {"path": str(background_path), "fallback_level": 11}}
    document["steps"] = [
        {"step": "background_frame", "source": "dark", "alignment": "samples"},
        {"step": "background_constant", "level": 2},
        document["steps"][0],
    ]
    path.write_text(yaml.safe_dump(document))
    before = path.read_bytes()
    data = np.arange(400, dtype=np.uint16).reshape(20, 20)
    expected = data[3:17, 2:15].astype(float) - (9 if available else 13)
    arrays = processing.process_images("beam", [data, data], config_dir=root)
    for array in arrays:
        np.testing.assert_array_equal(array, expected)
    document["steps"][1]["level"] = 4
    (figure,) = processing.render_document_ephemeral(
        AnalysisRecipe.model_validate(document), [data]
    )
    np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), expected - 2)
    assert path.read_bytes() == before


def test_file_background_geometry_error_does_not_retry_legacy(recipe_tree, monkeypatch):
    root, path, document = recipe_tree
    background_path = root / "wrong-shape.npy"
    np.save(background_path, np.ones((1, 1)))
    document["inputs"] = {"dark": {"path": str(background_path), "fallback_level": 11}}
    document["steps"].insert(
        0, {"step": "background_frame", "source": "dark", "alignment": "samples"}
    )
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="shape"):
        processing.process_images("beam", [np.ones((20, 20))], config_dir=root)
    with pytest.raises(ValueError, match="shape"):
        processing.render_document_ephemeral(
            AnalysisRecipe.model_validate(document), [np.ones((20, 20))]
        )


def test_a_v3_recipe_the_core_declines_is_a_clear_refusal():
    """A scan background no run computed is a 400 naming the reason."""
    from geecs_schemas.analysis import AnalysisRecipe

    from geecs_portal import processing

    recipe = AnalysisRecipe.model_validate(
        {
            "schema_version": 3,
            "device": "Cam",
            "input": {"kind": "camera"},
            "inputs": {"camera_background": {"from_scan": {"statistic": "median"}}},
            "steps": [{"step": "background_frame", "source": "camera_background"}],
        }
    )
    with pytest.raises(ValueError, match="cannot be drawn here.*scan background"):
        processing.render_document_as_run(recipe, [np.ones((4, 5))])
