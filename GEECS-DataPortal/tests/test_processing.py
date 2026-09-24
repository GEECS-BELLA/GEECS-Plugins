"""Core adoption, explicit legacy fallback and the portal's display contract."""

import builtins

import numpy as np
import pytest
import yaml

from geecs_portal.resources import figure_png

processing = pytest.importorskip("geecs_portal.processing")
AnalysisDiagnostic = pytest.importorskip("geecs_schemas.analysis").AnalysisDiagnostic


@pytest.fixture
def recipe_tree(tmp_path):
    folder = tmp_path / "analyzers"
    folder.mkdir()
    path = folder / "beam.yaml"
    document = {
        "name": "Camera",
        "image": {
            "type": "camera",
            "pipeline": ["roi", "background"],
            "roi": {"x_min": 2, "x_max": 15, "y_min": 3, "y_max": 17},
            "background": {"method": "constant", "constant_level": 5},
        },
        "analyzer": {"kind": "beam"},
    }
    path.write_text(yaml.safe_dump(document))
    return tmp_path, path, document


def test_supported_processing_and_unsaved_preview_do_not_import_legacy(
    recipe_tree, monkeypatch
):
    root, path, document = recipe_tree
    before = path.read_bytes()
    data = np.arange(400, dtype=np.uint16).reshape(20, 20)
    original = data.copy()
    imported = builtins.__import__

    def without_legacy(name, *args, **kwargs):
        if name.startswith("image_analysis"):
            raise AssertionError("ported recipe imported legacy analysis")
        return imported(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_legacy)
    assert processing.list_diagnostics(config_dir=root) == ["beam"]
    (array,) = processing.process_images("beam", [data], config_dir=root)
    np.testing.assert_array_equal(array, data[3:17, 2:15].astype(float) - 5)
    document["image"]["roi"]["x_min"] = 4
    (figure,) = processing.render_document_ephemeral(
        AnalysisDiagnostic.model_validate(document),
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


def test_only_capability_refusal_selects_legacy(recipe_tree, monkeypatch):
    from image_analysis import ephemeral

    root, path, document = recipe_tree
    document["image"]["pipeline"].append("transforms")
    document["image"]["transforms"] = {"flip_horizontal": True}
    path.write_text(yaml.safe_dump(document))
    array = np.arange(400, dtype=float).reshape(20, 20)
    doc = AnalysisDiagnostic.model_validate(document)
    (legacy,) = ephemeral.run_document_ephemeral(doc, [array])
    (actual,) = processing.process_images("beam", [array], config_dir=root)
    np.testing.assert_array_equal(actual, legacy.processed_image)
    (fig,) = processing.render_document_ephemeral(doc, [array])
    np.testing.assert_array_equal(
        fig.axes[0].images[0].get_array(), legacy.processed_image
    )

    document["image"]["pipeline"].remove("transforms")
    path.write_text(yaml.safe_dump(document))

    def fail_core(*args, **kwargs):
        raise RuntimeError("core failure")

    def forbid_retry(*args, **kwargs):
        raise AssertionError("unexpected legacy retry")

    import scan_analysis.core_preview as core_preview

    monkeypatch.setattr(processing, "analyze_v2", fail_core)
    # the render path analyses through ScanAnalysis' preview seam
    monkeypatch.setattr(core_preview, "analyze_v2", fail_core)
    monkeypatch.setattr(ephemeral, "run_document_ephemeral", forbid_retry)
    monkeypatch.setattr(ephemeral, "render_document_ephemeral", forbid_retry)
    with pytest.raises(RuntimeError, match="core failure"):
        processing.process_images("beam", [array], config_dir=root)
    with pytest.raises(RuntimeError, match="core failure"):
        processing.render_document_ephemeral(
            AnalysisDiagnostic.model_validate(document), [array]
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
    from image_analysis import ephemeral

    _, _, document = recipe_tree

    def forbid_retry(*args, **kwargs):
        raise AssertionError("unexpected legacy retry")

    monkeypatch.setattr(ephemeral, "render_document_ephemeral", forbid_retry)
    with pytest.raises(processing.RenderError):
        processing.render_document_ephemeral(
            AnalysisDiagnostic.model_validate(document),
            [np.ones((20, 20))],
            cmap="missing-colormap",
        )


def test_trace_preview_retains_physical_axis_and_never_mutates_input():
    document = AnalysisDiagnostic.model_validate(
        {
            "name": "Spectrum",
            "analyzer": {"kind": "line"},
            "image": {
                "type": "line",
                "data_loading": {"data_type": "tsv"},
                "x_scale_factor": 1000,
                "x_units": "MeV",
            },
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
    document["image"]["pipeline"] = ["background", "roi"]
    document["image"]["background"] = {
        "method": "from_file",
        "file_path": str(background_path),
        "constant_level": 11,
        "additional_constant": 2,
    }
    path.write_text(yaml.safe_dump(document))
    before = path.read_bytes()
    imported = builtins.__import__

    def without_legacy(name, *args, **kwargs):
        if name.startswith("image_analysis"):
            raise AssertionError("file background imported legacy analysis")
        return imported(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_legacy)
    data = np.arange(400, dtype=np.uint16).reshape(20, 20)
    expected = data[3:17, 2:15].astype(float) - (9 if available else 13)
    arrays = processing.process_images("beam", [data, data], config_dir=root)
    for array in arrays:
        np.testing.assert_array_equal(array, expected)
    document["image"]["background"]["additional_constant"] = 4
    (figure,) = processing.render_document_ephemeral(
        AnalysisDiagnostic.model_validate(document), [data]
    )
    np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), expected - 2)
    assert path.read_bytes() == before


def test_file_background_geometry_error_does_not_retry_legacy(recipe_tree, monkeypatch):
    from image_analysis import ephemeral

    root, path, document = recipe_tree
    background_path = root / "wrong-shape.npy"
    np.save(background_path, np.ones((1, 1)))
    document["image"]["background"] = {
        "method": "from_file",
        "file_path": str(background_path),
        "constant_level": 11,
    }
    path.write_text(yaml.safe_dump(document))

    def forbid_retry(*args, **kwargs):
        raise AssertionError("unexpected legacy retry")

    monkeypatch.setattr(ephemeral, "run_document_ephemeral", forbid_retry)
    monkeypatch.setattr(ephemeral, "render_document_ephemeral", forbid_retry)
    with pytest.raises(ValueError, match="shape"):
        processing.process_images("beam", [np.ones((20, 20))], config_dir=root)
    with pytest.raises(ValueError, match="shape"):
        processing.render_document_ephemeral(
            AnalysisDiagnostic.model_validate(document), [np.ones((20, 20))]
        )
