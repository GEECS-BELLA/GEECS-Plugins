"""The preview seam draws exactly what the sink writes: byte-equal PNGs."""

from __future__ import annotations

import io

import numpy as np
import pytest
import yaml
from geecs_analysis.compat.v2 import UnsupportedRecipe
from geecs_schemas.analysis import load_analysis_document

from scan_analysis.core_preview import (
    measure_frame,
    prepare_document,
    preview_frame,
    preview_summary,
)
from scan_analysis.core_products import Product, ProductPlan
from scan_analysis.core_recipe import scan_recipe
from scan_analysis.core_sink import save_products

RECIPE = {
    "schema_version": 3,
    "device": "cam",
    "input": {"kind": "camera"},
    "inputs": {"bg": {"path": "{scan_dir}/bg.npy"}},
    "steps": [
        {"step": "background_frame", "source": "bg"},
        {"step": "roi", "bounds": [[1, 11], [2, 14]]},
    ],
    "measure": {"kind": "beam"},
    "figure": {"imshow": {"cmap": "viridis", "vmin": 0}, "axes": {"title": "t"}},
    "summaries": [
        {"kind": "image_grid", "columns": 2, "panel_size": [3.0, 2.5]},
        {"kind": "average"},
    ],
}


@pytest.fixture()
def scan(tmp_path):
    """An existing scans/ScanNNN folder with the device folder and a background."""
    folder = tmp_path / "scans" / "Scan003"
    (folder / "cam").mkdir(parents=True)
    np.save(folder / "cam" / "bg.npy", np.full((12, 16), 100.0))
    (tmp_path / "analysis").mkdir()
    return folder


def _png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()


def _frames(n=3):
    rng = np.random.default_rng(5)
    return [rng.integers(200, 4000, size=(12, 16)).astype(np.uint16) for _ in range(n)]


def test_frame_preview_is_the_sinks_shot_product(scan):
    """preview_frame == the *_processed_visual.png the sink writes for that frame."""
    document = load_analysis_document(RECIPE)
    frame = _frames(1)[0]
    prepared = prepare_document(document, scan_folder=scan)
    # the background was loaded from the document's device folder, not left literal
    assert "bg" in prepared.inputs and float(prepared.inputs["bg"].data.mean()) == 100.0
    spec = scan_recipe(document)
    saved = save_products(
        ProductPlan(singles=(Product(7, measure_frame(prepared, frame)),)), spec, scan
    )
    (png,) = [p for p in saved.files if p.suffix == ".png"]
    assert png.name == "cam_7_processed_visual.png"
    assert _png(preview_frame(document, frame, scan_folder=scan)) == png.read_bytes()
    # without the scan folder the placeholder stays literal: the recipe has no
    # fallback level, so that preview is an error, never a different picture
    with pytest.raises(Exception, match="bg.npy|scan_dir|background"):
        preview_frame(document, frame)


def test_summary_preview_is_the_sinks_summary_figure(scan):
    """preview_summary == the sink's averaged_image_grid.png / average_processed_visual.png."""
    document = load_analysis_document(RECIPE)
    frames = _frames(3)
    prepared = prepare_document(document, scan_folder=scan)
    measurements = [measure_frame(prepared, f) for f in frames]
    panels = tuple(Product(i + 1, m, float(i + 1)) for i, m in enumerate(measurements))
    from geecs_analysis.compat.v2_average import average_results

    average = Product(
        "average", average_results(measurements, prepared.recipe, mode="noscan")
    )
    saved = save_products(
        ProductPlan(singles=panels + (average,), summary=panels, position_label="shot"),
        scan_recipe(document),
        scan,
    )
    by_name = {p.name: p for p in saved.files}
    grid = preview_summary(
        document, frames, [1.0, 2.0, 3.0], "shot", 0, scan_folder=scan
    )
    assert _png(grid) == by_name["cam_averaged_image_grid.png"].read_bytes()
    avg = preview_summary(
        document, frames, [1.0, 2.0, 3.0], "shot", 1, scan_folder=scan
    )
    assert _png(avg) == by_name["cam_average_processed_visual.png"].read_bytes()
    with pytest.raises(LookupError, match="no index 2"):
        preview_summary(document, frames, [1.0, 2.0, 3.0], "shot", 2, scan_folder=scan)
    with pytest.raises(ValueError, match="one position per frame"):
        preview_summary(document, frames, [1.0], "shot", 0, scan_folder=scan)


def test_a_v2_document_the_core_does_not_serve_is_refused_not_drawn():
    """An unported operation (a flip) raises at compile time; the host keeps its route."""
    document = load_analysis_document(
        yaml.safe_load(
            """
schema_version: 2
name: cam
analyzer: {kind: beam}
image: {type: camera, transforms: {flip_horizontal: true}, pipeline: [transforms]}
"""
        )
    )
    with pytest.raises(UnsupportedRecipe):
        preview_frame(document, _frames(1)[0])
