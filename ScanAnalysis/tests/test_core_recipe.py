"""Either document gives the scan host the same run facts; a recipe routes to the core alone."""

import subprocess
import sys

import numpy as np
import pytest
from geecs_analysis.measurement import Measurement
from geecs_analysis.recipe import RecipeError
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis import AnalysisRecipe

from scan_analysis.config.diagnostic_factory import create_scan_analyzer
from scan_analysis.core_analyzer import CoreScanAnalyzer
from scan_analysis.core_products import Product, ProductPlan
from scan_analysis.core_recipe import scan_recipe
from scan_analysis.core_sink import save_products


def recipe(**patch):
    return AnalysisRecipe.model_validate(
        {
            "device": "Device",
            "input": {"kind": "camera"},
            "measure": {"kind": "beam"},
            **patch,
        }
    )


def test_factory_routes_a_recipe_to_the_core():
    doc = recipe(summaries=[{"kind": "average"}])
    analyzer = create_scan_analyzer(doc, priority=None)
    assert isinstance(analyzer, CoreScanAnalyzer)
    assert analyzer.id == "Device" and analyzer.priority == 100
    assert analyzer.spec.summaries == tuple(doc.summaries)
    with pytest.raises(RecipeError, match="does not bind"):
        create_scan_analyzer(recipe(steps=[{"step": "sharpen"}]))


def product(identifier, value=1.0, position=None):
    frame = Frame.from_array(np.full((3, 3), value))
    return Product(identifier, Measurement({"x": 1.0}, frame), position)


@pytest.mark.parametrize(
    "summaries,expected",
    [
        ([], []),
        ([{"kind": "average"}], ["Device_average_processed_visual.png"]),
        ([{"kind": "image_grid", "columns": 1}], ["Device_averaged_image_grid.png"]),
        (
            [{"kind": "average"}, {"kind": "image_grid"}],
            ["Device_average_processed_visual.png", "Device_averaged_image_grid.png"],
        ),
    ],
)
def test_sink_draws_exactly_the_listed_summaries(tmp_path, summaries, expected):
    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    spec = scan_recipe(recipe(figure={"fig": {"dpi": 20}}, summaries=summaries))
    panels = tuple(product(i, i, float(i)) for i in (1, 2, 3))
    plan = ProductPlan(
        singles=(product("average"),), summary=panels, position_label="motor"
    )
    saved = save_products(plan, spec, scan)
    assert [p.name for p in saved.files] == ["Device_average_processed.h5", *expected]
    assert [p.name for p in saved.display_files] == expected


def test_sink_skips_a_kind_whose_product_this_run_lacks(tmp_path):
    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    spec = scan_recipe(
        recipe(
            figure={"fig": {"dpi": 20}},
            summaries=[{"kind": "image_grid"}, {"kind": "average"}],
        )
    )
    # a scanned run: bin singles and panels, no average
    panels = tuple(product(i, i, float(i)) for i in (1, 2))
    saved = save_products(
        ProductPlan(singles=panels, summary=panels, position_label="m"), spec, scan
    )
    assert [p.name for p in saved.files] == [
        "Device_1_processed.h5",
        "Device_1_processed_visual.png",
        "Device_2_processed.h5",
        "Device_2_processed_visual.png",
        "Device_averaged_image_grid.png",
    ]
    assert not saved.notes


def test_a_canonical_recipe_written_to_the_tree_loads_and_binds(tmp_path):
    """The written form (canonical YAML) reads back as the same recipe and binds to the core."""
    import yaml
    from geecs_analysis.recipe import binding_errors
    from geecs_schemas.analysis import canonical_document

    from scan_analysis.config import load_diagnostic

    root = tmp_path / "scan_analysis_configs"
    (root / "analyzers" / "HTU").mkdir(parents=True)
    (root / "groups").mkdir()
    v3 = recipe(
        output_name="Output",
        scalar_suffix="_s",
        steps=[{"step": "roi", "bounds": [[0, 2], [1, 3]]}],
        figure={"imshow": {"vmin": 0}, "fig": {"dpi": 30}},
        summaries=[
            {"kind": "image_grid", "panel_size": [6.0, 6.0]},
            {"kind": "average"},
        ],
    )
    canonical = canonical_document(v3)
    assert canonical["schema_version"] == 3 and binding_errors(v3) == []
    (root / "analyzers" / "HTU" / "New.yaml").write_text(
        yaml.safe_dump(canonical, sort_keys=False)
    )
    loaded = load_diagnostic("New", config_dir=root)
    assert loaded.model_dump() == v3.model_dump() and loaded.measure.kind == "beam"


def test_sink_resolves_every_kind_in_a_fresh_process():
    """The sink's imports alone register the kinds; no v2 document need load first."""
    script = (
        "from scan_analysis.core_sink import save_products  # noqa: F401\n"
        "from geecs_analysis.registry import summary_definitions\n"
        "print(sorted(d.filename for d in summary_definitions()))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == (
        "['average_processed_visual', 'averaged_image_grid', 'summary_scalar_fit', "
        "'summary_waterfall']"
    )
