"""Either document gives the scan host the same run facts; a recipe routes to the core alone."""

import subprocess
import sys

import numpy as np
import pytest
import yaml
from geecs_analysis.compat.convert import to_v3
from geecs_analysis.measurement import Measurement
from geecs_analysis.recipe import RecipeError
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis import AnalysisDiagnostic, AnalysisRecipe

from scan_analysis.config.diagnostic_factory import create_scan_analyzer
from scan_analysis.config_store import ConfigStore
from scan_analysis.core_analyzer import CoreScanAnalyzer, core_supports
from scan_analysis.core_products import Product, ProductPlan
from scan_analysis.core_recipe import scan_recipe
from scan_analysis.core_sink import save_products


def diagnostic(kind="beam", **scan):
    image = (
        {
            "type": "line",
            "data_loading": {"data_type": "npy"},
            "storage_dtype": "float32",
        }
        if kind == "line"
        else {
            "type": "camera",
            "pipeline": ["roi"],
            "roi": {"x_min": 1, "x_max": 3, "y_min": 0, "y_max": 2},
        }
    )
    return AnalysisDiagnostic.model_validate(
        {
            "name": "Device",
            "output_name": "Output",
            "metric_suffix": "_s",
            "analyzer": {"kind": kind},
            "image": image,
            "scan": {"renderer": {"dpi": 30}, **scan},
        }
    )


def recipe(**patch):
    return AnalysisRecipe.model_validate(
        {
            "device": "Device",
            "input": {"kind": "camera"},
            "measure": {"kind": "beam"},
            **patch,
        }
    )


@pytest.mark.parametrize(
    "doc",
    [
        diagnostic("beam", mode="per_bin", priority=2, save=False, file_tail=".tif"),
        diagnostic("line", device="Device-x", data_format="per_shot_files"),
    ],
    ids=["camera", "line"],
)
def test_converted_recipe_gives_the_same_run_facts(doc):
    v2 = scan_recipe(doc)
    v3 = scan_recipe(to_v3(doc).recipe)
    for field in (
        "recipe",
        "device",
        "folder",
        "output_name",
        "scalar_suffix",
        "file_tail",
        "prefer_stack",
        "line_loading_json",
        "average_frames_first",
        "save",
        "priority",
    ):
        assert getattr(v2, field) == getattr(v3, field), field
    assert [s.kind for s in v2.summaries] == [s.kind for s in v3.summaries]
    assert v2.line == v3.line


def test_factory_routes_a_recipe_to_the_core_only():
    doc = recipe(summaries=[{"kind": "average"}])
    assert core_supports(doc)
    analyzer = create_scan_analyzer(doc, priority=None)
    assert isinstance(analyzer, CoreScanAnalyzer)
    assert analyzer.id == "Device" and analyzer.priority == 100
    assert analyzer.spec.summaries == tuple(doc.summaries)
    with pytest.raises(ValueError, match="no legacy route"):
        create_scan_analyzer(doc, route="legacy")
    with pytest.raises(ValueError, match="injected"):
        create_scan_analyzer(doc, use_injected_data=True)
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


def test_store_lists_validates_and_writes_a_recipe(tmp_path):
    root = tmp_path / "scan_analysis_configs"
    (root / "analyzers" / "HTU").mkdir(parents=True)
    (root / "groups").mkdir()
    v2 = diagnostic("beam")
    v3 = to_v3(v2).recipe
    (root / "analyzers" / "HTU" / "Old.yaml").write_text(
        yaml.safe_dump(v2.model_dump(mode="json", exclude_none=True))
    )
    store = ConfigStore(root)
    report = store.validate("analyzer", v3.model_dump(mode="json", exclude_none=True))
    assert report.ok and report.canonical["schema_version"] == 3
    (root / "analyzers" / "HTU" / "New.yaml").write_text(report.yaml)
    entries = {e.id: e for e in store.list("analyzer")}
    assert entries["New"].valid and entries["New"].summary["schema_version"] == 3
    assert entries["New"].summary["analyzer_kind"] == "beam"
    assert entries["Old"].valid and entries["Old"].summary["schema_version"] == 2
    loaded = store.read("analyzer", "New")
    assert loaded.valid and loaded.document["schema_version"] == 3


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
        "['average_processed_visual', 'averaged_image_grid', 'summary_waterfall']"
    )


def test_legacy_image_analyzer_entry_points_refuse_a_recipe():
    from image_analysis.config import create_image_analyzer
    from image_analysis.ephemeral import _ephemeral_analyzer_for

    with pytest.raises(TypeError, match="analysis core"):
        create_image_analyzer(recipe())
    with pytest.raises(TypeError, match="no ephemeral"):
        _ephemeral_analyzer_for(recipe())
