"""Core sinks retain output contracts without ever creating raw scan folders."""

import h5py
import numpy as np
import pytest
from geecs_analysis.compat.v2 import compile_v2
from geecs_analysis.measurement import Measurement
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis import AnalysisDiagnostic

from scan_analysis.analyzers.renderers.config import parse_output_filename
from scan_analysis.core_products import Product, ProductPlan
from scan_analysis.core_sink import analysis_directory, save_products


def document(line=False, **kwargs):
    return AnalysisDiagnostic.model_validate(
        {
            "name": "Device",
            "output_name": "Output",
            "analyzer": {"kind": "line" if line else "beam"},
            "image": {
                "type": "line",
                "data_loading": {"data_type": "npy"},
                "storage_dtype": "float32",
            }
            if line
            else {"type": "camera"},
            "scan": {"renderer": {"dpi": 30}},
            **kwargs,
        }
    )


def product(line=False, identifier="average"):
    frame = (
        Frame.from_trace([[1, 0.1], [2, 0.2], [3, 0.3]])
        if line
        else Frame.from_array(np.arange(9).reshape(3, 3))
    )
    return Product(identifier, Measurement({"scalar": 4}, frame))


@pytest.mark.parametrize("line", [False, True])
def test_saved_data_schema_dtype_names_and_logical_vs_output_identity(tmp_path, line):
    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    doc = document(line)
    entry = product(line)
    result = save_products(ProductPlan(singles=(entry,)), compile_v2(doc), doc, scan)
    assert [p.name for p in result.files] == [
        "Device_average_processed.h5",
        "Device_average_processed_visual.png",
    ]
    assert result.files[0].parent == tmp_path / "analysis" / "Scan001" / "Output" / (
        "Array1DScanAnalyzer" if line else "Array2DScanAnalyzer"
    )
    assert all(parse_output_filename(p.name) == ("summary", None) for p in result.files)
    with h5py.File(result.files[0]) as handle:
        key = "data" if line else "image"
        assert list(handle) == [key]
        ds = handle[key]
        assert not dict(handle.attrs) and not dict(ds.attrs)
        assert ds.compression == "gzip" and ds.compression_opts == 4
        expected = (
            entry.measurement.frame.as_trace().astype("float32")
            if line
            else entry.measurement.frame.data
        )
        assert ds.dtype == expected.dtype
        np.testing.assert_array_equal(ds[:], expected)
    assert result.files[1].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result.display_files == (() if line else (result.files[1],))
    assert not list(scan.iterdir())


@pytest.mark.parametrize("line", [False, True])
def test_summary_filename_and_display_contract(tmp_path, line):
    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    doc = document(line)
    panels = tuple(Product(i, product(line).measurement, i * 2.0) for i in [1, 2, 3])
    plan = ProductPlan(
        summary=panels,
        summary_kind="waterfall" if line else "image_grid",
        position_label="motor",
    )
    saved = save_products(plan, compile_v2(doc), doc, scan)
    expected = (
        "Device_summary_waterfall.png" if line else "Device_averaged_image_grid.png"
    )
    assert [p.name for p in saved.files] == [expected]
    assert saved.display_files == saved.files
    assert not saved.notes


def test_save_disabled_and_empty_plan_do_not_touch_missing_scan(tmp_path):
    doc = document(scan={"save": False})
    plan = ProductPlan(singles=(product(),))
    assert not save_products(plan, compile_v2(doc), doc, tmp_path / "missing").files
    doc.scan.save = True
    assert not save_products(
        ProductPlan(), compile_v2(doc), doc, tmp_path / "missing"
    ).files
    assert not list(tmp_path.iterdir())


def test_missing_raw_folder_is_never_created(tmp_path):
    doc = document()
    with pytest.raises(FileNotFoundError):
        save_products(
            ProductPlan(singles=(product(),)),
            compile_v2(doc),
            doc,
            tmp_path / "scans" / "Scan001",
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["name", "output_name"])
def test_traversal_refused_before_output_creation(tmp_path, field):
    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    doc = document(**{field: "../escape"})
    with pytest.raises(ValueError, match="component"):
        save_products(ProductPlan(singles=(product(),)), compile_v2(doc), doc, scan)
    assert not (tmp_path / "analysis").exists()


def test_analysis_symlink_cannot_write_back_into_raw_tree(tmp_path):
    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    (tmp_path / "analysis").symlink_to(scan.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="raw scans"):
        analysis_directory(scan)
    assert not list(scan.iterdir())


def test_existing_output_symlink_cannot_overwrite_raw_file(tmp_path):
    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    raw = scan / "raw.h5"
    raw.write_bytes(b"untouched")
    target = tmp_path / "analysis" / "Scan001" / "Output" / "Array2DScanAnalyzer"
    target.mkdir(parents=True)
    (target / "Device_average_processed.h5").symlink_to(raw)
    doc = document()
    with pytest.raises(ValueError, match="escapes"):
        save_products(ProductPlan(singles=(product(),)), compile_v2(doc), doc, scan)
    assert raw.read_bytes() == b"untouched"


def test_bad_waterfall_skips_only_summary_and_reports_reason(tmp_path):
    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    doc = document(True)
    entry = product(True)
    short = Product(2, Measurement({}, Frame.from_trace([[1, 2]])), 2)
    plan = ProductPlan(
        singles=(entry,),
        summary=(Product(1, entry.measurement, 1), short),
        summary_kind="waterfall",
    )
    saved = save_products(plan, compile_v2(doc), doc, scan)
    assert len(saved.files) == 2
    assert not saved.display_files
    assert "equal-length" in saved.notes[0]
