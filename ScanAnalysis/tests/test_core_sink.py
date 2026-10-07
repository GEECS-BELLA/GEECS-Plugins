"""Core sinks retain output contracts without ever creating raw scan folders."""

import h5py
import numpy as np
import pytest
from geecs_analysis.measurement import Measurement
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis import AnalysisDiagnostic

from scan_analysis.analyzers.renderers.config import parse_output_filename
from scan_analysis.core_products import Product, ProductPlan
from scan_analysis.core_recipe import scan_recipe
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
    result = save_products(ProductPlan(singles=(entry,)), scan_recipe(doc), scan)
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
    plan = ProductPlan(summary=panels, position_label="motor")
    saved = save_products(plan, scan_recipe(doc), scan)
    expected = (
        "Device_summary_waterfall.png" if line else "Device_averaged_image_grid.png"
    )
    assert [p.name for p in saved.files] == [expected]
    assert saved.display_files == saved.files
    assert not saved.notes


def test_grid_carries_the_scan_parameter_label(tmp_path, monkeypatch):
    from dataclasses import replace

    from scan_analysis import core_sink

    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    seen = {}
    real = core_sink.summary_definition

    def recording(options):
        definition = real(options)

        def function(results, positions, label, *rest):
            seen[definition.filename] = label
            return definition.function(results, positions, label, *rest)

        return replace(definition, function=function)

    monkeypatch.setattr(core_sink, "summary_definition", recording)
    doc = document()
    panels = tuple(Product(i, product().measurement, float(i)) for i in [1, 2, 3])
    plan = ProductPlan(summary=panels, position_label="motor")
    save_products(plan, scan_recipe(doc), scan)
    assert seen == {"averaged_image_grid": "motor"}


def test_bin_products_round_trip_through_the_filename_parser(tmp_path):
    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    doc = document()
    saved = save_products(
        ProductPlan(singles=(product(identifier=3),)), scan_recipe(doc), scan
    )
    assert [parse_output_filename(p.name) for p in saved.files] == [("bin", 3)] * 2


def test_empty_output_name_falls_back_to_the_device_directory(tmp_path):
    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    doc = document(output_name="")
    saved = save_products(ProductPlan(singles=(product(),)), scan_recipe(doc), scan)
    assert (
        saved.files[0].parent
        == tmp_path / "analysis" / "Scan001" / "Device" / "Array2DScanAnalyzer"
    )


def test_save_disabled_and_empty_plan_do_not_touch_missing_scan(tmp_path):
    doc = document(scan={"save": False})
    plan = ProductPlan(singles=(product(),))
    assert not save_products(plan, scan_recipe(doc), tmp_path / "missing").files
    doc.scan.save = True
    assert not save_products(
        ProductPlan(), scan_recipe(doc), tmp_path / "missing"
    ).files
    assert not list(tmp_path.iterdir())


def test_missing_raw_folder_is_never_created(tmp_path):
    doc = document()
    with pytest.raises(FileNotFoundError):
        save_products(
            ProductPlan(singles=(product(),)),
            scan_recipe(doc),
            tmp_path / "scans" / "Scan001",
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["name", "output_name"])
def test_traversal_refused_before_output_creation(tmp_path, field):
    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    doc = document(**{field: "../escape"})
    with pytest.raises(ValueError, match="component"):
        save_products(ProductPlan(singles=(product(),)), scan_recipe(doc), scan)
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
        save_products(ProductPlan(singles=(product(),)), scan_recipe(doc), scan)
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
    )
    saved = save_products(plan, scan_recipe(doc), scan)
    assert len(saved.files) == 2
    assert not saved.display_files
    assert "equal-length" in saved.notes[0]


def fit_recipe(*summaries):
    from geecs_schemas.analysis import AnalysisRecipe

    return AnalysisRecipe.model_validate(
        {
            "device": "Device",
            "output_name": "Output",
            "input": {"kind": "camera"},
            "figure": {"fig": {"dpi": 30}},
            "summaries": list(summaries),
        }
    )


def test_scalar_fit_writes_its_png_and_a_json_sidecar(tmp_path):
    import json

    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    doc = fit_recipe(
        {"kind": "image_grid"},
        {"kind": "scalar_fit", "scalars": ["kick_1", "kick_2"]},
    )
    image = Frame.from_array(np.ones((3, 3)))
    panels = tuple(
        Product(i, Measurement({"kick_1": 2.0 * p - 1.0}, image), p)
        for i, p in enumerate([0.0, 1.0, 2.0], start=1)
    )
    plan = ProductPlan(summary=panels, position_label="hexapod x")
    saved = save_products(plan, scan_recipe(doc), scan)
    names = [p.name for p in saved.files]
    assert names == [
        "Device_averaged_image_grid.png",
        "Device_summary_scalar_fit.png",
        "Device_summary_scalar_fit.json",
    ]
    assert saved.files[1].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    # the sidecar is data, not a display figure
    assert saved.display_files == saved.files[:2]
    text = saved.files[2].read_text()
    assert "NaN" not in text and "Infinity" not in text
    sidecar = json.loads(text)
    assert sidecar["kind"] == "scalar_fit"
    assert sidecar["position_label"] == "hexapod x"
    scalars = sidecar["scalars"]
    assert scalars["kick_1_slope"] == pytest.approx(2.0)
    assert scalars["kick_1_zero_crossing"] == pytest.approx(0.5)
    assert scalars["kick_1_points"] == 3
    # an absent key: every number null, the reason in the notes
    assert scalars["kick_2_slope"] is None and scalars["kick_2_points"] == 0
    assert "kick_2: missing from 3 of 3 results" in sidecar["notes"]
    assert any("kick_2: missing" in n for n in saved.notes)
    assert all(parse_output_filename(name) == ("summary", None) for name in names[1:])
    assert not list(scan.iterdir())


def test_figure_only_kinds_write_no_sidecar(tmp_path):
    scan = tmp_path / "scans" / "Scan001"
    scan.mkdir(parents=True)
    doc = fit_recipe({"kind": "image_grid"})
    panels = tuple(Product(i, product().measurement, float(i)) for i in [1, 2])
    saved = save_products(ProductPlan(summary=panels), scan_recipe(doc), scan)
    assert [p.name for p in saved.files] == ["Device_averaged_image_grid.png"]
    target = saved.files[0].parent
    assert sorted(p.name for p in target.iterdir()) == [
        "Device_averaged_image_grid.png"
    ]


class TestShotStore:
    """One HDF5 per scan of every single-shot measurement's frame and extras."""

    @staticmethod
    def measurement(shot: int, *, extras=("a", "b"), shape=(2, 3)):
        frame = Frame.from_array(np.full(shape, shot, dtype=np.float64))
        return Measurement(
            {"s": shot},
            frame,
            extras={
                k: Frame.from_array(np.full(shape, shot * 10 + i))
                for i, k in enumerate(extras)
            },
        )

    def test_shots_append_in_order_and_the_file_appears_on_close(self, tmp_path):
        from scan_analysis.core_sink import ShotStore, shot_store_path

        scan = tmp_path / "scans" / "Scan001"
        scan.mkdir(parents=True)
        path = shot_store_path(scan_recipe(document()), scan, "wavefront")
        assert path == (
            tmp_path
            / "analysis"
            / "Scan001"
            / "Output"
            / "Array2DScanAnalyzer"
            / "Device_wavefront.h5"
        )
        store = ShotStore(path)
        assert store.close() is None and not path.parent.exists()  # nothing stored
        store = ShotStore(path)
        for shot in (3, 1, 2):
            store.add(shot, self.measurement(shot))
        assert store.part.exists() and not path.exists()
        assert store.close() == path
        assert not store.part.exists()
        with h5py.File(path) as f:
            assert f["shots"][:].tolist() == [3, 1, 2]
            assert f["frame"].dtype == np.float32 and f["frame"].shape == (3, 2, 3)
            assert f["frame"].chunks == (1, 2, 3)
            assert sorted(f["extras"]) == ["a", "b"]
            np.testing.assert_array_equal(f["frame"][1], np.full((2, 3), 1))
            np.testing.assert_array_equal(f["extras/b"][0], np.full((2, 3), 31))
        assert not list(scan.iterdir())

    def test_a_disagreeing_shot_is_refused_and_a_discarded_store_leaves_nothing(
        self, tmp_path
    ):
        from scan_analysis.core_sink import ShotStore

        path = (
            tmp_path
            / "analysis"
            / "Scan001"
            / "Output"
            / "Array2DScanAnalyzer"
            / "D_w.h5"
        )
        store = ShotStore(path)
        store.add(1, self.measurement(1))
        with pytest.raises(ValueError, match="shape"):
            store.add(2, self.measurement(2, shape=(3, 3)))
        with pytest.raises(ValueError, match="extras"):
            store.add(2, self.measurement(2, extras=("a",)))
        assert store.close(keep=False) is None
        assert not path.exists() and not store.part.exists()
        with pytest.raises(RuntimeError):
            with ShotStore(path) as store:
                store.add(1, self.measurement(1))
                raise RuntimeError("the run died")
        assert not path.exists() and not store.part.exists()

    def test_a_second_writer_on_the_same_part_is_refused(self, tmp_path):
        from scan_analysis.core_sink import ShotStore

        path = (
            tmp_path
            / "analysis"
            / "Scan001"
            / "Output"
            / "Array2DScanAnalyzer"
            / "D_w.h5"
        )
        first = ShotStore(path)
        first.add(1, self.measurement(1))
        second = ShotStore(path)
        with pytest.raises(OSError, match="exists"):
            second.add(1, self.measurement(1))
        # The refused writer discards nothing of the first's.
        assert second.close(keep=False) is None and first.part.exists()
        assert first.close() == path

    def test_the_store_name_is_one_component(self, tmp_path):
        from scan_analysis.core_sink import shot_store_path

        scan = tmp_path / "scans" / "Scan001"
        scan.mkdir(parents=True)
        with pytest.raises(ValueError):
            shot_store_path(scan_recipe(document()), scan, "../w")
