"""Tests for geecs_data_utils.data.dataset."""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from geecs_data_utils.data import DatasetBuilder, OutlierConfig


def _fake_scan(df: pd.DataFrame, number: int = 1):
    tag = SimpleNamespace(number=number, experiment="Undulator")
    paths = SimpleNamespace(tag=tag)
    return SimpleNamespace(data_frame=df, paths=paths)


def test_from_scan_applies_cleaning():
    df = pd.DataFrame({"x": [1.0, 2.0, 1000.0], "y": [1.0, -1.0, 1.0]})
    scan = _fake_scan(df)

    out = DatasetBuilder.from_scan(
        scan,
        outlier_config=OutlierConfig(method="nan", sigma=2.0),
        filters=[("y", ">", 0)],
        dropna=True,
    )

    assert out.rows_raw == 3
    assert out.rows_final <= 2
    assert "number" in out.scan_info


def test_from_scans_concatenates_and_records_metadata():
    a = _fake_scan(pd.DataFrame({"x": [1.0]}), number=1)
    b = _fake_scan(pd.DataFrame({"x": [2.0]}), number=2)

    out = DatasetBuilder.from_scans([a, b])
    assert out.rows_raw == 2
    assert out.rows_final == 2
    assert out.scan_info["total_scans"] == 2
    assert len(out.scan_info["scans"]) == 2


def test_from_scans_requires_at_least_one_loaded_frame():
    bad = SimpleNamespace(data_frame=None, paths=None)
    with pytest.raises(ValueError, match="No scans with loaded data_frame"):
        DatasetBuilder.from_scans([bad])


def test_load_scans_from_date_skips_missing():
    good = SimpleNamespace(
        data_frame=pd.DataFrame({"x": [1.0]}),
        paths=SimpleNamespace(tag=SimpleNamespace(number=1)),
    )

    def fake_from_date(*, number, **_kwargs):
        if number == 1:
            return good
        raise FileNotFoundError("no scan")

    with patch(
        "geecs_data_utils.scan_data.ScanData.from_date", side_effect=fake_from_date
    ):
        scans = DatasetBuilder.load_scans_from_date(
            year=2026,
            month=4,
            day=23,
            experiment="Undulator",
            numbers=[1, 2, 3],
        )
        report = DatasetBuilder.load_scans_from_date_report(
            year=2026,
            month=4,
            day=23,
            experiment="Undulator",
            numbers=[1, 2, 3],
        )
    assert len(scans) == 1
    assert scans[0] is good
    assert report.numbers_loaded == [1]
    assert len(report.skipped) == 2
    assert {n for n, _ in report.skipped} == {2, 3}


def test_load_scans_from_date_raise_on_missing():
    with patch(
        "geecs_data_utils.scan_data.ScanData.from_date",
        side_effect=FileNotFoundError("missing"),
    ):
        with pytest.raises(FileNotFoundError):
            DatasetBuilder.load_scans_from_date(
                year=2026,
                month=4,
                day=23,
                experiment="Undulator",
                numbers=[1],
                on_missing="raise",
            )


def test_from_date_scan_numbers_one_shot():
    good = SimpleNamespace(
        data_frame=pd.DataFrame({"x": [1.0]}),
        paths=SimpleNamespace(tag=SimpleNamespace(number=1)),
    )

    def fake_from_date(*, number, **_kwargs):
        if number == 1:
            return good
        raise FileNotFoundError("no scan")

    with patch(
        "geecs_data_utils.scan_data.ScanData.from_date", side_effect=fake_from_date
    ):
        out = DatasetBuilder.from_date_scan_numbers(
            year=2026,
            month=4,
            day=23,
            experiment="Undulator",
            numbers=[1, 2],
        )
    assert out.load_report is not None
    assert out.load_report.numbers_loaded == [1]
    assert len(out.frame) == 1
    assert out.scan_info["total_scans"] == 1


def test_load_scans_from_date_report_skips_empty_data_frame():
    empty_sd = SimpleNamespace(
        data_frame=None, paths=SimpleNamespace(tag=SimpleNamespace(number=5))
    )

    def fake_from_date(*, number, **_kwargs):
        return empty_sd

    with patch(
        "geecs_data_utils.scan_data.ScanData.from_date", side_effect=fake_from_date
    ):
        report = DatasetBuilder.load_scans_from_date_report(
            year=2026,
            month=4,
            day=23,
            experiment="Undulator",
            numbers=[5],
        )
    assert report.scans == []
    assert len(report.skipped) == 1
    assert report.skipped[0][0] == 5


def test_data_package_exports_load_scans_report():
    from geecs_data_utils.data import LoadScansReport

    r = LoadScansReport(scans=[], numbers_loaded=[], skipped=[(1, "test")])
    assert r.skipped[0][0] == 1


def test_from_date_scan_numbers_all_missing_raises():
    with patch(
        "geecs_data_utils.scan_data.ScanData.from_date",
        side_effect=FileNotFoundError("missing"),
    ):
        with pytest.raises(ValueError, match="No scans could be loaded"):
            DatasetBuilder.from_date_scan_numbers(
                year=2026,
                month=4,
                day=23,
                experiment="Undulator",
                numbers=[1, 2],
            )
