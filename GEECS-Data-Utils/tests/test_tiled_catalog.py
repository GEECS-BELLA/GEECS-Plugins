"""Tests for the ScanCatalog seam (hermetic — fake client objects, no network)."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from geecs_data_utils.tiled_catalog import (
    CatalogStatus,
    StubCatalog,
    TiledScanCatalog,
    read_tiled_config,
    summary_from_metadata,
)

TEST_DAY = date(2026, 7, 12)


def _epoch(hour: int, minute: int, day: date = TEST_DAY) -> float:
    return datetime(day.year, day.month, day.day, hour, minute).timestamp()


def _start_doc(scan_number: int, hour: int = 9, minute: int = 0, **extra) -> dict:
    doc = {
        "uid": f"uid-{scan_number:03d}",
        "time": _epoch(hour, minute),
        "geecs_event_schema": 1,
        "plan_name": "geecs_step_scan",
        "motor": None,
        "num_points": 1,
        "shots_per_step": 10,
        "experiment": "Undulator",
        "scan_number": scan_number,
        "save_sets": ["Amp4In"],
    }
    doc.update(extra)
    return doc


# ----------------------------------------------------------------------
# Config reading
# ----------------------------------------------------------------------


class TestReadTiledConfig:
    def test_reads_uri_and_api_key(self, tmp_path):
        config = tmp_path / "config.ini"
        config.write_text(
            "[tiled]\nuri = http://tiled.example:8000\napi_key = sekrit\n"
        )
        assert read_tiled_config(config) == ("http://tiled.example:8000", "sekrit")

    def test_missing_file(self, tmp_path):
        assert read_tiled_config(tmp_path / "nope.ini") == (None, None)

    def test_missing_section(self, tmp_path):
        config = tmp_path / "config.ini"
        config.write_text("[Paths]\ngeecs_data = /data\n")
        assert read_tiled_config(config) == (None, None)

    def test_empty_values_are_none(self, tmp_path):
        config = tmp_path / "config.ini"
        config.write_text("[tiled]\nuri =\n")
        assert read_tiled_config(config) == (None, None)

    def test_from_config_classmethod(self, tmp_path):
        config = tmp_path / "config.ini"
        config.write_text("[tiled]\nuri = http://x:8000\n")
        catalog = TiledScanCatalog.from_config(config)
        assert catalog.uri == "http://x:8000"

    def test_from_config_unconfigured_is_honest(self, tmp_path):
        catalog = TiledScanCatalog.from_config(tmp_path / "nope.ini")
        status = catalog.probe()
        assert status.ok is False
        assert "not configured" in status.label


# ----------------------------------------------------------------------
# StubCatalog
# ----------------------------------------------------------------------


class TestStubCatalog:
    def test_probe_is_honestly_offline(self):
        status = StubCatalog().probe()
        assert status.ok is False
        assert "not connected" in status.label

    def test_lists_no_runs(self):
        assert StubCatalog().list_runs("Undulator", TEST_DAY) == []

    def test_load_refuses(self):
        with pytest.raises(KeyError):
            StubCatalog().load_run("uid-001")


# ----------------------------------------------------------------------
# summary_from_metadata
# ----------------------------------------------------------------------


class TestSummaryFromMetadata:
    def test_full_metadata(self):
        summary = summary_from_metadata(
            "uid-002",
            _start_doc(2, hour=9, minute=27),
            {"exit_status": "success", "time": _epoch(9, 28)},
        )
        assert summary.scan_number == 2
        assert summary.mode == "NOSCAN"
        assert summary.shots == 10
        assert summary.exit_status == "success"
        assert summary.experiment == "Undulator"
        assert summary.save_sets == ("Amp4In",)

    def test_missing_stop_doc(self):
        summary = summary_from_metadata("u", {"time": 1.0}, None)
        assert summary.exit_status is None
        assert summary.scan_number is None
        assert summary.shots is None

    def test_string_save_set_coerced(self):
        summary = summary_from_metadata("u", {"time": 1.0, "save_sets": "Solo"}, {})
        assert summary.save_sets == ("Solo",)

    def test_modes_classified(self):
        assert summary_from_metadata("u", _start_doc(1, motor="mono"), {}).mode == "1D"
        assert (
            summary_from_metadata("u", _start_doc(1, motor=["a", "b"]), {}).mode
            == "GRID"
        )
        assert (
            summary_from_metadata(
                "u", _start_doc(1, plan_name="geecs_adaptive_scan"), {}
            ).mode
            == "OPT"
        )

    def test_filter_text_haystack(self):
        summary = summary_from_metadata(
            "u",
            _start_doc(2, additional_description="alignment check"),
            {"exit_status": "abort"},
        )
        haystack = summary.filter_text()
        for needle in ("scan 002", "noscan", "abort", "alignment", "amp4in"):
            assert needle in haystack


# ----------------------------------------------------------------------
# TiledScanCatalog against a fake client
# ----------------------------------------------------------------------


class _FakeRun:
    """Metadata + optional primary stream, quacking like a Tiled BlueskyRun."""

    def __init__(self, start_doc, stop_doc=None, frame=None):
        self.metadata = {"start": start_doc, "stop": stop_doc or {}}
        self._frame = frame

    def __getitem__(self, key):
        if key != "primary" or self._frame is None:
            raise KeyError(key)
        frame = self._frame

        class _Readable:
            def read(self):
                class _DataSet:
                    def to_dataframe(self):
                        return frame

                return _DataSet()

        return _Readable()


class _FakeClient:
    """Chainable ``search`` + ``items`` + ``__getitem__`` fake."""

    def __init__(self, runs: dict):
        self._runs = runs
        self.search_queries: list = []

    def search(self, query):
        self.search_queries.append(query)
        # Apply start.time range / experiment equality queries structurally.
        from tiled.queries import Comparison, Eq

        filtered = {}
        for uid, run in self._runs.items():
            start = run.metadata["start"]
            keep = True
            if isinstance(query, Comparison):
                value = start.get(str(query.key).removeprefix("start."))
                operator = getattr(query.operator, "value", query.operator)
                if operator == "ge":
                    keep = value is not None and value >= query.value
                elif operator == "lt":
                    keep = value is not None and value < query.value
            elif isinstance(query, Eq):
                keep = start.get(str(query.key).removeprefix("start.")) == query.value
            if keep:
                filtered[uid] = run
        child = _FakeClient(filtered)
        child.search_queries = self.search_queries
        return child

    def items(self):
        return list(self._runs.items())

    def __getitem__(self, uid):
        return self._runs[uid]


def _fake_catalog(runs):
    catalog = TiledScanCatalog(uri="http://fake:8000")
    catalog._client = _FakeClient(runs)
    return catalog


class TestTiledScanCatalogListRuns:
    def test_day_listing_newest_first_metadata_only(self):
        pytest.importorskip("tiled")
        runs = {
            "uid-001": _FakeRun(
                _start_doc(1, hour=9, minute=11),
                {"exit_status": "success", "time": _epoch(9, 12)},
            ),
            "uid-002": _FakeRun(
                _start_doc(2, hour=9, minute=27),
                {"exit_status": "success", "time": _epoch(9, 28)},
            ),
            "uid-old": _FakeRun(
                _start_doc(9, hour=9, minute=0, time=_epoch(9, 0) - 86400.0),
                {"exit_status": "success"},
            ),
        }
        catalog = _fake_catalog(runs)
        summaries = catalog.list_runs("Undulator", TEST_DAY)
        assert [s.uid for s in summaries] == ["uid-002", "uid-001"]  # newest first
        assert all(s.exit_status == "success" for s in summaries)

    def test_experiment_filter_applied(self):
        pytest.importorskip("tiled")
        runs = {
            "uid-001": _FakeRun(_start_doc(1, hour=9), {}),
            "uid-002": _FakeRun(_start_doc(2, hour=10, experiment="Thomson"), {}),
        }
        catalog = _fake_catalog(runs)
        summaries = catalog.list_runs("Undulator", TEST_DAY)
        assert [s.uid for s in summaries] == ["uid-001"]

    def test_no_experiment_lists_everything_that_day(self):
        pytest.importorskip("tiled")
        runs = {
            "uid-001": _FakeRun(_start_doc(1, hour=9), {}),
            "uid-002": _FakeRun(_start_doc(2, hour=10, experiment="Thomson"), {}),
        }
        catalog = _fake_catalog(runs)
        assert len(catalog.list_runs("", TEST_DAY)) == 2


class TestTiledScanCatalogLoadRun:
    def test_load_run_reads_primary_frame(self):
        import pandas as pd

        frame = pd.DataFrame({"scan_event_index": [1, 2], "cam-counts": [0.1, 0.2]})
        runs = {
            "uid-002": _FakeRun(_start_doc(2), {"exit_status": "success"}, frame=frame)
        }
        catalog = _fake_catalog(runs)
        detail = catalog.load_run("uid-002")
        assert detail.summary.scan_number == 2
        assert detail.stop_doc["exit_status"] == "success"
        assert list(detail.data["cam-counts"]) == [0.1, 0.2]
        # reset_index applied — the sequence column is a real column.
        assert "scan_event_index" in detail.data.columns

    def test_load_run_without_primary_stream(self):
        runs = {"uid-002": _FakeRun(_start_doc(2), {"exit_status": "abort"})}
        catalog = _fake_catalog(runs)
        detail = catalog.load_run("uid-002")
        assert detail.data is None
        assert detail.summary.exit_status == "abort"


class TestProbe:
    def test_unconfigured(self):
        status = TiledScanCatalog().probe()
        assert status == CatalogStatus(ok=False, label="tiled: not configured")

    def test_unreachable_never_raises(self, monkeypatch):
        import requests

        def boom(*args, **kwargs):
            raise requests.ConnectionError("no route")

        monkeypatch.setattr(requests, "get", boom)
        status = TiledScanCatalog(uri="http://fake:8000").probe()
        assert status.ok is False
        assert "unreachable" in status.label

    def test_ok_on_2xx(self, monkeypatch):
        import requests

        class _Resp:
            status_code = 200

        monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
        status = TiledScanCatalog(uri="http://fake:8000/").probe()
        assert status.ok is True
        assert status.label == "tiled: fake:8000"

    def test_http_error_code_reported(self, monkeypatch):
        import requests

        class _Resp:
            status_code = 500

        monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
        status = TiledScanCatalog(uri="http://fake:8000").probe()
        assert status.ok is False
        assert "HTTP 500" in status.label

    def test_query_methods_raise_clearly_when_unconfigured(self):
        catalog = TiledScanCatalog()
        with pytest.raises(RuntimeError, match="No Tiled URI"):
            catalog.load_run("uid-001")
