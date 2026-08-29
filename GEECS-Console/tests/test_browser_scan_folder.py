"""Pin the browser's scan-folder resolution invariant: read-only, never creates.

``resolve_scan_folder`` moved down to ``geecs_data_utils.tiled_catalog``
(portal arc phase 2) where its core invariant tests now live; what stays
here is the browser-path pin — the window resolves through the shared
implementation (import identity) and opening a scan folder through the
browser import path must never bring one into existence (analysis/GUI
code is a consumer of scan folders, never a producer).
"""

from __future__ import annotations

from geecs_data_utils import tiled_catalog
from geecs_data_utils.tiled_catalog import RunDetail, summary_from_metadata

from geecs_console.browser.browser_window import resolve_scan_folder
from fake_catalog import TEST_DAY, make_detail


def test_resolver_is_the_shared_data_utils_implementation():
    """The browser must not grow a shadowing resolver of its own."""
    assert resolve_scan_folder is tiled_catalog.resolve_scan_folder


def _tree_snapshot(root):
    """Every path under *root*, for before/after comparison."""
    return sorted(str(p) for p in root.rglob("*"))


class TestScanFolderResolutionInvariant:
    def test_existing_metadata_folder_resolves(self, tmp_path):
        scan_dir = tmp_path / "scans" / "Scan002"
        scan_dir.mkdir(parents=True)
        detail = make_detail(scan_folder=str(scan_dir))
        assert resolve_scan_folder(detail, TEST_DAY) == scan_dir

    def test_missing_folder_returns_none_and_touches_nothing(
        self, tmp_path, monkeypatch
    ):
        from geecs_data_utils import scan_paths as scan_paths_mod

        # Hermetic: a stale metadata path now falls through to the daily
        # fallback (host-specific-mount re-basing), which must not reach
        # the real config.ini data root in tests.
        monkeypatch.setattr(scan_paths_mod, "daily_scan_folder", lambda *a, **k: None)
        day_root = tmp_path / "data"
        day_root.mkdir()
        missing = day_root / "Y2026" / "07-Jul" / "26_0712" / "scans" / "Scan002"
        detail = make_detail(scan_folder=str(missing))
        before = _tree_snapshot(tmp_path)
        assert resolve_scan_folder(detail, TEST_DAY) is None
        assert _tree_snapshot(tmp_path) == before  # tree untouched

    def test_no_scan_number_and_no_metadata_returns_none(self, tmp_path):
        # A never-claimed run: no scan_folder key and no scan_number.
        start = dict(make_detail().start_doc)
        start.pop("scan_number")
        start.pop("scan_id")
        start.pop("scan_folder", None)
        rebuilt = RunDetail(
            summary=summary_from_metadata("u", start, {}),
            start_doc=start,
            stop_doc={},
            data=None,
        )
        before = _tree_snapshot(tmp_path)
        assert resolve_scan_folder(rebuilt, TEST_DAY) is None
        assert _tree_snapshot(tmp_path) == before
