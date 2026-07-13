"""Pin the browser's scan-folder resolution invariant: read-only, never creates.

The pure catalog/schema/drift layer lives in ``geecs_data_utils`` and is
tested there; what stays here is the browser's own B3 resolver
(``resolve_scan_folder``) and its repo-invariant pin — opening a scan
folder must never bring one into existence (analysis/GUI code is a
consumer of scan folders, never a producer).
"""

from __future__ import annotations

from geecs_data_utils.tiled_catalog import RunDetail, summary_from_metadata

from geecs_console.browser.browser_window import resolve_scan_folder
from fake_catalog import TEST_DAY, make_detail


def _tree_snapshot(root):
    """Every path under *root*, for before/after comparison."""
    return sorted(str(p) for p in root.rglob("*"))


class TestScanFolderResolutionInvariant:
    def test_existing_metadata_folder_resolves(self, tmp_path):
        scan_dir = tmp_path / "scans" / "Scan002"
        scan_dir.mkdir(parents=True)
        detail = make_detail(scan_folder=str(scan_dir))
        assert resolve_scan_folder(detail, TEST_DAY) == scan_dir

    def test_missing_folder_returns_none_and_touches_nothing(self, tmp_path):
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
