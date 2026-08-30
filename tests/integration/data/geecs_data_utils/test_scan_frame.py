"""Integration tests for the s-file reader and the union frame.

Requires network-mounted GEECS data (and, for the union test, the lab
Tiled server). Run with:
    pytest -m "integration and data" tests/integration/data/geecs_data_utils/
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.data]


def test_read_sfile_canonical(canonical_scan_tag):
    """The shared reader parses a real s-file with verbatim headers."""
    from geecs_data_utils import ScanPaths
    from geecs_data_utils.data.sfile import read_sfile, sfile_path_for_scan

    tag = canonical_scan_tag("undulator_2d")
    folder = ScanPaths(tag=tag).get_folder()
    frame = read_sfile(sfile_path_for_scan(folder))
    assert len(frame) > 0
    assert "Bin #" in frame.columns
    assert "Shotnumber" in frame.columns


def test_scan_frame_unions_both_providers(canonical_scan_tag):
    """End to end over the real catalog + share: run doc ∪ s-file.

    The full wave-1 substrate path: Tiled load_run → resolve_scan_folder
    → scan_frame, asserting both provenances present and the 1-based
    join aligned (Shotnumber == scan_event_index wherever both exist).
    """
    pytest.importorskip("tiled", reason="needs the data-utils 'tiled' extra")
    from datetime import date

    from geecs_data_utils.scan_frame import (
        PROVENANCE_RUN,
        PROVENANCE_SFILE,
        scan_frame,
    )
    from geecs_data_utils.tiled_catalog import (
        TiledScanCatalog,
        resolve_scan_folder,
    )

    tag = canonical_scan_tag("undulator_bluesky_1d")
    catalog = TiledScanCatalog.from_config()
    if not catalog.probe().ok:
        pytest.skip("Tiled server unreachable")
    day = date(tag.year, tag.month, tag.day)
    runs = [
        r for r in catalog.list_runs(tag.experiment, day) if r.scan_number == tag.number
    ]
    assert runs, f"canonical Bluesky scan {tag} not in the catalog"
    detail = catalog.load_run(runs[0].uid)
    folder = resolve_scan_folder(detail, day)
    assert folder is not None, "scan folder not resolvable on this host"

    result = scan_frame(detail, folder)
    tags = set(result.provenance.values())
    assert PROVENANCE_RUN in tags and PROVENANCE_SFILE in tags

    frame = result.frame
    both = frame.dropna(subset=["Shotnumber", "scan_event_index"])
    assert len(both) > 0
    assert (
        both["Shotnumber"].astype(int) == both["scan_event_index"].astype(int)
    ).all()
