"""Hermetic tests for the analysis-domain read tools (#675).

A tmp data-share tree stands in for the netapp (the `_base_directory`
seam points the pure ScanPaths builders at it); figures are real PNGs
generated with pillow.  The read-only discipline is pinned: the tools
must never create anything on the share.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml

from geecs_mcp import runtime
from geecs_mcp.analysis import read_tools


@pytest.fixture(autouse=True)
def _fresh_runtime(monkeypatch):
    runtime.clear_runtime_cache()
    monkeypatch.setattr(runtime, "get_experiment", lambda: "TestExp")
    yield
    runtime.clear_runtime_cache()


def _load(payload) -> dict:
    return json.loads(payload)


DAY = "2026-08-22"


def _tree_paths(base: Path) -> tuple[Path, Path]:
    date_dir = base / "TestExp" / "Y2026" / "08-Aug" / "26_0822"
    return date_dir / "scans" / "Scan007", date_dir / "analysis" / "Scan007"


@pytest.fixture
def share(tmp_path, monkeypatch):
    """A populated fake share: scan folder + statuses + analysis outputs."""
    scan_folder, analysis_folder = _tree_paths(tmp_path)
    status_dir = scan_folder / "analysis_status"
    status_dir.mkdir(parents=True)
    out_dir = analysis_folder / "UC_TopView"
    out_dir.mkdir(parents=True)

    from PIL import Image as PILImage

    image = PILImage.new("RGB", (2000, 500), color=(10, 200, 30))
    image.save(out_dir / "summary.png")
    (out_dir / "results.csv").write_text("a,b\n1,2\n")

    # The REAL TaskStatus.to_dict() shape (review finding: the first
    # fixtures pinned a stale-doc schema the writer never produces —
    # these keys/types are verified against ScanAnalysis task_queue.py).
    (status_dir / "topview_baseline.yaml").write_text(
        yaml.safe_dump(
            {
                "analyzer_id": "topview_baseline",
                "priority": 0,
                "state": "done",
                "error": None,
                "claimed_by": "runner-1",
                "claimed_at": "2026-08-22T10:00:00+00:00",
                "last_heartbeat": "2026-08-22T10:05:00+00:00",
                "display_files": [str(out_dir / "summary.png")],
            }
        )
    )
    (status_dir / "magspec.yaml").write_text(
        yaml.safe_dump({"analyzer_id": "magspec", "state": "queued"})
    )
    (status_dir / "broken_analyzer.yaml").write_text(
        yaml.safe_dump({"state": "failed", "error": "no s-file columns for UC_MagSpec"})
    )
    monkeypatch.setattr(read_tools, "_base_directory", lambda: tmp_path)
    return tmp_path


def _mtimes(root: Path) -> set:
    return {(p, p.stat().st_mtime_ns) for p in root.rglob("*")}


# ---------------------------------------------------------------------------
# get_scan_analysis
# ---------------------------------------------------------------------------


def test_analysis_reports_tasks_and_outputs(share):
    result = _load(read_tools._get_scan_analysis_impl(7, DAY))
    assert result["ok"] and result["analysis_present"]
    done = result["tasks"]["topview_baseline"]
    assert done["state"] == "done"
    assert done["display_files"]
    assert done["heartbeat_age_s"] is not None  # ISO string parsed to an age
    assert result["tasks"]["magspec"] == {
        "state": "queued",
        "error": None,
        "claimed_by": None,
        "heartbeat_age_s": None,
        "display_files": [],
    }
    # The single most useful failed-task field is surfaced (review finding).
    assert "s-file columns" in result["tasks"]["broken_analyzer"]["error"]
    assert result["outputs"]["UC_TopView"]["n_files"] == 2
    assert "summary.png" in result["outputs"]["UC_TopView"]["files"]


def test_analysis_missing_scan_is_not_found_and_creates_nothing(share):
    before = _mtimes(share)
    result = _load(read_tools._get_scan_analysis_impl(99, DAY))
    assert not result["ok"] and result["error_kind"] == "not_found"
    assert "Scan099" in result["message"] or "Scan" in result["message"]
    # THE read-only pin: a miss must not plant anything on the share
    # (the get_analysis_folder() instance accessor would have mkdir'd —
    # this asserts we never touch it).
    assert _mtimes(share) == before


def test_analysis_survives_torn_status_yaml(share):
    scan_folder, _ = _tree_paths(share)
    (scan_folder / "analysis_status" / "torn.yaml").write_text("{ not: [valid")
    result = _load(read_tools._get_scan_analysis_impl(7, DAY))
    assert result["ok"]
    assert result["tasks"]["torn"]["state"] == "unreadable"


def test_analysis_survives_odd_field_types(share):
    # Review finding 2: the tolerant parse must be tolerant AFTER the
    # yaml load too — odd field types degrade the ENTRY, never the tool.
    scan_folder, _ = _tree_paths(share)
    (scan_folder / "analysis_status" / "odd.yaml").write_text(
        yaml.safe_dump(
            {
                "state": "claimed",
                "last_heartbeat": ["not", "a", "string"],
                "display_files": "not-a-list.png",
            }
        )
    )
    result = _load(read_tools._get_scan_analysis_impl(7, DAY))
    assert result["ok"]
    odd = result["tasks"]["odd"]
    assert odd["state"] == "claimed"
    assert odd["heartbeat_age_s"] is None
    assert odd["display_files"] == []  # a scalar is not a file list


def test_figure_candidates_bounded_to_the_share(share, tmp_path_factory):
    # Review finding 3: a display_files entry escaping the share must be
    # dropped, not served (confused-deputy bounding).
    outside = tmp_path_factory.mktemp("outside") / "secret.png"
    from PIL import Image as PILImage

    PILImage.new("RGB", (10, 10)).save(outside)
    scan_folder, _ = _tree_paths(share)
    (scan_folder / "analysis_status" / "escape.yaml").write_text(
        yaml.safe_dump({"state": "done", "display_files": [str(outside)]})
    )
    result = _load(read_tools._get_scan_figure_impl(7, "secret", DAY))
    assert not result["ok"] and result["error_kind"] == "not_found"


def test_figure_decode_cap_refuses_giant_images(share, monkeypatch):
    # The cap raises in the impl; at the tool layer the shared guard turns
    # it into an internal_error envelope (pinned separately) — here we pin
    # that the cap actually fires before any full decode.
    monkeypatch.setattr(read_tools, "_MAX_FIGURE_PIXELS", 100)
    with pytest.raises(ValueError, match="decode cap"):
        read_tools._get_scan_figure_impl(7, "summary", DAY)


def test_analysis_bad_day_is_invalid_request(share):
    result = _load(read_tools._get_scan_analysis_impl(7, "yesterday"))
    assert not result["ok"] and result["error_kind"] == "invalid_request"


def test_analysis_without_experiment_is_invalid_request(share, monkeypatch):
    monkeypatch.setattr(runtime, "get_experiment", lambda: None)
    result = _load(read_tools._get_scan_analysis_impl(7, DAY))
    assert not result["ok"] and result["error_kind"] == "invalid_request"


# ---------------------------------------------------------------------------
# get_scan_figure
# ---------------------------------------------------------------------------


def test_single_figure_returns_downscaled_image(share):
    from fastmcp.utilities.types import Image
    from PIL import Image as PILImage

    result = read_tools._get_scan_figure_impl(7, None, DAY)
    assert isinstance(result, Image)
    with PILImage.open(io.BytesIO(result.data)) as image:
        assert max(image.size) <= read_tools._MAX_FIGURE_EDGE_PX
        assert image.format == "PNG"


def test_multiple_figures_list_candidates(share):
    from PIL import Image as PILImage

    _, analysis_folder = _tree_paths(share)
    PILImage.new("RGB", (10, 10)).save(analysis_folder / "UC_TopView" / "extra.png")
    result = _load(read_tools._get_scan_figure_impl(7, None, DAY))
    assert result["ok"] and sorted(result["figures"]) == [
        "UC_TopView/extra.png",
        "UC_TopView/summary.png",
    ]

    picked = read_tools._get_scan_figure_impl(7, "extra", DAY)
    from fastmcp.utilities.types import Image

    assert isinstance(picked, Image)


def test_figure_no_match_names_the_available(share):
    result = _load(read_tools._get_scan_figure_impl(7, "nope", DAY))
    assert not result["ok"] and result["error_kind"] == "not_found"
    assert "summary.png" in result["message"]


def test_no_figures_is_not_found(share):
    scan_folder, analysis_folder = _tree_paths(share)
    for path in (analysis_folder / "UC_TopView").iterdir():
        path.unlink()
    (scan_folder / "analysis_status" / "topview_baseline.yaml").write_text(
        yaml.safe_dump({"state": "done", "display_files": []})
    )
    result = _load(read_tools._get_scan_figure_impl(7, None, DAY))
    assert not result["ok"] and result["error_kind"] == "not_found"


def test_analysis_tools_registered():
    import anyio

    from geecs_mcp import tool_names
    from geecs_mcp.server import create_server

    registered = {t.name for t in anyio.run(create_server().list_tools)}
    assert tool_names.GET_SCAN_ANALYSIS in registered
    assert tool_names.GET_SCAN_FIGURE in registered
