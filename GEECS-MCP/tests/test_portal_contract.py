"""The analysis domain against the portal's real routes (in process)."""

from __future__ import annotations

import base64
import json
import time

import httpx
import pytest

from geecs_mcp import clients, config, guard
from geecs_mcp.analysis import tools

DAY = "2026-07-12"
PORTAL_SCAN = 2
PORTAL_UID = "uid-002"


def _j(envelope: str) -> dict:
    return json.loads(envelope)


async def run(impl, *args, **kwargs) -> str:
    return await guard.run_async(impl, *args, **kwargs)


async def _finished(scan=PORTAL_SCAN, analyzer="UC_Crop", timeout=10.0) -> dict:
    """Run and wait with a real sleep (the portal's worker is a thread)."""
    started = _j(await run(tools._run_analysis, scan, analyzer, DAY, None))
    assert started["ok"] is True, started
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = _j(await run(tools._wait_for_analysis, scan, analyzer, DAY, 1))
        if out["finished"]:
            return out
    raise AssertionError("the fake analyzer never finished")


class TestResolution:
    async def test_a_scan_the_day_does_not_hold_is_not_found_never_the_newest(
        self, wired
    ) -> None:
        out = _j(await run(tools._get_scan_analysis, 99, DAY))
        assert out["ok"] is False
        assert out["error_kind"] == "not_found"
        assert out["scan_number"] == 99 and out["day"] == DAY

    async def test_argument_checks(self, wired) -> None:
        assert (
            _j(await run(tools._get_scan_analysis, "two", DAY))["error_kind"]
            == "invalid_request"
        )
        assert (
            _j(await run(tools._get_scan_analysis, 0, DAY))["error_kind"]
            == "invalid_request"
        )

    async def test_an_unconfigured_portal_names_its_key(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "service_url", lambda service: None)
        out = _j(await run(tools._list_analyzers))
        assert out["error_kind"] == "service_unreachable"
        assert "portal_url" in out["message"]


class TestCatalog:
    async def test_list_analyzers_is_the_portal_catalog(self, wired) -> None:
        out = _j(await run(tools._list_analyzers))
        assert out["ok"] is True
        by_id = {a["id"]: a for a in out["analyzers"]}
        assert set(by_id) == {"UC_Crop", "UC_Compact"}
        assert by_id["UC_Compact"]["destructive"] is True
        assert by_id["UC_Compact"]["kind"] == "job"
        assert out["count"] == 2

    async def test_list_analysis_groups_with_members_in_order(self, wired) -> None:
        out = _j(await run(tools._list_analysis_groups))
        assert out["groups"] == [
            {
                "id": "HTU/all",
                "name": "HTU_all",
                "description": "",
                "members": ["UC_Crop"],
            }
        ]
        assert "one diagnostic per job" in out["note"]


class TestRunAndWait:
    async def test_run_then_wait_returns_the_artifacts(self, wired) -> None:
        out = await _finished()
        assert out["state"] == "done"
        assert out["error"] is None
        paths = {a["path"] for a in out["artifacts"]}
        assert "UC_Crop/Array2DScanAnalyzer/cam_summary_waterfall.png" in paths
        assert "UC_Crop/Array2DScanAnalyzer/cam_summary_scalar_fit.json" in paths
        assert out["job"]["state"] == "done"

    async def test_get_scan_analysis_shows_the_job_with_a_bounded_log(
        self, wired
    ) -> None:
        await _finished()
        out = _j(await run(tools._get_scan_analysis, PORTAL_SCAN, DAY))
        assert out["ok"] is True and out["uid"] == PORTAL_UID
        crop = next(a for a in out["analyzers"] if a["id"] == "UC_Crop")
        assert crop["applicable"] is True
        assert crop["job"]["state"] == "done"
        assert "cam_summary_scalar_fit.json" in " ".join(crop["files"])

    async def test_a_failed_run_is_a_finished_result_with_the_error(
        self, wired, portal_fakes
    ) -> None:
        portal_fakes.analyzer.behaviour = "fail"
        out = await _finished()
        assert out["finished"] is True
        assert out["state"] == "failed"
        assert "boom" in out["error"]

    async def test_a_second_run_while_one_is_active_is_refused_with_the_job(
        self, wired, portal_fakes
    ) -> None:
        portal_fakes.analyzer.behaviour = "block"
        first = _j(await run(tools._run_analysis, PORTAL_SCAN, "UC_Crop", DAY, None))
        assert first["ok"] is True
        second = _j(await run(tools._run_analysis, PORTAL_SCAN, "UC_Crop", DAY, None))
        assert second["ok"] is False
        assert second["error_kind"] == "policy_refusal"
        assert second["job"]["state"] in ("queued", "running")
        waited = _j(await run(tools._wait_for_analysis, PORTAL_SCAN, "UC_Crop", DAY, 0))
        assert waited["finished"] is False
        portal_fakes.analyzer.release.set()

    async def test_unknown_analyzer_lists_the_catalog(self, wired) -> None:
        out = _j(await run(tools._run_analysis, PORTAL_SCAN, "UC_Nope", DAY, None))
        assert out["error_kind"] == "not_found"
        assert "UC_Crop" in out["analyzers"]

    async def test_a_destructive_kind_needs_the_typed_confirmation_passed_through(
        self, wired
    ) -> None:
        refused = _j(
            await run(tools._run_analysis, PORTAL_SCAN, "UC_Compact", DAY, None)
        )
        assert refused["error_kind"] == "invalid_request"
        allowed = _j(
            await run(tools._run_analysis, PORTAL_SCAN, "UC_Compact", DAY, PORTAL_SCAN)
        )
        assert allowed["ok"] is True

    async def test_wait_before_any_run_reports_unknown_not_done(self, wired) -> None:
        out = _j(await run(tools._wait_for_analysis, PORTAL_SCAN, "UC_Crop", DAY, 0))
        assert out["finished"] is False
        assert out["state"] == "unknown"
        assert out["files"] == []


class TestNumbers:
    async def test_a_json_sidecar_comes_back_parsed(self, wired) -> None:
        await _finished()
        out = _j(
            await run(
                tools._get_analysis_artifact,
                PORTAL_SCAN,
                "UC_Crop/Array2DScanAnalyzer/cam_summary_scalar_fit.json",
                DAY,
                65536,
            )
        )
        assert out["ok"] is True, out
        assert out["data"] == {"slope": 0.0508, "crossing_mm": -3.054}
        assert out["truncated"] is False
        assert out["size_bytes"] > 0

    async def test_a_big_table_is_cut_at_the_cap_and_says_so(self, wired) -> None:
        await _finished()
        out = _j(
            await run(
                tools._get_analysis_artifact,
                PORTAL_SCAN,
                "UC_Crop/Array2DScanAnalyzer/cam_shots.txt",
                DAY,
                1024,
            )
        )
        assert out["ok"] is True
        assert out["truncated"] is True
        assert len(out["text"]) <= 1024
        assert out["size_bytes"] > 1024

    async def test_a_binary_artifact_is_refused_not_decoded(self, wired) -> None:
        await _finished()
        out = _j(
            await run(
                tools._get_analysis_artifact,
                PORTAL_SCAN,
                "UC_Crop/Array2DScanAnalyzer/cam_processed.h5",
                DAY,
                65536,
            )
        )
        assert out["error_kind"] == "invalid_request"
        assert "binary" in out["message"]

    async def test_an_image_path_is_refused_toward_get_scan_figure(self, wired) -> None:
        out = _j(
            await run(
                tools._get_analysis_artifact, PORTAL_SCAN, "x/summary.png", DAY, 1000
            )
        )
        assert out["error_kind"] == "invalid_request"
        assert "get_scan_figure" in out["message"]

    async def test_a_path_outside_the_folder_is_the_portal_404(self, wired) -> None:
        out = _j(
            await run(
                tools._get_analysis_artifact, PORTAL_SCAN, "../../etc/passwd", DAY, 1000
            )
        )
        assert out["error_kind"] == "not_found"

    async def test_scalars_are_the_portal_binned_numbers_without_the_figure(
        self, wired
    ) -> None:
        out = _j(
            await run(
                tools._get_scan_scalars,
                PORTAL_SCAN,
                ["cam-MaxCounts"],
                DAY,
                "",
                '{"bin_col": "mono"}',
                "",
            )
        )
        assert out["ok"] is True, out
        assert out["bins"] == [4.0, 5.0, 6.0]
        assert out["counts"] == [1, 1, 1]
        assert out["series"]["cam-MaxCounts"]["center"] == [10.0, 12.5, 11.0]
        assert "figure" not in out and "code" not in out

    async def test_scalar_argument_checks(self, wired) -> None:
        assert (
            _j(await run(tools._get_scan_scalars, PORTAL_SCAN, [], DAY, "", "", ""))[
                "error_kind"
            ]
            == "invalid_request"
        )
        five = [
            "a",
            "b",
            "c",
            "d",
            "e",
        ]  # the PORTAL's cap (its 400), not restated here
        over = _j(
            await run(tools._get_scan_scalars, PORTAL_SCAN, five, DAY, "", "", "")
        )
        assert over["error_kind"] == "invalid_request" and "4" in over["message"]
        bad_confirm = _j(
            await run(tools._run_analysis, PORTAL_SCAN, "UC_Compact", DAY, "two")
        )
        assert bad_confirm["error_kind"] == "invalid_request"
        unknown = _j(
            await run(
                tools._get_scan_scalars,
                PORTAL_SCAN,
                ["nope"],
                DAY,
                "",
                '{"bin_col": "mono"}',
                "",
            )
        )
        assert unknown["error_kind"] == "not_found"


class TestFigures:
    async def test_the_one_summary_figure_is_the_default_reference(self, wired) -> None:
        await _finished()
        out = _j(await run(tools._get_scan_figure, PORTAL_SCAN, None, DAY, False))
        assert out["ok"] is True, out
        assert out["figure"] == "UC_Crop/Array2DScanAnalyzer/cam_summary_waterfall.png"
        assert out["kind"] == "summary"
        assert out["figure_url"].startswith(
            f"http://portal/run/{PORTAL_UID}/artifact?path="
        )
        assert "thumbnail" not in out

    async def test_a_thumbnail_is_a_bounded_jpeg(self, wired) -> None:
        await _finished()
        out = _j(
            await run(
                tools._get_scan_figure,
                PORTAL_SCAN,
                "UC_Crop/Array2DScanAnalyzer/cam_summary_waterfall.png",
                DAY,
                True,
            )
        )
        assert out["ok"] is True, out
        assert out["width"] == 40 and out["height"] == 30
        jpeg = base64.b64decode(out["thumbnail"]["base64"])
        assert jpeg[:2] == b"\xff\xd8"  # JPEG magic

    async def test_an_unknown_label_lists_the_figures(self, wired) -> None:
        await _finished()
        out = _j(await run(tools._get_scan_figure, PORTAL_SCAN, "nope.png", DAY, False))
        assert out["error_kind"] == "not_found"
        assert out["figures"] == [
            "UC_Crop/Array2DScanAnalyzer/cam_summary_waterfall.png"
        ]

    async def test_no_figures_yet_is_an_empty_listing(self, wired) -> None:
        out = _j(await run(tools._get_scan_figure, PORTAL_SCAN, None, DAY, False))
        assert out["ok"] is True
        assert out["figures"] == []


class TestGetBytes:
    async def test_truncation_is_by_content_not_by_a_missing_length(
        self, monkeypatch
    ) -> None:
        body = b"x" * 1024

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=body, headers={"content-type": "text/plain"}
            )

        monkeypatch.setattr(config, "service_url", lambda service: "http://portal")
        monkeypatch.setattr(
            clients, "transport_for", lambda service: httpx.MockTransport(handler)
        )
        try:
            exact = await clients.get_bytes("portal", "/x", max_bytes=1024)
            cut = await clients.get_bytes("portal", "/x", max_bytes=100)
        finally:
            await clients.reset_clients()
        assert exact[0] == body and exact[2] is False and exact[3] == 1024
        assert len(cut[0]) == 100 and cut[2] is True and cut[3] == 1024

    async def test_a_portal_404_is_a_refusal(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "no such artifact"})

        monkeypatch.setattr(config, "service_url", lambda service: "http://portal")
        monkeypatch.setattr(
            clients, "transport_for", lambda service: httpx.MockTransport(handler)
        )
        try:
            with pytest.raises(clients.ServiceRefusal) as caught:
                await clients.get_bytes("portal", "/x", max_bytes=10)
        finally:
            await clients.reset_clients()
        assert caught.value.kind == "not_found"
