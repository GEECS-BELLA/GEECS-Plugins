"""post_logbook_entry against the logbook's and the portal's real routes (in process)."""

from __future__ import annotations

import json
import time

import httpx

from geecs_mcp import config, guard
from geecs_mcp.analysis import tools as analysis_tools
from geecs_mcp.logbook import tools

DAY = "2026-07-12"
SCAN = 2
FIGURE = "UC_Crop/Array2DScanAnalyzer/cam_summary_waterfall.png"
IDENTITY = "agent, admin supervision (Ada)"


def _j(envelope: str) -> dict:
    return json.loads(envelope)


async def post(*, figures=None, payload=None, scan=SCAN, operator="Ada", body="ok"):
    return _j(
        await guard.run_async(
            tools._post_logbook_entry, DAY, body, operator, scan, figures, payload
        )
    )


async def _entries(logbook) -> list[dict]:
    """Every entry the logbook holds, read through its own change feed."""
    transport = httpx.ASGITransport(app=logbook)
    async with httpx.AsyncClient(transport=transport, base_url="http://logbook") as c:
        reply = await c.get(
            "/api/entries", params={"since": "1970-01-01T00:00:00+00:00"}
        )
    reply.raise_for_status()
    return reply.json()["entries"]


async def _analysed() -> None:
    """The fake analyzer has written its figure (the portal's worker is a thread)."""
    started = _j(
        await guard.run_async(analysis_tools._run_analysis, SCAN, "UC_Crop", DAY, None)
    )
    assert started["ok"] is True, started
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        out = _j(
            await guard.run_async(
                analysis_tools._wait_for_analysis, SCAN, "UC_Crop", DAY, 1
            )
        )
        if out["finished"]:
            assert out["state"] == "done", out
            return
    raise AssertionError("the fake analyzer never finished")


class TestEntry:
    async def test_an_entry_is_an_agent_draft_in_the_scans_book(
        self, wired, logbook
    ) -> None:
        out = await post(body="PMQ3 crossing at -3.05 mm")
        assert out["ok"] is True, out
        assert out["url"] == f"http://logbook/entry/{out['entry_id']}"
        assert out["kind"] == "agent_analysis" and out["status"] == "draft"
        assert out["author"] == IDENTITY
        assert out["attachments"] == [] and out["complete"] is True
        (stored,) = await _entries(logbook)
        assert stored["entry_id"] == out["entry_id"]
        assert stored["book"] == "scans" and stored["scan"] == SCAN
        assert stored["author"] == IDENTITY
        assert stored["body_md"] == "PMQ3 crossing at -3.05 mm"

    async def test_a_payload_rides_through_and_the_logbook_validates_it(
        self, wired, logbook
    ) -> None:
        good = {"kind": "analysis", "analyzer": "UC_Crop", "metrics": {"slope": 0.05}}
        out = await post(payload=good)
        assert out["ok"] is True, out
        (stored,) = await _entries(logbook)
        assert stored["payload"]["metrics"] == {"slope": 0.05}

        bad = await post(payload={"kind": "analysis", "metrics": {"slope": "steep"}})
        assert bad["ok"] is False
        assert bad["error_kind"] == "invalid_request"
        assert len(await _entries(logbook)) == 1

    async def test_a_blank_operator_writes_nothing(self, wired, logbook) -> None:
        out = await post(operator="  ")
        assert out["error_kind"] == "invalid_request"
        assert "operator is required" in out["message"]
        assert await _entries(logbook) == []

    async def test_an_unconfigured_logbook_names_its_key(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "service_url", lambda service: None)
        out = await post()
        assert out["error_kind"] == "service_unreachable"
        assert out["service"] == "logbook"
        assert "logbook_url" in out["message"]


class TestFigures:
    async def test_a_figure_is_attached_by_the_portal_end_to_end(
        self, wired, logbook
    ) -> None:
        await _analysed()
        out = await post(
            figures=[{"scan_number": SCAN, "path": FIGURE, "caption": "waterfall"}]
        )
        assert out["ok"] is True, out
        assert out["complete"] is True
        (row,) = out["attachments"]
        assert row["attached"] is True and row["appended"] is True
        assert row["path"] == FIGURE and row["link"]
        assert "entry_id" not in row  # appended to OUR entry, not a fresh one
        (stored,) = await _entries(logbook)
        assert stored["entry_id"] == out["entry_id"]
        assert stored["kind"] == "agent_analysis" and stored["status"] == "draft"
        assert len(stored["attachments"]) == 1
        assert row["link"] in stored["body_md"]

    async def test_a_scan_the_day_does_not_hold_refuses_with_nothing_written(
        self, wired, logbook
    ) -> None:
        out = await post(figures=[{"scan_number": 99, "path": FIGURE}])
        assert out["error_kind"] == "not_found"
        assert out["scan_number"] == 99
        assert await _entries(logbook) == []

    async def test_a_non_image_is_refused_with_nothing_written(
        self, wired, logbook
    ) -> None:
        out = await post(
            figures=[{"scan_number": SCAN, "path": "UC_Crop/x/cam_summary_fit.json"}]
        )
        assert out["error_kind"] == "invalid_request"
        assert "get_analysis_artifact" in out["message"]
        assert await _entries(logbook) == []

    async def test_malformed_references_are_refused(self, wired, logbook) -> None:
        for figures in ("a.png", ["a.png"], [{"scan_number": SCAN}]):
            out = await post(figures=figures)
            assert out["error_kind"] == "invalid_request", (figures, out)
        assert await _entries(logbook) == []

    async def test_a_failed_figure_keeps_the_entry_and_says_so(
        self, wired, logbook
    ) -> None:
        """The entry exists once created: a failed send is a row, never a lost id."""
        await _analysed()
        out = await post(
            figures=[
                {"scan_number": SCAN, "path": "UC_Crop/missing.png"},
                {"scan_number": SCAN, "path": FIGURE},
            ]
        )
        assert out["ok"] is True, out
        assert out["complete"] is False
        assert "do not post it again" in out["note"]
        missing, attached = out["attachments"]
        assert missing["attached"] is False
        assert missing["error_kind"] == "not_found"
        assert attached["attached"] is True
        (stored,) = await _entries(logbook)
        assert stored["entry_id"] == out["entry_id"]
        assert len(stored["attachments"]) == 1
