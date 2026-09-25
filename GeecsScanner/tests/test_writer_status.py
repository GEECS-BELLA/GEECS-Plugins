"""The Tiled writer's heartbeat → one kit word, shown on the page and in /health.

Hermetic: a heartbeat file per state is written the way ``geecs-tiled-writer``
writes it (``geecs_bluesky.tiled_spool.write_heartbeat``), the service is
pointed at it, and the verdict is read through ``/health`` and the SSE
status.  The chip's word per state runs the page's own ``renderWriterChip``
under node.  The rule itself is the writer's (``tiled_spool.heartbeat_verdict``,
pinned in GeecsBluesky); what is pinned here is the projection — the
words reach the API per heartbeat (the measured 25–28 s per run:
``pending`` ≤ 1 fresh is ok, a ``.failed`` file or a backlog with a
failing attempt is failed, silence is degraded, a registration in flight
is not silence), that a missing or unreadable file is degraded and never
an error, and that **nothing about a submit changes** with the word.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from geecs_bluesky.tiled_spool import WriterHeartbeat, write_heartbeat
from geecs_web_theme import STATES
from geecs_web_theme.testing import node_available

from geecs_scanner.service import ProgressCache, ScannerService
from geecs_scanner.service.demo import (
    DemoQueueClient,
    DemoResolver,
    demo_preflight,
)
from geecs_scanner.service.models import SubmitIn
from geecs_bluesky.tiled_spool import (
    PENDING_BACKLOG_MIN,
    PENDING_OK_MAX,
    STALE_WHILE_REGISTERING_S,
)

from geecs_scanner.service.writer_status import read_writer_status, writer_verdict
from geecs_scanner.web import create_app

_PKG = Path(__file__).resolve().parents[1] / "geecs_scanner"
NOW = 1_700_000_000.0


def _heartbeat(**over) -> WriterHeartbeat:
    fields = dict(
        pid=4242,
        version="0.103.1",
        started_at=NOW - 3600,
        last_sweep=NOW - 1,
        sweep_interval=2.0,
        tiled_uri="http://tiled.test:8000",
        tiled_reachable=True,
        last_ok=NOW - 30,
        last_error=None,
        pending=0,
        in_progress=0,
        failed=0,
        done=12,
    )
    fields.update(over)
    return WriterHeartbeat(**fields)


# ---------------------------------------------------------------- the rule


@pytest.mark.parametrize(
    "over, state",
    [
        ({}, "ok"),
        (
            {"pending": PENDING_OK_MAX, "in_progress": 1},
            "ok",
        ),  # the run that just ended
        ({"pending": PENDING_OK_MAX + 1}, "degraded"),  # a second one waiting
        ({"pending": PENDING_BACKLOG_MIN}, "degraded"),  # short runs draining
        (
            {"pending": PENDING_BACKLOG_MIN, "last_error": "Scan012: 503"},
            "failed",
        ),  # a backlog because every attempt fails
        ({"failed": 1}, "failed"),  # a file set aside for an operator
        ({"tiled_reachable": False}, "degraded"),
        ({"last_sweep": NOW - 7}, "degraded"),  # > 3 sweeps of silence
        ({"last_sweep": NOW - 5}, "ok"),  # < 3 sweeps: alive
        (
            {"last_sweep": NOW - 30, "registering": "r", "registering_since": NOW - 30},
            "ok",
        ),  # a registration in flight is work, not silence
        (
            {
                "last_sweep": NOW - STALE_WHILE_REGISTERING_S - 1,
                "registering": "r",
                "registering_since": NOW - STALE_WHILE_REGISTERING_S - 1,
            },
            "degraded",
        ),  # …until it has taken longer than any registration
    ],
    ids=[
        "idle",
        "one-pending",
        "two-pending",
        "backlog-draining",
        "backlog-failing",
        "set-aside",
        "tiled-down",
        "stale",
        "fresh-enough",
        "registering",
        "registering-wedged",
    ],
)
def test_verdict_words(over: dict, state: str) -> None:
    out = writer_verdict(_heartbeat(**over), Path("/x/heartbeat.json"), now=NOW)
    assert out.state == state, out.detail
    assert out.state in STATES
    assert out.detail  # the chip's hover text always says why


def test_verdict_carries_the_counts_and_the_error() -> None:
    hb = _heartbeat(
        pending=3, in_progress=1, last_error="Scan012: HTTPStatusError: 503"
    )
    out = writer_verdict(hb, Path("/x"), now=NOW)
    assert (out.pending, out.in_progress, out.failed) == (3, 1, 0)
    assert out.last_ok == NOW - 30 and out.last_error == hb.last_error
    assert "503" in out.detail and out.stale is False


def test_stale_beats_everything_else() -> None:
    # A stale heartbeat's counts are history: silence is the finding.
    out = writer_verdict(_heartbeat(last_sweep=NOW - 60, failed=2), Path("/x"), now=NOW)
    assert out.state == "degraded" and out.stale is True and out.failed == 2


def test_missing_heartbeat_is_degraded_not_an_error(tmp_path: Path) -> None:
    out = read_writer_status(tmp_path / "heartbeat.json", now=NOW)
    assert out.state == "degraded" and out.stale is True
    assert "geecs-tiled-writer" in out.detail and str(tmp_path) in out.detail


def test_unreadable_heartbeat_is_degraded(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    path.write_text("{not json")
    assert read_writer_status(path, now=NOW).state == "degraded"
    # a directory at the path (GEECS_TILED_WRITER_STATE misconfigured), or a
    # file another account owns: still a word, never an exception through
    # /api/status (which the SSE generator polls every second)
    assert read_writer_status(tmp_path, now=NOW).state == "degraded"


# ------------------------------------------------------------ the service


def _service(tmp_path: Path, heartbeat: WriterHeartbeat | None) -> ScannerService:
    path = tmp_path / "heartbeat.json"
    if heartbeat is not None:
        write_heartbeat(path, heartbeat)
    streams = ProgressCache(clock=lambda: 1_000.0)
    return ScannerService(
        DemoQueueClient(streams, period=0.0, first_scan=46, user="t"),
        DemoResolver(),
        experiment="Demo",
        identity="t",
        streams=streams,
        preflight=demo_preflight,
        version="0.0.0+test",
        heartbeat_path=path,
    )


@pytest.mark.parametrize(
    "over, state",
    [
        ({}, "ok"),
        ({"pending": 2}, "degraded"),
        ({"failed": 1}, "failed"),
        ({"last_sweep": time.time() - 3600}, "degraded"),
        (None, "degraded"),
    ],
    ids=["ok", "two-pending", "set-aside", "stale", "no-file"],
)
def test_health_and_status_carry_the_writer_word(
    tmp_path: Path, over: dict | None, state: str
) -> None:
    hb = None
    if over is not None:
        hb = _heartbeat(
            **{"last_sweep": time.time(), "last_ok": time.time() - 30, **over}
        )
    client = TestClient(create_app(_service(tmp_path, hb)))
    health = client.get("/health").json()
    assert health["ok"] is True and health["readiness"] == "ready"
    tw = health["tiled_writer"]
    assert tw["state"] == state, tw
    assert set(tw) >= {
        "state",
        "detail",
        "pending",
        "in_progress",
        "failed",
        "last_ok",
        "last_error",
        "stale",
    }
    # the same object rides the status poll (the page's chip)
    assert client.get("/api/status").json()["tiled_writer"]["state"] == state
    # and the SSE status frame
    text = client.get("/api/events", params={"once": 1}).text
    frame = next(
        line
        for line in text.splitlines()
        if line.startswith("data:") and '"readiness"' in line
    )
    assert json.loads(frame[len("data:") :])["tiled_writer"]["state"] == state


def test_the_word_never_gates_a_submit(tmp_path: Path) -> None:
    """A failed writer changes nothing about a submit: the spool waits."""
    service = _service(
        tmp_path, _heartbeat(last_sweep=time.time(), failed=3, pending=9)
    )
    assert service.tiled_writer().state == "failed"
    out = service.submit(
        SubmitIn(
            preset=service.preset("jet_pressure_sweep"),
            acknowledged=["gateway_liveness"],
        )
    )
    assert out.item_uid
    # the preflight's questions are the demo's usual one — nothing about the writer
    report = service.preflight(service.preset("jet_pressure_sweep"))
    assert "writer" not in report.model_dump_json().lower()
    assert (
        service.status().tiled_writer.state == "failed"
    )  # still shown, still not a gate


def test_heartbeat_path_defaults_to_the_units_directory(monkeypatch) -> None:
    from geecs_scanner.service.writer_status import default_heartbeat_path

    monkeypatch.setenv("GEECS_TILED_WRITER_STATE", "/var/lib/geecs-tiled-writer")
    assert default_heartbeat_path() == Path(
        "/var/lib/geecs-tiled-writer/heartbeat.json"
    )


# --------------------------------------------------------------- the chip


def test_chip_word_per_state() -> None:
    """``renderWriterChip`` maps the verdict to a kit word and the hover text."""
    if not node_available():  # pragma: no cover
        pytest.skip("node not available")
    script = (_PKG / "static" / "scanner.js").read_text()
    m = re.search(
        r"\n  function renderWriterChip\(tw\) \{\n(.*?)\n  \}\n", script, re.S
    )
    assert m, "renderWriterChip is a top-level function of the page's IIFE"
    k = re.search(r"var K = \{[^}]*\};", script).group(0)
    harness = (
        "var chip = {}; function $(id) { return chip; }\n"
        + k
        + "\nfunction setChip(el, state, word, title) { el.state = state; el.word = word; el.title = title; }\n"
        + f"function render(tw) {{\n{m.group(1)}\n}}\n"
        + """
var out = [];
[null,
 {state: "ok", detail: "writer alive"},
 {state: "degraded", detail: "stale"},
 {state: "failed", detail: "2 set aside"},
 {state: "unknown", detail: ""}
].forEach(function (tw) { render(tw); out.push({state: chip.state, word: chip.word, title: chip.title}); });
console.log(JSON.stringify(out));
"""
    )
    result = subprocess.run(
        ["node", "-"], input=harness, text=True, capture_output=True, check=True
    )
    unread, ok, degraded, failed, other = json.loads(result.stdout)
    assert [r["word"] for r in (unread, ok, degraded, failed, other)] == [
        "tiled writer"
    ] * 5
    assert unread["state"] == "unknown"
    assert ok["state"] == "ok" and ok["title"] == "writer alive"
    assert degraded["state"] == "degraded" and failed["state"] == "failed"
    assert other["state"] == "degraded"  # an unexpected word is never shown as ok
    assert failed["title"] == "2 set aside"


def test_page_has_the_chip(client: TestClient) -> None:
    html = client.get("/").text
    assert 'id="chip-writer"' in html and ">tiled writer<" in html
