"""The scan.log tail: read-only, whole lines, resumable, honest when unreadable."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from geecs_scanner.service.demo import DemoQueueClient
from geecs_scanner.service.scanlog import read_scan_log


def test_reader_returns_whole_lines_and_resumes(tmp_path: Path) -> None:
    log = tmp_path / "scan.log"
    log.write_text("one\ntwo\nthr")
    out = read_scan_log(str(tmp_path), 0)
    assert out.available and out.lines == ["one", "two"]
    assert out.offset == len("one\ntwo\n")
    # the partial line stays unread until its newline lands
    again = read_scan_log(str(tmp_path), out.offset)
    assert again.lines == [] and again.offset == out.offset
    log.write_text("one\ntwo\nthree\nfour\n")
    more = read_scan_log(str(tmp_path), again.offset)
    assert more.lines == ["three", "four"]
    assert more.offset == log.stat().st_size


def test_reader_restarts_when_the_file_shrank_and_bounds_a_chunk(
    tmp_path: Path,
) -> None:
    log = tmp_path / "scan.log"
    log.write_text("".join(f"line {i}\n" for i in range(1000)))
    out = read_scan_log(str(tmp_path), 0, limit=100)
    assert out.more is True and 0 < len(out.lines) < 1000
    assert out.offset <= 100
    # an offset past the end (the file was replaced) starts over
    log.write_text("fresh\n")
    assert read_scan_log(str(tmp_path), 5000).lines == ["fresh"]


def test_reader_never_creates_anything(tmp_path: Path) -> None:
    missing = tmp_path / "Scan999"
    out = read_scan_log(str(missing), 0)
    assert out.available is False and "no scan.log readable" in out.detail
    assert not missing.exists()
    (tmp_path / "Scan001").mkdir()
    out = read_scan_log(str(tmp_path / "Scan001"), 0)
    assert out.available is False
    assert list((tmp_path / "Scan001").iterdir()) == []


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        event = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event and data is not None:
            out.append((event, data))
    return out


def test_stream_carries_the_scan_log_from_the_start_documents_folder(
    client: TestClient, preset_doc: dict, manager: DemoQueueClient
) -> None:
    # before any run: no folder, said once, nothing invented
    assert client.get("/api/scanlog").json()["available"] is False
    r = client.post(
        "/api/submit", json={"preset": preset_doc, "acknowledged": ["gateway_liveness"]}
    )
    assert r.status_code == 200, r.text
    for _ in range(10):
        manager.step()
    progress = client.get("/api/progress").json()
    assert progress["scan_folder"].endswith("Scan047")
    assert progress["day"] is not None and len(progress["day"]) == 10
    events = _events(client.get("/api/events?once=1").text)
    logs = [d for e, d in events if e == "log"]
    assert len(logs) == 1 and logs[0]["available"] is True
    assert logs[0]["scan_number"] == 47
    assert "starting (dir=" in logs[0]["lines"][0]
    assert any("step 1/" in ln for ln in logs[0]["lines"])
    # the plain route resumes from an offset
    direct = client.get("/api/scanlog").json()
    assert direct["lines"] == logs[0]["lines"]
    assert client.get(f"/api/scanlog?offset={direct['offset']}").json()["lines"] == []


def test_finished_rows_carry_run_uids_for_the_portal_links(
    client: TestClient, preset_doc: dict, manager: DemoQueueClient
) -> None:
    client.post(
        "/api/submit", json={"preset": preset_doc, "acknowledged": ["gateway_liveness"]}
    )
    while manager.status().re_state == "running":
        manager.step()
    row = client.get("/api/queue").json()["finished"][0]
    assert row["scan_numbers"] == [47] and len(row["run_uids"]) == 1
    assert client.get("/api").json()["portal"] is None  # no --portal-url in tests
