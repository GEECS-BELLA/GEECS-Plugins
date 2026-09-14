"""The SSE stream: one round carries status, progress and console lines as JSON."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from geecs_scanner.service.demo import DemoQueueClient


def _events(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event and data is not None:
            out.append((event, data))
    return out


def test_one_round_carries_the_three_event_types(
    client: TestClient, preset_doc: dict, manager: DemoQueueClient
) -> None:
    assert (
        client.post(
            "/api/submit",
            json={"preset": preset_doc, "acknowledged": ["gateway_liveness"]},
        ).status_code
        == 200
    )
    for _ in range(3):
        manager.step()
    r = client.get("/api/events?once=1")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    assert r.headers["x-accel-buffering"] == "no"
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "status" and kinds[1] == "progress"
    assert kinds.count("console") >= 2
    status = dict(events)["status"]
    assert status["re_state"] == "running"
    progress = dict(events)["progress"]
    assert progress["scan_number"] == 47 and progress["shots_done"] == 3
    console = [d for e, d in events if e == "console"]
    assert console[0]["seq"] == 1
    assert any("Scan 047 claimed" in d["text"] for d in console)


def test_since_resumes_the_console_cursor(
    client: TestClient, manager: DemoQueueClient
) -> None:
    manager.streams.push_console_line("one")
    manager.streams.push_console_line("two")
    manager.streams.push_console_line("three")
    events = _events(client.get("/api/events?once=1&since=2").text)
    console = [d for e, d in events if e == "console"]
    assert [d["text"] for d in console] == ["three"]
