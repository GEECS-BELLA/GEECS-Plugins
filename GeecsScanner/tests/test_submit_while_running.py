"""#905: a submit while a plan runs queues the item behind it — never removed.

The demo manager never had the bug (it appends and starts only when
idle), so this runs the REAL client's add-then-start sequence
(``ZmqQueueClient``) over a stub manager API that is mid-plan: the queue
is started, ``queue_start`` answers *busy*, and the item must stay queued.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from geecs_bluesky.plan_names import GEECS_PLAN_NAMES
from geecs_bluesky.qs_client.client import QserverConfig, ZmqQueueClient

from geecs_scanner.service import ProgressCache, ScannerService
from geecs_scanner.service.demo import DemoResolver, demo_preflight
from geecs_scanner.web import create_app


class _BusyManagerAPI:
    """A RE Manager executing its queue: adds land behind the running item."""

    def __init__(self) -> None:
        self.items: list[dict] = []
        self.removed: list[str] = []

    def status(self) -> dict:
        return {
            "re_state": "running",
            "manager_state": "executing_queue",
            "worker_environment_exists": True,
            "worker_environment_state": "executing_plan",
            "items_in_queue": len(self.items),
            "running_item_uid": "run-1",
        }

    def queue_get(self) -> dict:
        return {
            "items": list(self.items),
            "running_item": {"item_uid": "run-1", "name": "count"},
        }

    def queue_clear(self) -> None:
        self.items = []

    def item_add(self, item: Any, *, user: str | None = None) -> dict:
        row = dict(item.to_dict())
        row["item_uid"] = f"uid-{len(self.items) + 1}"
        row["user"] = user
        self.items.append(row)
        return {"item": row}

    def queue_start(self) -> None:
        raise RuntimeError("Request failed: RE Manager is busy.")

    def item_remove(self, *, uid: str | None = None, pos: Any = None) -> None:
        self.removed.append(uid or "")
        self.items = [i for i in self.items if i["item_uid"] != uid]

    def plans_allowed(self, **_: Any) -> dict:
        return {
            "success": True,
            "plans_allowed": {n: {"name": n} for n in GEECS_PLAN_NAMES},
        }


def _service_over(api: _BusyManagerAPI) -> ScannerService:
    client = ZmqQueueClient(QserverConfig("tcp://x:1", "tcp://x:2", "x:3"), user="t")
    client._api = api  # noqa: SLF001 — the client's injection point, as its own tests use it
    return ScannerService(
        client,
        DemoResolver(),
        experiment="Demo",
        identity="t",
        streams=ProgressCache(),
        preflight=demo_preflight,
    )


def test_submit_while_a_plan_runs_is_queued_behind_it() -> None:
    api = _BusyManagerAPI()
    service = _service_over(api)
    web = TestClient(create_app(service))
    preset = service.preset("jet_pressure_sweep")
    r = web.post(
        "/api/submit", json={"preset": preset, "acknowledged": ["gateway_liveness"]}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["item_uid"] == "uid-1" and body["planned_shots"] == 70
    # the item stays queued behind the running one; nothing was removed
    assert [i["item_uid"] for i in api.items] == ["uid-1"]
    assert api.removed == []
    assert web.get("/api/status").json()["items_in_queue"] == 1
