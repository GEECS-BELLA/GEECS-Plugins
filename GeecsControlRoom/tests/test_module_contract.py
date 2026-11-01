"""The Control Room against GeecsCanopy's own module, in process: the route shapes the pages read.

The module is built the way its own suite builds it (``geecs_canopy.testing``:
a ``FakeHare``, the device documents, the journal in a temp file), its app
mounted at ``/geecs`` on an ``httpx.ASGITransport`` that ``ModuleClient`` is
pointed at. A route or field rename in GeecsCanopy fails this suite in the
same PR. geecs-canopy is a dev-only dependency: ``geecs_control_room`` never
imports it (the ``control-room-edges`` contract).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from geecs_canopy.testing import EXPERIMENT, FakeSquirrel, live, make_env

from geecs_control_room.app import create_app
from geecs_control_room.images import Images
from geecs_control_room.module_client import ModuleClient
from geecs_control_room.settings import Settings
from tests.test_images import FakeCa

#: Every field screen.js reads from a role of /api/devices/{name}.
ROLE_FIELDS = {"role", "axis", "pv", "read", "writable", "verdict"}
#: Every field home.js reads from a journal row (and home.py's ``device``).
#: And the camera page's: a role's label and its run-lock flag.
CAMERA_FIELDS = ROLE_FIELDS | {"label", "locked"}
JOURNAL_FIELDS = {"device", "time", "outcome", "role", "axis", "value", "operator", "canopy_code"}  # fmt: skip


class _Failing(httpx.AsyncBaseTransport):
    """The module's transport, failing every request whose path ends in one of ``broken``."""

    def __init__(self, inner: httpx.AsyncBaseTransport, broken: set[str]) -> None:
        self.inner, self.broken = inner, broken

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if any(request.url.path.endswith(b) for b in self.broken):
            raise httpx.ConnectError("refused", request=request)
        return await self.inner.handle_async_request(request)


@pytest.fixture
async def room(tmp_path: Path):
    env = make_env(tmp_path, squirrel=FakeSquirrel())
    canopy = FastAPI()
    canopy.mount("/geecs", env.app)
    broken: set[str] = set()
    module = ModuleClient(
        "http://canopy.test",
        "read-key",
        transport=_Failing(httpx.ASGITransport(app=canopy), broken),
    )
    settings = Settings(experiment=EXPERIMENT, configs_root=tmp_path, canopy_url="http://canopy.test", read_key="read-key")  # fmt: skip
    ca = FakeCa()

    async def fetch(url: str) -> bytes:
        return b"\xff\xd8" + url.encode()

    app = create_app(settings, module, camera=Images(ca, fetch))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://room.test") as client:  # fmt: skip
        yield SimpleNamespace(env=env, client=client, broken=broken, ca=ca)


async def put(room, device: str, **body):
    r = await room.client.post(f"/api/devices/{device}/put", json={"operator": "ann", **body})  # fmt: skip
    return r.status_code, r.json()


async def test_a_device_carries_what_the_screen_reads(room) -> None:
    first = await room.client.get("/api/devices/ps1")
    assert first.status_code == 200 and {"state", "heartbeat"} <= set(first.json())  # fmt: skip
    roles = {r["role"]: r for r in first.json()["roles"]}
    assert all(ROLE_FIELDS <= set(r) for r in roles.values())
    await asyncio.sleep(0)  # the module subscribes as a task
    room.env.hare.on_frame({roles["current"]["pv"]: live(1.0)})
    again = {r["role"]: r for r in (await room.client.get("/api/devices/ps1")).json()["roles"]}  # fmt: skip
    assert again["current"]["verdict"] == "live"
    assert again["current_sp"]["writable"] is True and again["current"]["read"] == "monitor"  # fmt: skip


async def test_an_accepted_put_names_its_journal_row(room) -> None:
    status, body = await put(room, "Mot1", role="homf", axis="m1", value=1, confirm=True)  # fmt: skip
    assert (status, body["outcome"]) == (200, "accepted")
    assert isinstance(body["journalId"], int) and "code" not in body


@pytest.mark.parametrize(
    "device, body, status, code",
    [
        ("ps1", {"role": "reset", "value": 1, "operator": "bob"}, 403, "not_permitted"),
        ("Mot1", {"role": "homf", "axis": "m1", "value": 1}, 428, "confirm_required"),
    ],
)  # fmt: skip
async def test_a_refusal_carries_outcome_code_and_journal_row(room, device, body, status, code) -> None:  # fmt: skip
    got, answer = await put(room, device, **body)
    assert (got, answer["code"], answer["outcome"]) == (status, code, "denied")
    assert isinstance(answer["journalId"], int)


async def test_out_of_bounds_carries_its_detail(room) -> None:
    roles = (await room.client.get("/api/devices/ps1")).json()["roles"]
    sp = next(r["pv"] for r in roles if r["role"] == "current_sp")
    room.env.hare.meta[sp] = {"controlLow": 0.0, "controlHigh": 10.0}
    status, body = await put(room, "ps1", role="current_sp", value=11.0)
    assert (status, body["code"]) == (422, "out_of_bounds")
    assert body["outcome"] and body["detail"]


async def test_the_home_lists_come_from_the_roster_and_the_journal(room) -> None:
    await put(room, "Mot1", role="homf", axis="m1", value=1, confirm=True)
    roles = (await room.client.get("/api/devices/ps1")).json()["roles"]
    await asyncio.sleep(0)
    current = next(r["pv"] for r in roles if r["role"] == "current")
    room.env.hare.on_frame({current: live(1.0, severity="MAJOR", alarmStatus=3)})
    body = (await room.client.get("/api/home/devices")).json()
    assert body["ok"] and body["journal"] and body["unread"] == []
    assert {d["name"] for d in body["devices"]} == {"Cam1", "Mot1", "ps1"}
    assert {d["name"] for d in body["not_live"]} == {
        "Cam1",
        "Mot1",
        "ps1",
    }  # no heartbeat yet
    assert [d["name"] for d in body["in_alarm"]] == ["ps1"]
    last = next(d["last"] for d in body["not_live"] if d["name"] == "Mot1")
    assert JOURNAL_FIELDS <= set(last) and last["outcome"] == "accepted"


async def test_a_live_device_leaves_not_live_and_its_family_reaches_the_pickers(room) -> None:  # fmt: skip
    roles = (await room.client.get("/api/devices/ps1")).json()["roles"]
    await asyncio.sleep(0)
    room.env.hare.on_frame({r["pv"]: live(0) for r in roles if r["read"] == "monitor"})
    body = (await room.client.get("/api/home/devices")).json()
    families = {d["name"]: d["family"] for d in body["devices"]}
    assert families == {"Cam1": "ad_camera_chain", "Mot1": "motor_record", "ps1": "easydriver"}  # fmt: skip
    assert {d["name"] for d in body["not_live"]} == {"Cam1", "Mot1"}


async def test_a_device_route_that_fails_is_unread(room) -> None:
    room.broken.add("/devices/Cam1")
    body = (await room.client.get("/api/home/devices")).json()
    assert body["ok"] and body["unread"] == [{"name": "Cam1"}]
    assert (await room.client.get("/api/devices/Cam1")).json()["code"] == "module_unreachable"  # fmt: skip


async def test_a_button_runs_an_action_plan_and_the_page_reads_its_run(room) -> None:
    folder = room.env.folder
    room.env.hare.meta["bench:ps1:current:SP"] = {"controlLow": 0.0, "controlHigh": 5.0}
    (folder / "action_library").mkdir()
    (folder / "action_library" / "actions.yaml").write_text(
        "plans:\n  park:\n    description: park it\n    steps:\n"
        "      - {do: set, device: ps1, variable: current_sp, value: 1.0}\n"
    )
    (folder / "screens").mkdir()
    (folder / "screens" / "s.yaml").write_text(
        "title: S\ngroups:\n  - {title: G, at: [1, 1], widgets: [{kind: button, label: Park, bind: {action: park}}]}\n"  # fmt: skip
    )
    page = await room.client.get("/screens/s")
    assert page.status_code == 200 and 'data-action="park"' in page.text
    listed = (await room.client.get("/api/actions")).json()["plans"]
    assert {"name", "description", "hazardSteps", "running"} <= set(listed[0])
    started = await room.client.post("/api/actions/park/run", json={"operator": "ann"})
    assert started.status_code == 202
    for _ in range(100):
        run = (
            await room.client.get(f"/api/actions/runs/{started.json()['runId']}")
        ).json()
        if run["state"] != "running":
            break
        await asyncio.sleep(0.01)
    assert {"id", "plan", "state", "steps"} <= set(run) and run["state"] == "done"
    assert {"index", "state", "journalId"} <= set(run["steps"][0])


#: Every field home.js reads from a preset, a diff row and the diff's counts.
PRESET_FIELDS = {"id", "name", "time", "operator", "scope", "golden", "pvCount"}
ROW_FIELDS = {"device", "role", "axis", "saved", "live", "tolerance", "word"}


async def test_the_presets_region_reads_what_the_module_answers(room) -> None:
    sp, mot = "bench:ps1:current:SP", "test:mot1:m1.VAL"
    fake = room.env.squirrel  # the FakeSquirrel behind the module's client
    room.env.hare.meta[sp] = {"controlLow": 0.0, "controlHigh": 5.0}
    for pv, value in (
        (sp, 1.0),
        ("bench:ps1:current", 1.0),
        (mot, 2.0),
        ("test:mot1:m1.RBV", 2.0),
    ):
        room.env.hare.values[pv] = live(value)
    for address, rb, device, role in ((sp, "bench:ps1:current", "ps1", "current_sp"), (mot, "test:mot1:m1.RBV", "Mot1", "m1.val")):  # fmt: skip
        fake.handle(httpx.Request("POST", "http://s/squirrel/v1/pvs", json={"setpointAddress": address, "readbackAddress": rb, "device": device, "description": role}))  # fmt: skip
    fake.values = {sp: 1.0, mot: 2.0}
    saved = await room.client.post("/api/presets", json={"name": "bench-a", "operator": "ann", "scope": {"devices": ["ps1", "Mot1"]}})  # fmt: skip
    assert saved.status_code == 201 and PRESET_FIELDS <= set(saved.json())
    listing = (await room.client.get("/api/home/presets")).json()
    assert listing["ok"] and [p["name"] for p in listing["presets"]] == ["bench-a"] and listing["screens"] == []  # fmt: skip
    room.env.hare.values[sp] = live(1.5)
    sid = saved.json()["id"]
    diff = (
        await room.client.get(f"/api/presets/{sid}/diff", params={"devices": ["ps1"]})
    ).json()
    assert [r["device"] for r in diff["rows"]] == ["ps1"] and ROW_FIELDS <= set(
        diff["rows"][0]
    )
    assert diff["rows"][0]["word"] == "differs" and diff["counts"]["differs"] == 1
    gold = await room.client.post(
        f"/api/presets/{sid}/golden", json={"operator": "ann"}
    )
    assert gold.status_code == 200 and gold.json()["golden"] is True
    chip = (await room.client.get("/api/home/baseline")).json()
    assert (chip["ok"], chip["golden"], chip["differs"], chip["incomplete"], chip["presets"]) == (True, 1, 1, 0, [{"name": "bench-a", "differs": 1, "incomplete": 0}])  # fmt: skip
    del room.env.hare.values[
        mot
    ]  # the motor's setpoint unread now: the baseline is incomplete, never "as baseline"
    chip = (
        await room.client.get("/api/home/baseline", params={"fresh": "true"})
    ).json()
    assert (chip["differs"], chip["incomplete"]) == (1, 1)
    room.env.hare.values[mot] = live(2.0)
    started = await room.client.post(f"/api/presets/{sid}/restore", json={"operator": "ann", "confirm": True, "devices": ["ps1"]})  # fmt: skip
    assert started.status_code == 202 and {"runId", "id", "plan", "state", "steps"} <= set(started.json())  # fmt: skip
    for _ in range(100):
        run = (
            await room.client.get(f"/api/actions/runs/{started.json()['runId']}")
        ).json()
        if run["state"] != "running":
            break
        await asyncio.sleep(0.01)
    assert (
        run["state"] == "done"
        and run["plan"] == f"restore:{sid}"
        and fake.restores == []
    )


#: Every field device_pages.py and device.js read from the device route (the roles' are ROLE_FIELDS).
DEVICE_FIELDS = {"family", "axes", "state", "heartbeat", "roles"}


@pytest.mark.parametrize(
    "path, status, words",
    [
        ("/device/camera/Cam1", 200, ['data-role="exposure"', 'id="device-state"']),
        ("/device/motor/Mot1", 200, ['data-axis="m1"']),
        ("/device/supply/ps1", 200, ['data-role="current"']),
        ("/device/camera/Mot1", 422, ["Wrong kind: motor_record is a motor"]),
        ("/device/camera/Nope", 404, ["not on the module&#39;s roster"]),
    ],
)  # fmt: skip
async def test_a_device_page_resolves_through_the_module(room, path: str, status: int, words: list[str]) -> None:  # fmt: skip
    r = await room.client.get(path)
    assert r.status_code == status and all(w in r.text for w in words)


async def test_the_device_page_reads_its_fields_and_the_picker_its_states(room) -> None:  # fmt: skip
    body = (await room.client.get("/api/devices/Mot1")).json()
    assert DEVICE_FIELDS <= set(body) and body["axes"] == ["m1"]
    assert all(ROLE_FIELDS <= set(r) for r in body["roles"])
    listing = (await room.client.get("/api/device/camera")).json()
    assert listing["roster"] is True and [(d["name"], d["state"]) for d in listing["devices"]] == [("Cam1", "unknown")]  # fmt: skip


async def test_the_camera_page_reads_the_image_and_arrays_at_the_pvs_the_module_names(room) -> None:  # fmt: skip
    roles = {r["role"]: r for r in (await room.client.get("/api/devices/Cam1")).json()["roles"]}  # fmt: skip
    assert all(CAMERA_FIELDS <= set(r) for r in roles.values())
    assert roles["image"]["read"] == "once" and roles["exposure"]["locked"] is True
    image = roles["image"]["pv"]
    frame = await room.client.get("/api/devices/Cam1/frame.jpg")
    assert frame.status_code == 200 and room.ca.calls == [
        ("text", image),
        ("host", image),
    ]
    arrays = (await room.client.get("/api/devices/Cam1/arrays")).json()
    asked = {r: e["pv"] for r, e in roles.items() if e["read"] == "on_request"}
    assert arrays["pvs"] == asked and set(asked) == {"profile_x", "profile_y", "cursor_profile_x", "cursor_profile_y", "histogram"}  # fmt: skip
    page = await room.client.get("/device/camera/Cam1")
    assert 'id="camera"' in page.text and "/api/devices/Cam1/frame.jpg" in page.text
    assert (
        'id="hazard-dialog"' in page.text
    )  # a policy-marked role is asked, then sent with confirm
