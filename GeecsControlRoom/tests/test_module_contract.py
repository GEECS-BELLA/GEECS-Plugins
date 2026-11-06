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
import html
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
import numpy as np
from geecs_canopy.testing import (
    EXPERIMENT,
    PANELS,
    REGISTRY,
    POLICY,
    FakeSquirrel,
    live,
    make_env,
    write_documents,
)

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
    counter = roles["image_counter"][
        "pv"
    ]  # the JPEG is held on MJPG1's counter, at the PV the module names
    assert frame.status_code == 200 and room.ca.calls == [
        ("scalar", counter),
        ("text", image),
        ("host", image),
    ]
    assert counter.endswith("MJPG1:ArrayCounter_RBV") and frame.headers["etag"]
    arrays = (await room.client.get("/api/devices/Cam1/arrays")).json()
    asked = {r: e["pv"] for r, e in roles.items() if e["read"] == "on_request" and e["type"] == "array"}  # fmt: skip
    assert arrays["pvs"] == asked and set(asked) == {"profile_x", "profile_y", "cursor_profile_x", "cursor_profile_y", "histogram"}  # fmt: skip
    # the raw frame's source: the module names it (read never) and its on-request scalars, none judged, none in /arrays
    assert (
        roles["raw_frame"]["read"] == "never" and roles["raw_frame"]["verdict"] is None
    )
    assert {roles[r]["read"] for r in ("raw_width", "raw_height", "raw_counter")} == {"on_request"}  # fmt: skip
    assert all(roles[r]["verdict"] is None for r in ("raw_width", "raw_height", "raw_counter"))  # fmt: skip
    raw = await room.client.get(
        "/api/devices/Cam1/frame.raw"
    )  # read at the PVs the module names
    assert raw.status_code == 200 and ("frame", tuple(roles[r]["pv"] for r in ("raw_frame", "raw_width", "raw_height"))) in room.ca.calls  # fmt: skip
    assert {("scalar", roles["raw_counter"]["pv"]), ("scalar", roles["full_scale"]["pv"])} <= set(room.ca.calls)  # fmt: skip
    page = await room.client.get("/device/camera/Cam1")
    assert 'id="camera"' in page.text and "/api/devices/Cam1/frame.jpg" in page.text
    assert (
        'id="hazard-dialog"' in page.text
    )  # a policy-marked role is asked, then sent with confirm


def page_json(text: str, element: str) -> object:
    """The JSON a page carries in ``<script type="application/json" id=element>``."""
    m = re.search(
        rf'<script type="application/json" id="{element}">(.*?)</script>', text
    )
    assert m, element
    return json.loads(m.group(1))


def stop_start(role: str) -> str:
    """A panel button pair as the trace template draws it: 0 then 1, disabled until the module's word."""
    return "".join(f'<button type="button" data-role="{role}" data-value="{v}" data-pair aria-pressed="false" disabled>{w}</button>' for v, w in ((0, "Stop"), (1, "Start")))  # fmt: skip


async def test_the_trace_page_draws_the_familys_panel_and_reads_the_trace_at_the_pvs_the_module_names(room, tmp_path: Path) -> None:  # fmt: skip
    # a bench scope beside the shared documents, written before the module's first read
    write_documents(tmp_path, panels=PANELS + '  scope1: {family: scope_contract, prefix: "bench:scope1:"}\n', policy=POLICY.replace("easydriver]", "easydriver, scope_contract]"))  # fmt: skip
    roles = {r["role"]: r for r in (await room.client.get("/api/devices/scope1")).json()["roles"]}  # fmt: skip
    assert all(CAMERA_FIELDS <= set(r) for r in roles.values())
    assert (roles["data"]["read"], roles["time_axis"]["read"], roles["arr_counter"]["read"]) == ("on_request", "on_request", "monitor")  # fmt: skip
    assert roles["data"]["verdict"] is None and roles["ch_a_range"]["writable"] is True
    page = await room.client.get("/device/trace/scope1")
    assert page.status_code == 200 and 'id="trace"' in page.text and "/api/devices/scope1/trace" in page.text  # fmt: skip
    # the rail is the family's panel, each role drawn by its type: enums as selects, setpoints as numbers
    assert [t for t in ("<h4>Acquisition</h4>", "<h4>Trigger</h4>", "<h4>Channel A</h4>", "<h4>Channel D</h4>") if t not in page.text] == []  # fmt: skip
    assert re.search(
        r'<select id="c-\d+-\d+" data-role="ch_a_range" disabled>', page.text
    )
    assert 'data-role="trigger_level" step="any"' in page.text and 'data-role="num_images" step="1"' in page.text  # fmt: skip
    assert ["ch_a_enable", "ch_a_range", "ch_a_coupling", "ch_a_offset"] in [json.loads(html.unescape(g)) for g in re.findall(r'data-roles="([^"]*)"', page.text)]  # fmt: skip
    assert 'class="live" data-role="ch_a_overrange"' in page.text
    assert page_json(page.text, "trace-ys") == [{"role": "data", "label": "capture", "units": "V", "counter": "arr_counter"}]  # fmt: skip
    assert 'id="yrole"' not in page.text  # one array: no toggle
    # acquire, named in the panel's buttons, is a Stop / Start pair writing 0 and 1, not a select
    assert (
        stop_start("acquire") in page.text and '<select id="c-acquire"' not in page.text
    )
    # the plot is Plotly through the theme's template: the vendored bundle and plot.js, both served from /theme
    assert '/theme/plotly-cartesian-3.1.1.min.js' in page.text and '/theme/plot.js' in page.text and '<canvas id="trace-plot"' not in page.text  # fmt: skip
    for asset in ("/theme/plotly-cartesian-3.1.1.min.js", "/theme/plot.js"):
        assert (await room.client.get(asset)).status_code == 200, asset
    # the trace: the counter, the capture and its channel count, the axis, at the PVs the module names
    room.ca.floats_by = {roles["data"]["pv"]: np.zeros(8, dtype=np.float32), roles["time_axis"]["pv"]: np.arange(2, dtype=np.float32)}  # fmt: skip
    room.ca.scalars = {roles["arr_channels"]["pv"]: 4.0}
    trace = await room.client.get("/api/devices/scope1/trace")
    assert trace.status_code == 200 and (trace.headers["x-trace-channels"], trace.headers["x-trace-samples"]) == ("4", "2")  # fmt: skip
    asked = [a for k, a in room.ca.calls if k in ("floats", "scalar")]
    assert asked == [roles["arr_counter"]["pv"], (roles["data"]["pv"],), roles["arr_channels"]["pv"], roles["arr_counter"]["pv"], (roles["time_axis"]["pv"],)]  # fmt: skip


async def test_the_trace_page_draws_a_spectrometer_through_the_same_template(room, tmp_path: Path) -> None:  # fmt: skip
    # the bench spectrometer beside the shared documents: the other array_detector family, one channel
    write_documents(tmp_path, panels=PANELS + '  cct1: {family: spectrometer_contract, prefix: "bench:cct1:"}\n', policy=POLICY.replace("easydriver]", "easydriver, spectrometer_contract]"))  # fmt: skip
    roles = {r["role"]: r for r in (await room.client.get("/api/devices/cct1")).json()["roles"]}  # fmt: skip
    assert (roles["spectrum"]["read"], roles["wavelength"]["read"], roles["arr_counter"]["read"]) == ("on_request", "on_request", "monitor")  # fmt: skip
    assert roles["background_save"]["read"] == "never" and roles["background_save"]["writable"] is True  # fmt: skip
    page = await room.client.get("/device/trace/cct1")
    assert page.status_code == 200 and 'id="trace"' in page.text and "/api/devices/cct1/trace" in page.text  # fmt: skip
    assert [t for t in ("<h4>Acquisition</h4>", "<h4>Trigger</h4>", "<h4>Background</h4>") if t not in page.text] == []  # fmt: skip
    assert (
        "<h4>Channel A</h4>" not in page.text
    )  # the spectrometer's rail, not the scope's
    assert '<button class="btn sm" type="button" data-role="background_save" data-value="1" disabled>Save background</button>' in page.text  # fmt: skip
    assert 'data-role="integration_time" step="any"' in page.text and 'data-role="averages" step="1"' in page.text  # fmt: skip
    assert 'class="live" data-role="saturated"' in page.text and 'data-x-units="nm"' in page.text  # fmt: skip
    # two arrays over the wavelengths, each by its own counter; the toggle shows processed while subtraction is on
    assert page_json(page.text, "trace-ys") == [
        {"role": "spectrum", "label": "raw spectrum", "units": "counts", "counter": "arr_counter"},
        {"role": "processed", "label": "processed spectrum", "units": "counts", "counter": "proc_counter"},
    ]  # fmt: skip
    assert page_json(page.text, "trace-show-while") == {
        "processed": "background_enable"
    }
    assert '<button type="button" data-y="processed" aria-pressed="false">processed spectrum</button>' in page.text  # fmt: skip
    assert stop_start("acquire") in page.text
    assert (roles["processed"]["read"], roles["proc_counter"]["read"], roles["processed"]["verdict"]) == ("on_request", "monitor", None)  # fmt: skip
    # one channel: the counter, the spectrum (no channel count asked), the counter, the wavelengths
    room.ca.floats_by = {roles["spectrum"]["pv"]: np.zeros(8, dtype=np.float32), roles["wavelength"]["pv"]: np.arange(8, dtype=np.float32)}  # fmt: skip
    trace = await room.client.get("/api/devices/cct1/trace")
    assert trace.status_code == 200 and (trace.headers["x-trace-channels"], trace.headers["x-trace-samples"]) == ("1", "8")  # fmt: skip
    asked = [a for k, a in room.ca.calls if k in ("floats", "scalar")]
    assert asked == [roles["arr_counter"]["pv"], (roles["spectrum"]["pv"],), roles["arr_counter"]["pv"], (roles["wavelength"]["pv"],)]  # fmt: skip
    # ?y=processed: PROC's own array, held on PROC's own counter, at the PVs the module names; the axis is shared
    room.ca.floats_by[roles["processed"]["pv"]] = np.ones(8, dtype=np.float32)
    room.ca.calls.clear()
    processed = await room.client.get("/api/devices/cct1/trace?y=processed")
    assert processed.status_code == 200 and np.frombuffer(processed.content, dtype="<f4")[8:].tolist() == [1.0] * 8  # fmt: skip
    asked = [a for k, a in room.ca.calls if k in ("floats", "scalar")]
    assert asked[:3] == [roles["proc_counter"]["pv"], (roles["processed"]["pv"],), roles["proc_counter"]["pv"]]  # fmt: skip
    unknown = await room.client.get("/api/devices/cct1/trace?y=wavelength")
    assert unknown.status_code == 404 and unknown.json()["detail"] == "the axis wavelength draws no array 'wavelength': it draws spectrum, processed"  # fmt: skip
    # Saturated_RBV is an enum reading, but its alarm is the IOC's (MAJOR on the CCT), and the
    # module's rule stands: a MAJOR role is in error, and so is the device while it reads so
    await asyncio.sleep(0)  # the module subscribes as a task
    room.env.hare.on_frame(
        {roles["saturated"]["pv"]: live(1, severity="MAJOR", alarmStatus=1)}
    )
    device = (await room.client.get("/api/devices/cct1")).json()
    saturated = next(r for r in device["roles"] if r["role"] == "saturated")
    assert (saturated["verdict"], device["state"]) == ("error", "error")
    # a STAT1 reading is fresh by the plugin's counter AND computed by its flag: with the
    # centroid off while spectra keep coming, the held centroid reads 'off', never 'live'
    room.env.hare.on_frame({roles["stat_counter"]["pv"]: live(1), roles["compute_centroid"]["pv"]: live(0), roles["compute_statistics"]["pv"]: live(1), roles["centroid_index"]["pv"]: live(25.0), roles["total"]["pv"]: live(4.0e6)})  # fmt: skip
    room.env.hare.on_frame({roles["stat_counter"]["pv"]: live(2)})
    by_role = {r["role"]: r["verdict"] for r in (await room.client.get("/api/devices/cct1")).json()["roles"]}  # fmt: skip
    assert by_role["centroid_index"] == "off" and by_role["total"] == "live"


async def test_the_directory_reads_the_states_and_heartbeats_the_module_serves(room, tmp_path: Path) -> None:  # fmt: skip
    # a bench scope (served by an IOC, so a heartbeat) and a bench spectrometer beside the shared documents
    panels = PANELS + '  scope1: {family: scope_contract, prefix: "bench:scope1:", ioc: scopeioc}\n  cct1: {family: spectrometer_contract, prefix: "bench:cct1:"}\n'  # fmt: skip
    write_documents(tmp_path, panels=panels)
    body = (await room.client.get("/api/home/devices")).json()
    assert body["ok"] and set(body["words"]) == {
        "Cam1",
        "Mot1",
        "ps1",
        "scope1",
        "cct1",
    }
    assert {n: body["words"][n]["state"] for n in ("Cam1", "scope1", "cct1")} == {"Cam1": "unknown", "scope1": "unknown", "cct1": "unknown"}  # fmt: skip
    assert body["words"]["Cam1"]["heartbeat"] is None and body["words"]["scope1"]["heartbeat"] == "unknown"  # fmt: skip
    # the scope's IOC heartbeat advances: the module's word reaches the row
    await asyncio.sleep(0)
    room.env.hare.on_frame({"SCOPEIOC:HEARTBEAT": live(1)})
    room.env.hare.on_frame({"SCOPEIOC:HEARTBEAT": live(2)})
    beat = (await room.client.get("/api/home/devices")).json()["words"]["scope1"]
    assert (beat["heartbeat"], beat["heartbeatKit"]) == ("live", "ok")
    # every device under its page kind, linked; the trace group holds both array detectors
    html = (await room.client.get("/")).text
    groups = re.findall(r'data-group="([^"]+)"', html)
    assert groups == ["camera", "trace", "motor", "supply"]
    for kind, name in (
        ("camera", "Cam1"),
        ("trace", "scope1"),
        ("trace", "cct1"),
        ("supply", "ps1"),
    ):
        assert f'<a href="/device/{kind}/{name}">{name}</a>' in html
    assert '<td class="mono">scopeioc</td>' in html
    # and the page it links to leads back
    page = (await room.client.get("/device/trace/cct1")).text
    assert '<a href="/device/trace">trace</a>' in page and '<span class="mono" aria-current="page">cct1</span>' in page  # fmt: skip


# ---- the settables page: a motor (two axes), a supply, and the module's DB devices, through one template


def _db_devices() -> list:
    """Two GEECS DB devices as the module's gateway source serves them (``gateway._role``'s shape): a PLC's channels, a filter wheel."""
    from geecs_canopy.native import Resolved
    from geecs_schemas import Role

    def var(name: str, pv: str, kind: str, writable: bool, **more) -> Role:
        fields = {
            "name": name,
            "suffix": pv,
            "label": name,
            "units": None,
            "type": kind,
        }
        return Role.model_construct(**fields | ({"writable": True, "write": pv + ":SP"} if writable else {}) | more)  # fmt: skip

    status = Role(name="connected", suffix="connected", type="enum", label="connected")
    plc = [var(f"DO.Ch{i}", f"do_ch{i}", "enum", True) for i in range(1, 6)] + [var("DI.Ch1", "di_ch1", "enum", False), status]  # fmt: skip
    wheel = [var("Position", "position", "enum", True), var("Comment", "comment", "setpoint", True, text="string"), var("Temperature", "temperature", "readback", False), status]  # fmt: skip

    def resolved(name: str, family: str, roles: list) -> Resolved:
        return Resolved(name, family, f"Test:{name.lower()}:", (), tuple(roles), (), None, frozenset(r.name for r in roles if r.writable), "gateway")  # fmt: skip

    return [resolved("U_PLC", "PLC Digital Output", plc), resolved("U_Wheel", "Thorlabs FW102C", wheel)]  # fmt: skip


@pytest.fixture
async def db_room(tmp_path: Path):
    """The room over a module that also serves two DB devices (read before the first request)."""
    env = make_env(tmp_path, gateway=_db_devices)
    write_documents(tmp_path, registry=REGISTRY.replace("axes: [m1]", "axes: [m1, m2]"), policy=POLICY.replace("easydriver]", "easydriver, PLC Digital Output]"))  # fmt: skip
    await env.module.read_gateway()
    canopy = FastAPI()
    canopy.mount("/geecs", env.app)
    module = ModuleClient("http://canopy.test", "read-key", transport=httpx.ASGITransport(app=canopy))  # fmt: skip
    settings = Settings(experiment=EXPERIMENT, configs_root=tmp_path, canopy_url="http://canopy.test", read_key="read-key")  # fmt: skip
    app = create_app(settings, module, camera=Images(FakeCa(), lambda url: b""))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://room.test") as client:  # fmt: skip
        yield SimpleNamespace(env=env, client=client)


async def _meta(room, device: str, role: str, axis, meta: dict) -> None:
    """Hare's metadata for one role's PVs (read and write), as the module reads its bounds."""
    entry = next(r for r in (await room.client.get(f"/api/devices/{device}")).json()["roles"] if (r["role"], r["axis"]) == (role, axis))  # fmt: skip
    for pv in {entry["pv"], entry["writePv"]}:
        room.env.hare.meta[pv] = {"pv": pv} | meta


def _groups(text: str) -> list[tuple[str, list[str]]]:
    """Each rail group's heading and roles, in the page's order."""
    found = re.findall(r'<section class="group([^"]*)" data-roles="([^"]*)">\s*<h4>([^<]*)</h4>', text)  # fmt: skip
    return [(title + (" (grid)" if "grid-cells" in cls else ""), json.loads(html.unescape(roles))) for cls, roles, title in found]  # fmt: skip


async def test_a_motor_draws_one_rail_per_axis_with_its_tweak(db_room) -> None:
    page = (await db_room.client.get("/device/motor/Mot1")).text
    # the axis selector, one pane per axis, the second hidden
    assert re.findall(r'data-axis-pick="(\w+)"', page) == ["m1", "m2"]
    assert '<div class="camera axis-pane" data-axis-pane="m1">' in page and '<div class="camera axis-pane" data-axis-pane="m2" hidden>' in page  # fmt: skip
    # the family's panel, once per axis; the tweak's roles stay in their group, drawn with the target
    assert _groups(page) == [("Move", ["val", "twv", "twf", "twr", "stop"]), ("Home", ["homf", "homr"]), ("Speed", ["velo"])] * 2  # fmt: skip
    for axis in ("m1", "m2"):
        assert f'data-role="twr" data-axis="{axis}" data-value="1" aria-label="tweak reverse" disabled>−</button>' in page  # fmt: skip
        assert f'<input type="number" id="c-{axis}-1-1-step" data-role="twv" data-axis="{axis}" step="any" disabled>' in page  # fmt: skip
        assert f'class="live" data-role="hls" data-axis="{axis}"' in page
    assert (
        len(re.findall(r'<input[^>]*data-role="twv"', page)) == 2
    )  # in the tweak only, never a field of its own
    assert "Laid out automatically" not in page
    # every write names its axis: the module takes it and answers (its bounds read from the record, as Hare serves them)
    await _meta(db_room, "Mot1", "val", "m2", {"controlLow": -5.0, "controlHigh": 5.0})
    status, answer = await put(db_room, "Mot1", role="val", axis="m2", value=1.5)
    assert status == 200 and answer["outcome"] == "accepted", json.dumps(answer)
    assert db_room.env.hare.puts[-1][0].endswith("m2.VAL")


async def test_a_supply_draws_its_output_pair_and_readings(db_room) -> None:
    page = (await db_room.client.get("/device/supply/ps1")).text
    assert _groups(page) == [
        ("Output", ["output_sp", "current_sp"]),
        ("Faults", ["reset"]),
    ]
    assert 'data-role="output_sp" data-value="0" data-pair aria-pressed="false" disabled>Off</button>' in page  # fmt: skip
    assert [r for r in ("current", "voltage", "output", "fault", "interlock") if f'class="live" data-role="{r}"' not in page] == []  # fmt: skip
    assert 'id="hazard-dialog"' in page and 'id="operator"' in page and 'data-axis-pick' not in page  # fmt: skip


async def test_a_db_device_is_laid_out_automatically_and_written_through_the_module(db_room) -> None:  # fmt: skip
    roster = (await db_room.client.get("/api/device/settables")).json()
    assert [(d["name"], d["family"]) for d in roster["devices"]] == [("U_PLC", "PLC Digital Output"), ("U_Wheel", "Thorlabs FW102C")]  # fmt: skip
    plc = await db_room.client.get("/device/settables/U_PLC")
    assert plc.status_code == 200 and "Laid out automatically" in plc.text
    # five writable channels: a grid of their own; the read-only channel and the state are readings
    assert _groups(plc.text) == [
        ("Settings (grid)", [f"DO.Ch{i}" for i in range(1, 6)])
    ]
    assert re.findall(r'class="live" data-role="([^"]+)"', plc.text) == [
        "DI.Ch1",
        "connected",
    ]
    wheel = (await db_room.client.get("/device/settables/U_Wheel")).text
    assert _groups(wheel) == [("Settings", ["Position", "Comment"])]
    assert re.search(r'<select id="c-1-1" data-role="Position" disabled>', wheel)
    # a native device under settables, and a DB device under motor: the wrong kind, with the way to the right one
    wrong = await db_room.client.get("/device/motor/U_PLC")
    assert wrong.status_code == 422 and "/device/settables/U_PLC" in wrong.text
    assert (await db_room.client.get("/device/settables/Mot1")).status_code == 422
    # a channel written through the module's put route, the operator's name, sent once
    await _meta(db_room, "U_PLC", "DO.Ch3", None, {"enumChoices": ["off", "on"]})
    status, answer = await put(db_room, "U_PLC", role="DO.Ch3", value=1)
    assert status == 200 and answer["outcome"] == "accepted", json.dumps(answer)
    assert db_room.env.hare.puts[-1][0] == "Test:u_plc:do_ch3:SP"  # fmt: skip
    # and the home's directory lists them under settables, linked
    home = (await db_room.client.get("/")).text
    assert '<a href="/device/settables/U_Wheel">U_Wheel</a>' in home
