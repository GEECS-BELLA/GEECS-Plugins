"""The device pages: the picker lists the registry by kind, a page resolves its device with the module, and every misfit is a pane."""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from geecs_control_room.app import create_app
from geecs_schemas import load_families

from geecs_control_room.device_pages import (
    GRID_MIN,
    KINDS,
    auto_panel,
    drawn_panel,
    page_panel,
    registered,
)
from geecs_control_room.home import launcher
from geecs_control_room.module_client import ModuleClient
from geecs_control_room.settings import Settings

_NATIVE = """schema_version: 1
devices:
  Cam1: {kind: aravis_camera, trigger_source: line0}
  Mot1: {kind: epics_motor, axes: [m1, m2]}
"""
_PANEL = """schema_version: 1
devices:
  Cam2: {family: ad_camera_chain, omit: [image], labels: {exposure: Shutter}}
  ps1: {family: easydriver}
"""
_ROSTER = [
    {"name": "Cam1", "family": "ad_camera_chain", "state": "live"},
    {"name": "Mot1", "family": "motor_record", "state": "stale"},
    {"name": "Cam2", "family": "ad_camera_chain", "state": "error"},
    {"name": "U_DB", "family": "PhantomCamera", "source": "gateway", "state": "live"},
]


def _body(name: str) -> dict:
    d = next(d for d in _ROSTER if d["name"] == name)
    axes = ["m1", "m2"] if name == "Mot1" else []
    roles = [{"role": "desc", "axis": a, "label": "description"} for a in axes]
    if name == "Cam2":  # as the module resolves it: image omitted, exposure relabelled
        roles = [{"role": "exposure", "axis": None, "label": "Shutter"}, {"role": "cam1_counter", "axis": None, "label": "camera frames"}]  # fmt: skip
    return d | {"axes": axes, "heartbeat": "live", "roles": roles}


@pytest.fixture
def folder(settings: Settings) -> Path:
    folder = settings.experiment_dir
    folder.mkdir(parents=True)
    (folder / "native_devices.yaml").write_text(_NATIVE)
    (folder / "panel_devices.yaml").write_text(_PANEL)
    return folder


def _room(settings: Settings, down: bool = False, refuse: int = 0) -> TestClient:
    def answer(request: httpx.Request) -> httpx.Response:
        if down:
            raise httpx.ConnectError("refused", request=request)
        if refuse:
            return httpx.Response(refuse, json={"code": "refused", "detail": "no read for this key"})  # fmt: skip
        path = request.url.path.removeprefix("/geecs/v1")
        if path == "/devices":
            return httpx.Response(200, json={"devices": _ROSTER})
        name = path.removeprefix("/devices/")
        if name not in {d["name"] for d in _ROSTER}:
            return httpx.Response(404, json={"code": "not_found", "detail": f"no device {name!r}"})  # fmt: skip
        return httpx.Response(200, json=_body(name))

    module = ModuleClient("http://canopy.test", "k", transport=httpx.MockTransport(answer))  # fmt: skip
    return TestClient(create_app(settings, module))


def _pane(html: str) -> tuple[str, str]:
    state, title = re.search(r'class="state" data-state="(\w+)">\s*<div class="t">([^<]*)<', html).groups()  # fmt: skip
    return state, title


def test_the_registry_lists_each_kind_from_both_files(settings: Settings, folder: Path) -> None:  # fmt: skip
    cams, problem = registered(settings, "camera")
    assert problem is None and cams == [{"name": "Cam1", "family": "ad_camera_chain"}, {"name": "Cam2", "family": "ad_camera_chain"}]  # fmt: skip
    assert [d["name"] for d in registered(settings, "motor")[0]] == ["Mot1"]
    assert [d["name"] for d in registered(settings, "supply")[0]] == ["ps1"]
    assert registered(settings, "trace")[0] == []


def test_the_roster_read_annotates_each_row_or_says_unknown(settings: Settings, folder: Path) -> None:  # fmt: skip
    with _room(settings) as c:
        body = c.get("/api/device/camera").json()
    assert body["roster"] is True and [(d["name"], d["state"], d["kit"]) for d in body["devices"]] == [("Cam1", "live", "ok"), ("Cam2", "error", "failed")]  # fmt: skip
    with _room(settings, down=True) as c:
        body = c.get("/api/device/camera").json()
    assert body["roster"] is False and {d["state"] for d in body["devices"]} == {"unknown"}  # fmt: skip
    assert body["detail"].startswith("the module did not answer")


def test_the_picker_draws_the_select_and_the_free_text_goes(settings: Settings, folder: Path) -> None:  # fmt: skip
    with _room(settings) as c:
        r = c.get("/device/camera")
        assert r.status_code == 200 and re.findall(r'<option value="(\w+)"', r.text) == ["Cam1", "Cam2"]  # fmt: skip
        assert 'name="typed"' in r.text
        go = c.get("/device/camera", params={"typed": " Bench Cam "}, follow_redirects=False)  # fmt: skip
        assert (go.status_code, go.headers["location"]) == (303, "/device/camera/Bench%20Cam")  # fmt: skip


def test_a_device_draws_its_resolved_roles_by_level(settings: Settings, folder: Path) -> None:  # fmt: skip
    with _room(settings) as c:
        r = c.get("/device/camera/Cam2")
    assert r.status_code == 200 and 'id="device-state" data-state="failed">error<' in r.text  # fmt: skip
    roles = re.findall(r'<tr data-role="(\w+)"', r.text)
    assert (
        "image" not in roles and "exposure" in roles
    )  # the panel entry omits and relabels
    assert "<td>Shutter</td>" in r.text and '<details class="disc"><summary>Expert' in r.text  # fmt: skip
    assert 'data-current="Cam2"' in r.text  # the switcher


def test_a_per_axis_device_has_a_table_per_axis(settings: Settings, folder: Path) -> None:  # fmt: skip
    with _room(settings) as c:
        r = c.get("/device/motor/Mot1")
    assert r.status_code == 200
    assert set(re.findall(r'data-axis="(\w+)"', r.text)) == {"m1", "m2"} and r.text.count("axis m") == 2  # fmt: skip


@pytest.mark.parametrize(
    "path, status, state, title",
    [
        ("/device/camera/Nope", 404, "denied", "Nope is not on the module's roster"),
        ("/device/camera/Mot1", 422, "denied", "Wrong kind: motor_record is a motor"),
        ("/device/camera/U_DB", 422, "denied", "Wrong kind: U_DB (a GEECS DB device, PhantomCamera) is a settables"),
        ("/device/pump/Cam1", 404, "error", "No device kind 'pump'"),
        ("/device/pump", 404, "error", "No device kind 'pump'"),
    ],
)  # fmt: skip
def test_a_device_the_page_cannot_draw_is_a_pane(settings: Settings, folder: Path, path: str, status: int, state: str, title: str) -> None:  # fmt: skip
    with _room(settings) as c:
        r = c.get(path)
    assert r.status_code == status and _pane(r.text) == (state, title.replace("'", "&#39;"))  # fmt: skip


def test_a_silent_module_is_stale_and_the_page_asks_again(settings: Settings, folder: Path) -> None:  # fmt: skip
    with _room(settings, down=True) as c:
        r = c.get("/device/camera/Cam1")
    assert r.status_code == 503 and _pane(r.text) == ("stale", "The module did not answer")  # fmt: skip
    assert (
        'data-recover="1"' in r.text and "<option" in r.text
    )  # the picker still draws


def test_the_home_pickers_carry_their_kind(settings: Settings, folder: Path) -> None:  # fmt: skip
    (folder / "home.yaml").write_text("device_pages:\n  - {title: Cameras, families: [ad_camera_chain]}\n  - {title: Mixed, families: [ad_camera_chain, motor_record]}\n")  # fmt: skip
    assert [p.kind for p in launcher(settings).device_pages] == ["camera", None]
    assert set(KINDS) == {"camera", "trace", "motor", "supply", "settables"}


@pytest.mark.parametrize("status, state", [(401, "denied"), (403, "denied"), (409, "error"), (503, "stale")])  # fmt: skip
def test_only_a_silent_module_asks_again(settings: Settings, folder: Path, status: int, state: str) -> None:  # fmt: skip
    with _room(settings, refuse=status) as c:
        r = c.get("/device/camera/Cam1")
    assert _pane(r.text)[0] == state and ("data-recover" in r.text) == (state == "stale")  # fmt: skip
    assert state == "stale" or "no read for this key" in r.text


def test_a_device_the_two_files_disagree_on_is_no_kind(settings: Settings, folder: Path) -> None:  # fmt: skip
    (folder / "panel_devices.yaml").write_text(_PANEL + "  Mot1: {family: easydriver}\n")  # fmt: skip
    rows, problem = registered(settings, "motor")
    assert rows == [] and problem == "Mot1: native_devices.yaml names motor_record, panel_devices.yaml easydriver; not listed"  # fmt: skip
    assert "Mot1" not in [d["name"] for d in registered(settings, "supply")[0]]


@pytest.mark.parametrize(
    ("path", "trail"),
    [
        ("/device/camera/Cam2", '<a href="/device/camera">camera</a><span class="sep" aria-hidden="true">›</span><span class="mono" aria-current="page">Cam2</span>'),
        ("/device/camera/Nope", '<a href="/device/camera">camera</a><span class="sep" aria-hidden="true">›</span><span class="mono" aria-current="page">Nope</span>'),
        ("/device/motor", '<span aria-current="page">motor</span>'),
    ],
)  # fmt: skip
def test_every_device_page_has_the_way_back(settings: Settings, folder: Path, path: str, trail: str) -> None:  # fmt: skip
    """The brand is the home's link and the trail's first step; then the kind's picker; the page itself is not a link."""
    with _room(settings) as c:
        html = c.get(path).text
    assert '<a class="brand" href="/">' in html
    nav = re.search(r'<nav class="crumbs" aria-label="Breadcrumb">(.*?)</nav>', html, re.S).group(1)  # fmt: skip
    assert trail in nav
    assert html.index('class="brand"') < html.index(
        'class="crumbs"'
    )  # right after the brand


def test_an_unknown_kind_has_no_trail_but_the_brand(settings: Settings, folder: Path) -> None:  # fmt: skip
    with _room(settings) as c:
        html = c.get("/device/nope").text
    assert '<a class="brand" href="/">' in html and 'class="crumbs"' not in html


# ---- the panel a page draws: the family's, else the automatic layout


def _entry(role: str, kind: str, writable: bool, read: str = "monitor") -> dict:
    return {"role": role, "type": kind, "writePv": f"pv:{role}:SP" if writable else None, "read": read}  # fmt: skip


def test_the_automatic_layout_is_every_writable_role_then_every_monitored_reading() -> (
    None
):
    entries = [_entry("Velocity", "setpoint", True), _entry("Position", "readback", False), _entry("Home", "command", True, "never"),
               _entry("Mode", "enum", True), _entry("Trace", "array", False, "on_request")]  # fmt: skip
    panel = auto_panel(entries)
    assert panel["rail"] == [{"title": "Settings", "roles": ["Velocity", "Home", "Mode"], "layout": "fields"}]  # role order, nothing hidden  # fmt: skip
    assert panel["readings"] == [
        "Position"
    ]  # an array is no reading (it carries no word)


def test_many_switches_get_a_grid_of_their_own() -> None:
    switches = [_entry(f"DO{i}", "enum", True) for i in range(GRID_MIN)]
    assert auto_panel(switches)["rail"] == [{"title": "Settings", "roles": [f"DO{i}" for i in range(GRID_MIN)], "layout": "grid"}]  # fmt: skip
    mixed = auto_panel([_entry("Gain", "setpoint", True), *switches])["rail"]
    assert [(g["title"], g["layout"]) for g in mixed] == [("Settings", "fields"), ("Switches", "grid")]  # fmt: skip
    few = auto_panel([_entry("Gain", "setpoint", True), *switches[: GRID_MIN - 1]])[
        "rail"
    ]
    assert [(g["title"], g["layout"]) for g in few] == [("Settings", "fields")]
    assert (
        auto_panel([_entry("T", "readback", False)])["rail"] == []
    )  # nothing settable


def test_a_device_draws_only_the_roles_it_serves_and_a_tweak_only_whole() -> None:
    motor = load_families()["motor_record"].panel.model_dump(mode="json")
    served = {r.name for r in load_families()["motor_record"].roles}
    whole = drawn_panel(motor, served)
    assert whole["tweak"] == {"val": {"step": "twv", "forward": "twf", "reverse": "twr"}} and whole["inside"] == {"twv", "twf", "twr"}  # fmt: skip
    # a device that omits the step: no tweak; the target and the two commands each drawn by their own type
    part = drawn_panel(motor, served - {"twv"})
    assert part["tweak"] == {} and part["inside"] == set()
    assert part["rail"][0]["roles"] == ["val", "twf", "twr", "stop"]
    # a group with none of its roles served is not drawn; a reading the device omits is not either
    gone = drawn_panel(motor, served - {"homf", "homr", "hls"})
    assert [g["title"] for g in gone["rail"]] == ["Move", "Speed"] and "hls" not in gone["readings"]  # fmt: skip


@pytest.mark.parametrize("panel", ["none", "naming it"])
def test_a_cameras_rail_never_carries_its_camera_roi(panel: str) -> None:
    """The camera ROI is the camera page's own (one draft, behind its dialog): never a rail field, whatever the family's panel says."""
    from geecs_schemas import FamilyFile

    doc = load_families()["ad_camera_chain"].model_dump(mode="json")
    if (
        panel == "none"
    ):  # a camera family with no panel: the automatic layout, every writable role
        doc["panel"] = None
    else:  # a panel that names the camera ROI in a rail group
        doc["panel"]["rail"][0]["roles"] += ["min_x", "size_x"]
    family = FamilyFile.model_validate(doc)
    body = {
        "roles": [
            _entry(r.name, r.type, r.writable, r.read_mode) | {"axis": None}
            for r in family.roles
        ]
    }
    drawn = page_panel(body, family, [], kind="camera")
    railed = {r for g in drawn["panel"]["rail"] for r in g["roles"]}
    assert railed and not railed & {"min_x", "min_y", "size_x", "size_y"}
