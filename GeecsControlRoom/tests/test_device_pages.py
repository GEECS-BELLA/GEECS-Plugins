"""The device pages: the picker lists the registry by kind, a page resolves its device with the module, and every misfit is a pane."""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from geecs_control_room.app import create_app
from geecs_control_room.device_pages import KINDS, registered
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
    assert registered(settings, "scope")[0] == []


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
        ("/device/camera/U_DB", 422, "denied", "U_DB is not a native device"),
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
    assert set(KINDS) == {"camera", "scope", "motor", "supply"}


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
