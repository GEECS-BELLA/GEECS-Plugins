"""The settables page's trends: the hosts the page draws for a family that names trends, and their history from Vole (stubbed with httpx's MockTransport)."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import httpx
import pytest
from fastapi.testclient import TestClient
from geecs_schemas import load_families

from geecs_control_room import trends
from geecs_control_room.app import create_app
from geecs_control_room.device_pages import (
    DEFAULT_WINDOW_S,
    WINDOWS_S,
    drawn_trends,
    window_label,
)
from geecs_control_room.module_client import ModuleClient
from geecs_control_room.settings import Settings

_NS = 1_000_000_000


def _device(family: str, omit: tuple[str, ...] = ()) -> dict[str, Any]:
    """A device as the module's route serves it: every role of ``family`` under ``bench:x:``."""
    roles = [
        {"role": r.name, "axis": None, "label": r.label, "type": r.type, "pv": f"bench:x:{r.suffix}",
         "writePv": f"bench:x:{r.write_suffix}" if r.writable else None, "read": r.read_mode}
        for r in load_families()[family].roles if r.name not in omit
    ]  # fmt: skip
    return {"name": "x", "family": family, "state": "live", "heartbeat": "live", "axes": [], "roles": roles}  # fmt: skip


class Vole:
    """Vole's two routes as the brief gives them: what was asked, and a scripted answer per PV."""

    def __init__(self, samples: Optional[dict[str, list[dict]]] = None, down: bool = False, refuse: int = 0) -> None:  # fmt: skip
        self.samples, self.down, self.refuse = samples or {}, down, refuse
        self.asked: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.asked.append(request)
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        if self.refuse:
            return httpx.Response(self.refuse, json={"detail": "no"})
        path = request.url.path.removeprefix("/vole/v1/pvs/")
        if request.method == "POST":
            return httpx.Response(200, json={"pv": path.removeprefix("subscribe/")})
        found = self.samples.get(path, [])
        return httpx.Response(200, json={"pv": path, "count": len(found), "samples": found})  # fmt: skip


def _room(settings: Settings, body: Optional[dict], vole: Vole) -> TestClient:
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/devices"):
            return httpx.Response(200, json={"devices": []})
        if body is None:
            return httpx.Response(404, json={"code": "not_found", "detail": "no device"})  # fmt: skip
        return httpx.Response(200, json=body)

    module = ModuleClient("http://canopy.test", "k", transport=httpx.MockTransport(answer))  # fmt: skip
    client = trends.VoleClient("http://canopy.test", "read-key", transport=httpx.MockTransport(vole))  # fmt: skip
    (settings.experiment_dir).mkdir(parents=True, exist_ok=True)
    return TestClient(create_app(settings, module, vole=client))


def _sample(t_ns: int, value: Any, severity: str = "NONE") -> dict:
    return {"tsNs": t_ns, "value": value, "severity": severity, "source": "live"}


# ---- the page ---------------------------------------------------------------


def test_a_family_with_trends_draws_a_host_per_trend_and_loads_plotly(settings: Settings) -> None:  # fmt: skip
    html = _room(settings, _device("loop_contract"), Vole()).get("/device/supply/x").text  # fmt: skip
    hosts = re.findall(r'<article class="panel plot-panel trend" data-trend="([^"]*)"><header><span class="eyebrow">([^<]*)<', html)  # fmt: skip
    assert [title for _, title in hosts] == ["Value", "Actuator"]
    lines = [json.loads(roles.replace("&#34;", '"')) for roles, _ in hosts]
    assert [[r["role"] for r in t] for t in lines] == [["value_display", "setpoint_display"], ["actuator", "target"]]  # fmt: skip
    assert lines[1][0] == {"role": "actuator", "label": "actuator", "units": "", "type": "readback"}  # fmt: skip
    assert 'data-trends-url="/api/devices/x/trends"' in html
    assert html.count('class="trend-plot"') == 2
    # the viewer's window: the choices, the default pressed
    pressed = re.findall(r'data-window="(\d+)" aria-pressed="(\w+)">([^<]*)<', html)
    assert pressed == [("120", "false", "2 min"), ("600", "true", "10 min"), ("3600", "false", "1 h")]  # fmt: skip
    # under the readings, in the pane
    assert html.index('class="grid tight readings"') < html.index('class="trends"')
    for script in ("/theme/plot.js", "/static/trend_rules.js", "/static/trend.js"):
        assert f'src="{script}"' in html
    assert len(re.findall(r"/theme/plotly-cartesian-[0-9.]+\.min\.js", html)) == 1


def test_a_family_without_trends_draws_none_and_loads_no_plotly(settings: Settings) -> None:  # fmt: skip
    html = _room(settings, _device("easydriver"), Vole()).get("/device/supply/x").text
    assert 'class="grid tight readings"' in html  # the page itself is drawn
    for absent in ('class="trends"', "trend-plot", "plotly", "trend.js", "trend_rules.js"):  # fmt: skip
        assert absent not in html


def test_a_trend_is_drawn_with_the_roles_the_device_serves() -> None:
    family = load_families()["loop_contract"]
    specs = {r.name: {"label": r.label, "units": "mm" if r.name == "target" else None, "type": r.type} for r in family.roles if r.name not in ("actuator", "value_display", "setpoint_display")}  # fmt: skip
    # an engine not yet serving the actuator: its line is left out; a trend with no role left is dropped
    assert drawn_trends(family, specs) == [{"title": "Actuator", "roles": [{"role": "target", "label": "last target", "units": "mm", "type": "readback"}]}]  # fmt: skip
    assert drawn_trends(load_families()["easydriver"], specs) == []
    assert drawn_trends(None, specs) == []  # a DB device: no family, no trends


def test_the_windows_read_as_their_buttons() -> None:
    assert [window_label(s) for s in WINDOWS_S] == ["2 min", "10 min", "1 h"]
    assert DEFAULT_WINDOW_S in WINDOWS_S


# ---- the route --------------------------------------------------------------


def test_the_route_subscribes_each_trend_pv_and_serves_its_history(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    now = 1_800_000_000 * _NS
    monkeypatch.setattr(trends.time, "time_ns", lambda: now)
    vole = Vole({
        "bench:x:ValueDisplay_RBV": [_sample(now - 2 * _NS, 1.5), _sample(now - 3 * _NS, 1.25), _sample(now - _NS, None), {"value": 9}],
        "bench:x:Actuator_RBV": [_sample(now - _NS, None, "INVALID"), _sample(now - 2 * _NS, True), _sample(now - 3 * _NS, "moving"), _sample(now - 4 * _NS, 7)],
    })  # fmt: skip
    r = _room(settings, _device("loop_contract"), vole).get("/api/devices/x/trends?seconds=120")  # fmt: skip
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    body = r.json()
    ms = now / 1e6
    assert (body["device"], body["to"], body["from"]) == ("x", ms, ms - 120_000)
    assert set(body["roles"]) == {"value_display", "setpoint_display", "actuator", "target"}  # fmt: skip
    # oldest first, in milliseconds; an undefined record's null is a gap, and a sample without a time is dropped
    assert body["roles"]["value_display"] == {"pv": "bench:x:ValueDisplay_RBV", "why": None, "samples": [[ms - 3000, 1.25], [ms - 2000, 1.5], [ms - 1000, None]]}  # fmt: skip
    # an undefined record, a boolean and a word are no number: gaps, never zeros
    assert body["roles"]["actuator"]["samples"] == [[ms - 4000, 7.0], [ms - 3000, None], [ms - 2000, None], [ms - 1000, None]]  # fmt: skip
    assert body["roles"]["target"] == {"pv": "bench:x:Target_RBV", "samples": [], "why": "nothing buffered yet"}  # fmt: skip
    # each PV subscribed, then read over the window, with the read key and nothing else
    posts = sorted(q.url.path for q in vole.asked if q.method == "POST")
    assert posts == sorted(f"/vole/v1/pvs/subscribe/bench:x:{s}" for s in ("ValueDisplay_RBV", "SetpointDisplay_RBV", "Actuator_RBV", "Target_RBV"))  # fmt: skip
    gets = [q for q in vole.asked if q.method == "GET"]
    assert len(gets) == 4 and {q.method for q in vole.asked} == {"GET", "POST"}
    assert all(dict(q.url.params) == {"from": str(now - 120 * _NS), "to": str(now), "maxPoints": str(trends.MAX_POINTS)} for q in gets)  # fmt: skip
    assert all(q.headers["x-api-key"] == "read-key" and not q.content for q in vole.asked)  # fmt: skip
    for pv in {
        q.url.path.rsplit("/", 1)[1] for q in gets
    }:  # subscribed before it is read
        order = [q.method for q in vole.asked if q.url.path.endswith(pv)]
        assert order == ["POST", "GET"]


def test_a_value_that_is_no_finite_number_is_a_gap() -> None:
    got = trends.points([_sample(2_000_000, float("nan")), _sample(1_000_000, float("inf")), _sample(3_000_000, 4), _sample(4_000_000, 5, "INVALID"), "junk", {"tsNs": "soon", "value": 1}])  # fmt: skip
    assert got == [[1.0, None], [2.0, None], [3.0, 4.0], [4.0, None]]


def test_the_window_is_held_to_the_choices(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    now = 1_800_000_000 * _NS
    monkeypatch.setattr(trends.time, "time_ns", lambda: now)
    room = _room(settings, _device("loop_contract"), Vole())
    spans = {s: (b := room.get(f"/api/devices/x/trends?seconds={s}").json())["to"] - b["from"] for s in (0, 600, 10**9)}  # fmt: skip
    assert spans == {0: 1000, 600: 600_000, 10**9: max(WINDOWS_S) * 1000}
    assert room.get("/api/devices/x/trends").json()["from"] == (now - DEFAULT_WINDOW_S * _NS) / 1e6  # fmt: skip


@pytest.mark.parametrize(
    "vole, why",
    [
        (Vole(down=True), "Vole did not answer"),
        (Vole(refuse=403), "Vole answered 403 to the subscribe"),
        (Vole(refuse=500), "Vole answered 500 to the subscribe"),
    ],
)  # fmt: skip
def test_vole_silent_or_refusing_is_words_per_role_never_an_error(settings: Settings, vole: Vole, why: str) -> None:  # fmt: skip
    r = _room(settings, _device("loop_contract"), vole).get("/api/devices/x/trends")
    assert r.status_code == 200
    roles = r.json()["roles"]
    assert len(roles) == 4 and all(v["samples"] == [] and v["why"].startswith(why) for v in roles.values())  # fmt: skip


def test_an_answer_that_is_no_history_is_words(settings: Settings) -> None:
    def odd(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"pv": "x"} if request.method == "GET" else {})

    module = ModuleClient("http://canopy.test", "k", transport=httpx.MockTransport(lambda q: httpx.Response(200, json=_device("loop_contract"))))  # fmt: skip
    vole = trends.VoleClient("http://canopy.test", "k", transport=httpx.MockTransport(odd))  # fmt: skip
    r = TestClient(create_app(settings, module, vole=vole)).get("/api/devices/x/trends")
    assert {v["why"] for v in r.json()["roles"].values()} == {"Vole's answer carried no samples"}  # fmt: skip


def test_only_the_roles_the_device_serves_are_asked(settings: Settings) -> None:
    vole = Vole()
    body = _device("loop_contract", omit=("actuator",))
    r = _room(settings, body, vole).get("/api/devices/x/trends")
    assert set(r.json()["roles"]) == {"value_display", "setpoint_display", "target"}
    assert not any("Actuator" in q.url.path for q in vole.asked)


def test_a_device_without_trends_is_a_404_and_vole_is_never_asked(settings: Settings) -> None:  # fmt: skip
    vole = Vole()
    r = _room(settings, _device("easydriver"), vole).get("/api/devices/x/trends")
    assert r.status_code == 404 and r.json()["detail"] == "x serves no trend role"
    gateway = _device("loop_contract") | {
        "source": "gateway"
    }  # a DB device spelled like a family: no family file's trends
    assert _room(settings, gateway, vole).get("/api/devices/x/trends").status_code == 404  # fmt: skip
    off_roster = _room(settings, None, vole).get("/api/devices/x/trends")
    assert off_roster.status_code == 404 and "not on the module's roster" in off_roster.json()["detail"]  # fmt: skip
    assert vole.asked == []


def test_a_silent_module_is_a_503(settings: Settings) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    module = ModuleClient("http://canopy.test", "k", transport=httpx.MockTransport(down))  # fmt: skip
    vole = Vole()
    client = trends.VoleClient("http://canopy.test", "k", transport=httpx.MockTransport(vole))  # fmt: skip
    r = TestClient(create_app(settings, module, vole=client)).get("/api/devices/x/trends")  # fmt: skip
    assert r.status_code == 503 and r.json()["code"] == "unavailable" and vole.asked == []  # fmt: skip


def test_a_per_axis_family_reads_the_axis_asked() -> None:
    from geecs_schemas import FamilyFile

    doc = load_families()["motor_record"].model_dump(mode="json")
    doc["panel"]["trends"] = [{"title": "Position", "roles": ["rbv"]}]
    family = FamilyFile.model_validate(doc)
    body = {"family": "motor_record", "roles": [{"role": "rbv", "axis": a, "pv": f"bench:{a}.RBV"} for a in ("m1", "m2")]}  # fmt: skip
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("geecs_control_room.device_pages.load_families", lambda: {"motor_record": family})  # fmt: skip
        assert trends.trend_pvs(body, "m2") == {"rbv": "bench:m2.RBV"}
        assert (
            trends.trend_pvs(body, None) == {}
        )  # a per-axis role has no PV without its axis
