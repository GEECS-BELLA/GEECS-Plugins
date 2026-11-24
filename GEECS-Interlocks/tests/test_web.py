"""The web client: its routes over a real engine, the stream, and the kit's template guards."""

import json
from pathlib import Path

import pytest
import yaml
from conftest import FakeSource, threshold
from fastapi.testclient import TestClient
from geecs_web_theme import PANE_STATES, STATES
from geecs_web_theme.testing import (
    bare_url_for_calls,
    inline_scripts,
    unknown_data_states,
)

from geecs_interlocks.config import Settings
from geecs_interlocks.engine import Engine
from geecs_interlocks.web import create_app

WEB = Path(__file__).parent.parent / "geecs_interlocks" / "web"
CAM = {
    "kind": "threshold",
    "device": "Cam",
    "variable": "Max",
    "operator": ">=",
    "limit": 4000,
}
DRIFT = {
    "kind": "alignment",
    "device": "Cam",
    "variable": "X",
    "target_variable": "X0",
    "tolerance": 5,
}


@pytest.fixture
def engine(tmp_path):
    path = tmp_path / "main.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "stale_after_s": None,
                "interlocks": [
                    threshold(),
                    {"name": "cam", "conditions": [CAM, DRIFT]},
                ],
            }
        )
    )

    def make(_doc):
        source = FakeSource()
        source.push("Gauge", P=1.0)
        source.push("Cam", Max=4095, X=1, X0=0)
        return source

    eng = Engine(path, make, Settings(status_port=0))
    eng.start()
    eng.monitor.step()
    yield eng
    eng.stop()


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine, experiment="Demo"))


class TestRoutes:
    def test_status_worst_first_with_kit_words(self, client):
        data = client.get("/api/status").json()
        assert data["headline"] == "Tripped" and data["kit_state"] == "failed"
        names = [s["name"] for s in data["statuses"]]
        assert names == ["cam", "vac"]
        cam = data["statuses"][0]
        assert cam["kit_state"] == "failed" and len(cam["checks"]) == 2
        assert cam["checks"][0]["unsafe"] and not cam["checks"][1]["unsafe"]
        assert {s["kit_state"] for s in data["statuses"]} <= set(STATES)

    def test_trend_one_series_per_check(self, client):
        data = client.get("/api/trend", params={"name": "cam"}).json()
        assert [s["limit"] for s in data["series"]] == [4000, 5]
        assert data["series"][0]["v"][-1] == 4095

    def test_trend_of_an_unknown_interlock(self, client):
        assert client.get("/api/trend", params={"name": "nope"}).status_code == 404

    def test_events_newest_first_and_by_name(self, client):
        events = client.get("/api/events").json()
        assert events and events[0]["at"] >= events[-1]["at"]
        only = client.get("/api/events", params={"name": "vac"}).json()
        assert {e["name"] for e in only} == {"vac"}

    def test_the_page(self, client):
        page = client.get("/")
        assert page.status_code == 200
        assert 'class="chip xl"' in page.text and "interlocks.js" in page.text
        assert client.get("/static/interlocks.js").status_code == 200
        assert client.get("/theme/kit.css").status_code == 200

    def test_nothing_writes(self, client):
        for route in ("/api/status", "/api/trend", "/api/events", "/api/stream"):
            assert client.post(route).status_code == 405


def test_the_stream_sends_the_snapshot(engine):
    from geecs_interlocks.web import snapshot_json

    data = json.loads(json.dumps(snapshot_json(engine.snapshot(), engine)))
    assert data["statuses"][0]["name"] == "cam"
    assert data["configuration"] == "main"


class TestTemplateGuards:
    def test_every_url_goes_through_root(self):
        assert bare_url_for_calls(WEB / "templates" / "index.html") == []

    @pytest.mark.parametrize("name", ["templates/index.html", "static/interlocks.js"])
    def test_only_known_states(self, name):
        assert unknown_data_states(WEB / name, [*STATES, *PANE_STATES]) == []

    def test_no_inline_scripts(self):
        assert inline_scripts(WEB / "templates" / "index.html") == []


def test_a_stalled_snapshot_is_never_reported_safe(engine):
    """/api/status follows the stream's rule: an old snapshot is no snapshot."""
    import dataclasses
    import time as _time

    from geecs_interlocks.monitor import InterlockStatus, Snapshot
    from geecs_interlocks.web import snapshot_json

    safe = Snapshot((InterlockStatus("a", "", "", "ok"),), mono=_time.monotonic() - 5)
    assert safe.safe
    data = snapshot_json(safe, engine)
    assert data["safe"] is False and data["stalled"] is True
    assert data["headline"] == "Not updating" and data["kit_state"] == "failed"
    fresh = dataclasses.replace(safe, mono=_time.monotonic())
    assert snapshot_json(fresh, engine)["safe"] is True


HOLD = {"kind": "config_change", "device": "W", "variables": ["Pos"]}


@pytest.fixture
def armable(tmp_path):
    path = tmp_path / "main.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "stale_after_s": None,
                "interlocks": [
                    threshold(),
                    {"name": "stage", "conditions": [HOLD]},
                ],
            }
        )
    )

    def make(_doc):
        source = FakeSource()
        source.push("Gauge", P=1.0)
        source.push("W", Pos=3.0)
        return source

    eng = Engine(path, make, Settings(status_port=0))
    eng.start()
    eng.monitor.step()
    yield eng, path, TestClient(create_app(eng))
    eng.stop()


class TestActions:
    def test_an_action_needs_an_operator(self, armable):
        eng, _, client = armable
        for path in ("/api/arm", "/api/reload"):
            for body in ({}, {"operator": "  "}):
                r = client.post(path, json=body)
                assert r.status_code == 400 and "operator" in r.json()["detail"]
        assert eng.monitor.baseline("stage", 0) is None  # nothing was armed

    def test_an_action_not_sent_as_json_is_refused(self, armable):
        """A cross-origin text/plain POST (no preflight) never reaches the engine."""
        eng, _, client = armable
        body = '{"operator": "Ada"}'
        for path in ("/api/arm", "/api/reload"):
            r = client.post(path, content=body, headers={"Content-Type": "text/plain"})
            assert r.status_code == 415
        assert eng.monitor.baseline("stage", 0) is None

    def test_arm_all_records_the_settings_and_logs_who(self, armable, caplog):
        eng, _, client = armable
        caplog.set_level("INFO")
        r = client.post("/api/arm", json={"operator": "Ada"})
        assert r.status_code == 200
        assert r.json() == {"ok": True, "message": "armed stage", "armed": ["stage"]}
        assert eng.monitor.baseline("stage", 0) == {"Pos": 3.0}
        assert "armed stage by Ada" in caplog.text

    def test_arm_one_by_name(self, armable):
        _, _, client = armable
        r = client.post("/api/arm", json={"operator": "Ada", "interlock": "stage"})
        assert r.status_code == 200 and r.json()["armed"] == ["stage"]

    def test_arm_refused_is_409_with_the_engines_words(self, armable, caplog):
        eng, _, client = armable
        r = client.post("/api/arm", json={"operator": "Ada", "interlock": "vac"})
        assert r.status_code == 409 and not r.json()["ok"]
        assert "holds no settings" in r.json()["message"]
        assert "by Ada refused" in caplog.text

    def test_arm_refused_while_a_setting_has_no_value(self, armable):
        eng, _, client = armable
        eng.monitor._source.drop("W")
        r = client.post("/api/arm", json={"operator": "Ada"})
        assert r.status_code == 409 and "cannot arm" in r.json()["message"]
        assert eng.monitor.baseline("stage", 0) is None

    def test_reload_swaps_the_monitor_and_drops_the_arming(self, armable):
        eng, _, client = armable
        client.post("/api/arm", json={"operator": "Ada"})
        before = eng.monitor
        r = client.post("/api/reload", json={"operator": "Ada"})
        assert r.status_code == 200 and r.json()["message"].startswith("reloaded 2")
        assert eng.monitor is not before
        assert eng.monitor.baseline("stage", 0) is None

    def test_reload_refused_keeps_the_running_interlocks(self, armable):
        eng, path, client = armable
        before = eng.monitor
        path.write_text("schema_version: 1\ninterlocks: [{name: x}]\n")
        r = client.post("/api/reload", json={"operator": "Ada"})
        assert r.status_code == 409 and r.json()["message"].startswith("reload refused")
        assert eng.monitor is before

    def test_page_carries_the_operator_and_the_actions(self, armable):
        _, _, client = armable
        html = client.get("/").text
        for marker in (
            'id="operator"',
            'id="btn-arm"',
            'id="btn-reload"',
            'id="btn-arm-one"',
            '<dialog id="confirm"',
        ):
            assert marker in html
