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
