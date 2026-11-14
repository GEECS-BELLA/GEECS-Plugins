"""Setups on the surface: capture reads and never writes, the diff and the apply are the scanner's, relayed.

Two layers: a fake scanner (an httpx mock transport recording every request)
for what this service sends and refuses, and the scanner's real demo app for
the routes' shapes (a rename there fails here). A captured setup is saved
through the one save route, so its author, tier and ownership are the
save's (scratch bare repo as origin).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from conftest import git
from fastapi.testclient import TestClient

from geecs_configurator.app import create_app
from geecs_configurator.settings import Settings

SCANNER = "http://scanner.test"
B = "?experiment=Bench"


class FakeScanner:
    """The scanner's routes as a recording mock: readbacks by name, a canned diff and apply."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str, Any]] = []
        self.readings: dict[str, Any] = {"U_PS:Current": 1.25, "U_Jet:Pressure": 2.0}
        self.words: dict[str, str] = {"U_PLC:shutter1": "off"}
        self.apply_answer: tuple[int, Any] = (409, {"error": {"kind": "policy_refusal", "message": "acknowledge", "needs_acknowledgement": [{"check": "setup:abc", "title": "Apply", "message": "m"}], "digest": "abc"}})  # fmt: skip

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.seen.append((request.method, request.url.path, body))
        path, q = request.url.path, request.url.params
        if path == "/api/readback":
            name, word = q["variable"], q.get("word") == "true"
            if word and name in self.words:
                return httpx.Response(200, json={"variable": name, "ok": True, "text": self.words[name]})  # fmt: skip
            if name in self.words:
                return httpx.Response(200, json={"variable": name, "ok": False, "detail": "any words the scanner chooses"})  # fmt: skip
            if name in self.readings:
                return httpx.Response(200, json={"variable": name, "ok": True, "value": self.readings[name]})  # fmt: skip
            return httpx.Response(200, json={"variable": name, "ok": False, "detail": "no answer from the gateway"})  # fmt: skip
        if path == "/api/settables":
            return httpx.Response(200, json={"items": [{"name": "U_PS:Current", "device": "U_PS", "variable": "Current"}], "source": "db"})  # fmt: skip
        if path.endswith("/diff"):
            return httpx.Response(200, json={"name": "low_power", "digest": "abc", "rows": [], "changes": 0, "unread": 0})  # fmt: skip
        if path.endswith("/apply"):
            return httpx.Response(self.apply_answer[0], json=self.apply_answer[1])
        return httpx.Response(404, json={"detail": "no route"})


@pytest.fixture
def fake() -> FakeScanner:
    return FakeScanner()


def _app(settings: Settings, fake: FakeScanner) -> TestClient:
    return TestClient(create_app(settings, consumer_http=httpx.Client(transport=httpx.MockTransport(fake))))  # fmt: skip


@pytest.fixture
def surface(root: Path, fake: FakeScanner) -> TestClient:
    """Read-only (no commit path) with a scanner for Bench."""
    with _app(Settings(configs_root=root, experiment="Bench", scanner_url=SCANNER), fake) as c:  # fmt: skip
        yield c


@pytest.fixture
def committing_surface(origin: Path, fake: FakeScanner, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> TestClient:  # fmt: skip
    """The commit path on over origin, owning Bench, with a scanner for Bench."""
    monkeypatch.setattr("geecs_configurator.app.FRESH_S", -1.0)
    env = {"GEECS_CONFIGURATOR_EXPERIMENTS": "Bench", "GEECS_CONFIGURATOR_STATE_DIR": str(tmp_path_factory.mktemp("service")), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0", "GEECS_CONFIGURATOR_EXPERIMENT": "Bench", "GEECS_CONFIGURATOR_SCANNER_URL": f"Bench={SCANNER},Empty={SCANNER}"}  # fmt: skip
    with _app(Settings.from_env(env), fake) as c:
        yield c


CAPTURE = {"name": "warm", "description": "as it is now", "operator": "Ada", "variables": ["U_PS:Current", "U_PLC:shutter1", "U_Gone:X", "nonsense"]}  # fmt: skip


# ------------------------------------------------------------------ capture


def test_capture_reads_every_variable_and_writes_nothing(surface: TestClient, fake: FakeScanner) -> None:  # fmt: skip
    r = surface.post(f"/api/setups/capture{B}", json=CAPTURE)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["id"] == "warm.yaml"
    steps = out["document"]["steps"]
    assert steps == [
        {"do": "set", "device": "U_PS", "variable": "Current", "value": 1.25},
        {
            "do": "set",
            "device": "U_PLC",
            "variable": "shutter1",
            "value": "off",
        },  # read again as a word
    ]
    captured = out["document"]["captured"]
    assert captured["by"] == "Ada" and captured["readbacks"] == [{"device": "U_PS", "variable": "Current"}, {"device": "U_PLC", "variable": "shutter1"}]  # fmt: skip
    assert [u["variable"] for u in out["unread"]] == ["U_Gone:X", "nonsense"]
    # a read, only: every request to the scanner was a GET of its readback
    assert {(m, p) for m, p, _ in fake.seen} == {("GET", "/api/readback")}


def test_capture_refuses_without_an_operator_a_bad_name_or_any_reading(surface: TestClient, fake: FakeScanner) -> None:  # fmt: skip
    for body, words in (
        ({**CAPTURE, "operator": "  "}, "operator is required"),
        ({**CAPTURE, "variables": []}, "choose the variables"),
        ({**CAPTURE, "name": "warm.yaml"}, "file stem"),
        ({**CAPTURE, "variables": ["U_Gone:X"]}, "nothing to capture"),
    ):
        r = surface.post(f"/api/setups/capture{B}", json=body)
        assert r.status_code == 422 and words in r.text, r.text


def test_no_scanner_no_capture_diff_or_apply(root: Path, fake: FakeScanner) -> None:
    with _app(Settings(configs_root=root, experiment="Bench"), fake) as c:
        assert c.post(f"/api/setups/capture{B}", json=CAPTURE).status_code == 503
        assert c.get(f"/api/setups/low_power/diff{B}").status_code == 503
        assert c.post(f"/api/setups/low_power/apply{B}", json={"operator": "Ada"}).status_code == 503  # fmt: skip
    assert fake.seen == []


def test_a_capture_saves_through_the_save_route_as_the_operators(committing_surface: TestClient, origin: Path) -> None:  # fmt: skip
    draft = committing_surface.post(f"/api/setups/capture{B}", json=CAPTURE).json()
    r = committing_surface.put(f"/api/documents/setup/{draft['id']}{B}", json={"document": draft["document"], "etag": "", "operator": "Ada", "create": True})  # fmt: skip
    assert r.status_code == 200 and r.json()["outcome"] == "committed", r.text
    assert (
        git("log", "-1", "--format=%an", "main", cwd=origin) == "Ada via configurator"
    )
    saved = yaml.safe_load(git("show", "main:scanner_configs/experiments/Bench/setups/warm.yaml", cwd=origin))  # fmt: skip
    assert saved["steps"][0]["value"] == 1.25 and saved["captured"]["by"] == "Ada"
    # a second capture under the same name never replaces the first
    r = committing_surface.put(f"/api/documents/setup/{draft['id']}{B}", json={"document": draft["document"], "etag": "", "operator": "Bo", "create": True})  # fmt: skip
    assert r.status_code == 409


# ------------------------------------------------------------------ the diff and the apply, relayed


def test_the_diff_and_the_settables_are_the_scanners(surface: TestClient, fake: FakeScanner) -> None:  # fmt: skip
    assert surface.get(f"/api/setups/low_power/diff{B}").json()["digest"] == "abc"
    assert (
        surface.get(f"/api/settables{B}").json()["items"][0]["name"] == "U_PS:Current"
    )
    assert [(m, p) for m, p, _ in fake.seen] == [("GET", "/api/setups/low_power/diff"), ("GET", "/api/settables")]  # fmt: skip


def test_the_apply_relays_the_questions_and_never_answers_one(surface: TestClient, fake: FakeScanner) -> None:  # fmt: skip
    r = surface.post(f"/api/setups/low_power/apply{B}", json={"operator": "Ada"})
    assert r.status_code == 409 and r.json() == fake.apply_answer[1]  # verbatim
    fake.apply_answer = (200, {"item_uid": "u1", "submitted_as": "x", "plan": "apply_setup", "summary": "setup low_power"})  # fmt: skip
    r = surface.post(f"/api/setups/low_power/apply{B}", json={"operator": "Ada", "acknowledged": ["setup:abc"]})  # fmt: skip
    assert r.status_code == 200 and r.json()["item_uid"] == "u1"
    # what reached the scanner: exactly the page's answers, nothing added
    assert [b for m, p, b in fake.seen if p.endswith("/apply")] == [
        {"operator": "Ada", "acknowledged": []},
        {"operator": "Ada", "acknowledged": ["setup:abc"]},
    ]


def test_another_boxs_experiment_is_never_applied_from_here(committing_surface: TestClient, fake: FakeScanner) -> None:  # fmt: skip
    r = committing_surface.post("/api/setups/low_power/apply?experiment=Empty", json={"operator": "Ada"})  # fmt: skip
    assert r.status_code == 403 and "not this services box's" in r.text
    assert fake.seen == []


def test_a_read_only_box_naming_what_it_owns_applies_only_those(root: Path, fake: FakeScanner) -> None:  # fmt: skip
    """No commit path, but EXPERIMENTS named: the other scanner its site names is never reached; the page's Apply is off."""
    other = root / "scanner_configs/experiments/Empty/setups"
    other.mkdir(parents=True)
    (other / "low_power.yaml").write_text(
        "name: low_power\nsteps:\n  - {do: wait, seconds: 1}\n"
    )
    env = {"GEECS_CONFIGS_ROOT": str(root), "GEECS_EXPERIMENT": "Bench", "GEECS_CONFIGURATOR_EXPERIMENTS": "Bench", "GEECS_CONFIGURATOR_SCANNER_URL": f"Bench={SCANNER},Empty={SCANNER}"}  # fmt: skip
    with _app(Settings.from_env(env), fake) as c:
        assert c.post("/api/setups/low_power/apply?experiment=Empty", json={"operator": "Ada"}).status_code == 403  # fmt: skip
        assert fake.seen == []
        assert c.post(f"/api/setups/low_power/apply{B}", json={"operator": "Ada"}).status_code == 409  # its questions  # fmt: skip
        page = c.get("/?experiment=Empty&kind=setup&id=low_power.yaml").text
        assert 'id="live-apply" disabled' in page


def test_a_silent_scanner_is_a_502(root: Path) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with TestClient(create_app(Settings(configs_root=root, experiment="Bench", scanner_url=SCANNER), consumer_http=httpx.Client(transport=httpx.MockTransport(down)))) as c:  # fmt: skip
        assert c.get(f"/api/setups/low_power/diff{B}").status_code == 502
        r = c.post(f"/api/setups/capture{B}", json=CAPTURE)
        assert r.status_code == 422 and "nothing to capture" in r.text


# ------------------------------------------------------------------ against the scanner's real demo app


@pytest.fixture
def demo(root: Path, tmp_path: Path) -> TestClient:
    """This app, its scanner the scanner's demo app (stepped by hand)."""
    from geecs_scanner.service import ScannerService, StreamCache
    from geecs_scanner.service.demo import DemoQueueClient, DemoReadback, DemoResolver, demo_preflight  # fmt: skip
    from geecs_scanner.web import create_app as scanner_app

    streams = StreamCache(clock=lambda: 1_000.0)
    manager = DemoQueueClient(streams, period=0.0, user="contract")
    service = ScannerService(manager, DemoResolver(), experiment="Demo", identity="contract", streams=streams, preflight=demo_preflight, readback=DemoReadback(manager), heartbeat_path=tmp_path / "hb.json")  # fmt: skip
    with TestClient(scanner_app(service)) as scanner:
        settings = Settings(configs_root=root, experiment="Bench", scanner_url=str(scanner.base_url))  # fmt: skip
        with TestClient(create_app(settings, consumer_http=scanner)) as c:
            yield c


def test_capture_diff_and_apply_against_the_scanners_routes(demo: TestClient) -> None:
    out = demo.post(f"/api/setups/capture{B}", json={**CAPTURE, "variables": ["U_S1H:current"]}).json()  # fmt: skip
    assert out["document"]["steps"][0]["value"] == 0.0 and out["unread"] == []
    d = demo.get(f"/api/setups/low_power/diff{B}").json()
    assert {r["variable"] for r in d["rows"]} == {"current", "Jet pressure", "shutter1", "shutter2"}  # fmt: skip
    r = demo.post(f"/api/setups/low_power/apply{B}", json={"operator": "Ada"})
    asked = [q["check"] for q in r.json()["error"]["needs_acknowledgement"]]
    assert r.status_code == 409 and asked[0].startswith("setup:")
    r = demo.post(f"/api/setups/low_power/apply{B}", json={"operator": "Ada", "acknowledged": asked})  # fmt: skip
    assert r.status_code == 200 and r.json()["plan"] == "apply_setup"


def test_the_scanners_readback_reads_a_word(demo: TestClient) -> None:
    from geecs_configurator.apply import ScannerClient

    scanner: ScannerClient = demo.app.state.consumers.scanners["Bench"]
    status, body = scanner.readback("U_148_PLC:shutter1", word=True)
    assert status == 200 and body["text"] == "off"
