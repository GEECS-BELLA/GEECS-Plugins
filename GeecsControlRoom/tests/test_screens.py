"""The renderer: the example screen drawn through one template, misfits named, writes passed through once."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from geecs_web_theme import STATES

from geecs_control_room.app import create_app
from geecs_control_room.module_client import ModuleClient
from geecs_control_room.screens import KIT_STATE
from geecs_control_room.settings import Settings
from tests.conftest import body_of

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sim_bench.yaml"
_PANEL = """devices:
  UC_VisaEBeam6: {family: ad_camera_chain}
  SimMotor1: {family: motor_record, axes: [m1, m2]}
  eztest1: {family: easydriver}
  fpstest1: {family: fastps}
"""


def _screens(settings: Settings, screen: str | None = None) -> None:
    folder = settings.experiment_dir
    (folder / "screens").mkdir(parents=True)
    (folder / "panel_devices.yaml").write_text(_PANEL)
    if screen is None:
        shutil.copy(_EXAMPLE, folder / "screens" / "sim_bench.yaml")
    else:
        (folder / "screens" / "sim_bench.yaml").write_text(screen)


def test_the_example_draws_every_widget_bound(client: TestClient, settings: Settings) -> None:  # fmt: skip
    _screens(settings)
    r = client.get("/screens/sim_bench")
    assert r.status_code == 200
    widgets = re.findall(r'<div class="w"[^>]*>', r.text)
    assert len(widgets) == 21 and all("data-device=" in w for w in widgets)
    # the rail's controls: a motor's target with its readback; a hazard role asks behind the rail's dialog
    assert 'data-readback="rbv"' in r.text and 'id="hazard-dialog"' in r.text
    assert set(re.findall(r'data-tone="([^"]+)"', r.text)) == {"trace-1", "trace-2", "trace-3", "accent"}  # fmt: skip


def test_the_words_are_the_vocabulary_and_wear_kit_states() -> None:
    words = {"live", "stale", "not answering", "unreachable", "off", "error", "unknown"}
    assert set(KIT_STATE) == words and set(KIT_STATE.values()) <= set(STATES)


def test_a_missing_screen_is_404_and_a_misfit_names_itself(client: TestClient, settings: Settings) -> None:  # fmt: skip
    assert client.get("/screens/nope").status_code == 404
    _screens(settings, "title: T\ngroups: [{title: G, at: [1, 1], widgets: [{kind: setpoint, bind: {device: eztest1, role: current}}]}]\n")  # fmt: skip
    r = client.get("/screens/sim_bench")
    assert (
        r.status_code == 422 and "current is a readback, not for a setpoint" in r.text
    )


def test_a_put_goes_once_to_the_module_and_its_answer_comes_back(client: TestClient, sent: list[httpx.Request]) -> None:  # fmt: skip
    body = {"role": "output_sp", "value": 1, "operator": "A. Operator", "confirm": True}
    r = client.post("/api/devices/fps 1/put", json=body)
    assert (r.status_code, r.json()) == (428, {"outcome": "denied", "code": "confirm_required"})  # fmt: skip
    (request,) = sent  # once: a refusal is shown, never retried
    assert request.url.raw_path == b"/geecs/v1/devices/fps%201/put"
    assert request.headers["X-API-Key"] == "read-key-for-tests" and body_of(request) == body  # fmt: skip


def test_an_unreachable_module_is_an_unknown_outcome(settings: Settings) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    module = ModuleClient(
        "http://canopy.test", "k", transport=httpx.MockTransport(down)
    )
    with TestClient(create_app(settings, module)) as c:
        r = c.post(
            "/api/devices/d/put", json={"role": "r", "value": 1, "operator": "A"}
        )
    assert r.status_code == 502
    assert r.json()["outcome"] == "unknown" and r.json()["code"] == "module_unreachable"


_DB_SCREEN = """title: DB
groups:
  - title: Left top
    at: [1, 1]
    columns: 2
    widgets:
      - {kind: flipper, label: Amp2, bind: {pv: "x:u_plc:do_ch3"}, indicators: [{pv: "x:u_plc:di_ch5"}, {pv: "x:u_plc:di_ch6"}]}
      - {kind: annotation, label: "line one\\nline two"}
  - title: Right
    at: [1, 2]
    widgets: [{kind: setpoint, label: Jet X, bind: {pv: "x:u_jet:position_axis_1"}}]
  - title: Left below
    at: [3, 1]
    widgets: [{kind: readout, label: Gauge, bind: {pv: "x:u_gauge:ai"}}]
"""


def test_columns_pack_top_to_bottom_when_no_group_spans_them(client: TestClient, settings: Settings) -> None:  # fmt: skip
    """A LabVIEW panel packs each column: no grid row ties a group's top to the tallest neighbour's."""
    _screens(settings, _DB_SCREEN)
    page = client.get("/screens/sim_bench").text
    stacks = re.findall(r'<div class="scol">(.*?)</div>\s*(?=<div class="scol">|</main>)', page, re.S)  # fmt: skip
    titles = [re.findall(r"<h3>([^<]+)</h3>", s) for s in stacks]
    assert titles == [["Left top", "Left below"], ["Right"]] and "grid-row" not in page
    spanning = _DB_SCREEN.replace("at: [1, 2]", "at: [1, 2]\n    size: [1, 2]")
    _screens_again(settings, spanning)
    page = client.get("/screens/sim_bench").text
    assert 'class="scol"' not in page and "grid-row: 3 / span 1" in page


def test_a_bare_pv_is_drawn_read_only_with_its_lamps_and_columns(client: TestClient, settings: Settings) -> None:  # fmt: skip
    _screens(settings, _DB_SCREEN)
    page = client.get("/screens/sim_bench").text
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in page
    assert re.search(r'<span class="ind" data-pv="x:u_plc:di_ch5">in <b>', page)
    assert re.search(r'<span class="ind" data-pv="x:u_plc:di_ch6">out <b>', page)
    assert (
        page.count('title="A bare PV is read only') == 2
    )  # the flipper and the setpoint
    # disabled in the HTML itself, not only once screen.js has run: the rail's field and the flipper's pair
    assert re.search(
        r'<label for="(w-[^"]+)">Jet X .*?<input type="number" id="\1"[^>]* disabled>',
        page,
        re.S,
    )
    assert re.search(r'data-pair data-enum-label aria-pressed="false" disabled>', page)


def test_an_unlabelled_db_control_is_named_by_its_pv(client: TestClient, settings: Settings) -> None:  # fmt: skip
    _screens(settings, 'title: T\ngroups: [{title: G, at: [1, 1], widgets: [{kind: setpoint, bind: {pv: "x:u_a:b"}}, {kind: button, bind: {pv: "x:u_a:go"}}]}]\n')  # fmt: skip
    page = client.get("/screens/sim_bench").text
    assert (
        re.search(r'<label for="w-[^"]+">x:u_a:b ', page)
        and ">x:u_a:go</button>" in page
    )


def test_every_device_an_indicator_names_is_polled() -> None:
    """A lamp on a device no widget binds is still read: the poll covers indicators' devices (no node test of screen.js yet)."""
    js = (Path(__file__).resolve().parents[1] / "geecs_control_room" / "static" / "screen.js").read_text()  # fmt: skip
    assert 'querySelectorAll(".w[data-device], .ind[data-device]")' in js


def test_the_dense_layout_rules_hold() -> None:
    """Narrow cells: labels shrink and wrap, inputs fit the cell, notes keep their line breaks."""
    css = (Path(__file__).resolve().parents[1] / "geecs_control_room" / "static" / "screen.css").read_text()  # fmt: skip
    assert (
        ".kit .w .field>label{flex:1 1 4.5rem;min-width:0;overflow-wrap:anywhere" in css
    )
    assert "max-width:100%" in css.split(".kit .w .field>input{", 1)[1].split("}", 1)[0]
    assert (
        "white-space:pre-line"
        in css.split(".kit .sgroup .note{", 1)[1].split("}", 1)[0]
    )
    assert ".kit .sgroup{display:grid;" in css


def _screens_again(settings: Settings, screen: str) -> None:
    (settings.experiment_dir / "screens" / "sim_bench.yaml").write_text(screen)


def _roster(names: list[str]) -> ModuleClient:
    def answer(request: httpx.Request) -> httpx.Response:
        devices = [{"name": n, "source": "gateway"} for n in names]
        return httpx.Response(200, json={"experiment": "Bench", "devices": devices})

    return ModuleClient("http://canopy.test", "k", transport=httpx.MockTransport(answer))  # fmt: skip


_ROSTER_SCREEN = "title: T\ngroups: [{title: G, at: [1, 1], widgets: [{kind: toggle, bind: {device: U_PLC, role: DO.Ch1}}]}]\n"  # fmt: skip


def test_a_db_device_on_the_roster_draws_deferred_and_an_unknown_one_is_refused(settings: Settings) -> None:  # fmt: skip
    _screens(settings, _ROSTER_SCREEN)
    with TestClient(create_app(settings, _roster(["U_PLC"]))) as c:
        r = c.get("/screens/sim_bench")
    assert r.status_code == 200
    assert 'data-device="U_PLC" data-role="DO.Ch1"' in r.text and 'data-fits="bit enum"' in r.text  # fmt: skip
    with TestClient(create_app(settings, _roster([]))) as c:
        r = c.get("/screens/sim_bench")
    assert r.status_code == 422 and "no device &#39;U_PLC&#39;" in r.text


def test_concurrent_pages_wait_for_the_one_roster_read() -> None:
    import asyncio

    from geecs_control_room.screens import _Roster

    class Slow:
        reads = 0

        async def devices(self):
            Slow.reads += 1
            await asyncio.sleep(0.05)
            return 200, {"devices": [{"name": "U_PLC"}]}

    async def two():
        roster, module = _Roster(), Slow()
        return await asyncio.gather(roster.names(module), roster.names(module))

    assert asyncio.run(two()) == [(frozenset({"U_PLC"}), None)] * 2 and Slow.reads == 1


def test_a_db_the_module_has_not_read_is_named_when_a_db_device_is_refused(settings: Settings) -> None:  # fmt: skip
    def answer(request: httpx.Request) -> httpx.Response:
        down = {"gateway": {"readAt": None, "error": "GEECS DB unavailable: refused"}}
        return httpx.Response(200, json={"experiment": "Bench", "devices": [], "sources": down})  # fmt: skip

    _screens(settings, _ROSTER_SCREEN)
    module = ModuleClient("http://canopy.test", "k", transport=httpx.MockTransport(answer))  # fmt: skip
    with TestClient(create_app(settings, module)) as c:
        r = c.get("/screens/sim_bench")
    assert (
        r.status_code == 422
        and "has not read the GEECS DB: GEECS DB unavailable" in r.text
    )
    # the cause leads; the widget's own line follows it
    assert r.text.index("roster is incomplete") < r.text.index(
        "no device &#39;U_PLC&#39;"
    )


def test_many_refused_bindings_fold_behind_the_cause(settings: Settings) -> None:
    widgets = ", ".join(f"{{kind: toggle, bind: {{device: U_PLC, role: DO.Ch{i}}}}}" for i in range(5))  # fmt: skip
    _screens(settings, f"title: T\ngroups: [{{title: G, at: [1, 1], widgets: [{widgets}]}}]\n")  # fmt: skip
    with TestClient(create_app(settings, _roster([]))) as c:
        r = c.get("/screens/sim_bench")
    assert r.status_code == 422 and "<summary>5 bindings refused</summary>" in r.text
    assert r.text.count("no device &#39;U_PLC&#39;") == 5


def test_an_undrawable_screen_asks_again_and_a_missing_one_does_not(settings: Settings) -> None:  # fmt: skip
    _screens(settings, _ROSTER_SCREEN)
    with TestClient(create_app(settings, _roster([]))) as c:
        refused, missing = c.get("/screens/sim_bench"), c.get("/screens/nope")
    assert refused.status_code == 422 and 'http-equiv="refresh" content="30"' in refused.text  # fmt: skip
    assert missing.status_code == 404 and "http-equiv" not in missing.text


def _configs_checkout() -> Path | None:
    """The sibling GEECS-Plugins-Configs checkout (``GEECS_PLUGINS_CONFIGS`` first), as GEECS-Schemas' corpus tests find it."""
    import os

    override = os.environ.get("GEECS_PLUGINS_CONFIGS")
    candidates = [Path(override)] if override else [p / "GEECS-Plugins-Configs" for p in Path(__file__).resolve().parents]  # fmt: skip
    return next((c for c in candidates if (c / "scanner_configs").is_dir()), None)


def test_every_deployed_screen_draws_every_widget(tmp_path: Path) -> None:
    """The configs repo's screens (the HTU Control Center) drawn whole: no refusal, one control per bound widget (skips without the checkout)."""
    import yaml

    repo = _configs_checkout()
    if repo is None:
        import pytest

        pytest.skip("sibling GEECS-Plugins-Configs checkout not found")
    for path in sorted((repo / "scanner_configs" / "experiments").glob("*/screens/*.yaml")):  # fmt: skip
        experiment, doc = path.parent.parent.name, yaml.safe_load(path.read_text())
        widgets = [w for g in doc["groups"] for w in g["widgets"]]
        names = sorted({b["device"] for w in widgets for b in [w.get("bind") or {}, *w.get("indicators", [])] if b.get("device")})  # fmt: skip
        folder = tmp_path / "scanner_configs" / "experiments" / experiment / "screens"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            path, folder / path.name
        )  # the screen alone: its devices are the roster's (DB devices)

        roster = {"devices": [{"name": n, "source": "gateway"} for n in names]}

        def answer(request: httpx.Request, roster: dict = roster) -> httpx.Response:
            return httpx.Response(200, json=roster)

        settings = Settings(experiment=experiment, configs_root=tmp_path, canopy_url="http://canopy.test", read_key="k")  # fmt: skip
        app = create_app(settings, ModuleClient("http://canopy.test", "k", transport=httpx.MockTransport(answer)))  # fmt: skip
        page = TestClient(app).get(f"/screens/{path.stem}")
        assert page.status_code == 200, (path.name, page.text[-600:])
        bound = [w for w in widgets if w["kind"] not in ("annotation", "image")]
        assert len(re.findall(r'<div class="w"', page.text)) == len(bound), path.name
        controls = [w for w in bound if not (w.get("bind") or {}).get("action")]
        assert len(re.findall(r'<div class="field', page.text)) == len(controls), (
            path.name
        )  # every control the rail's


def test_a_readout_without_units_of_its_own_takes_the_records(client: TestClient, settings: Settings) -> None:  # fmt: skip
    """No widget or family units: a placeholder the rail fills from Hare's metadata (a gauge's Torr), as a setpoint's is."""
    _screens(settings, 'title: T\ngroups: [{title: G, at: [1, 1], widgets: [{kind: readout, label: Gauge, bind: {pv: "x:u_a:p"}}, {kind: readout, label: Set, bind: {pv: "x:u_a:q"}, units: mbar}]}]\n')  # fmt: skip
    page = client.get("/screens/sim_bench").text
    assert (
        '<label>Gauge <small data-egu="x:u_a:p" data-type="readback"></small></label>'
        in page
    )
    assert (
        "<label>Set <small>mbar</small></label>" in page
    )  # the screen's own units win


def test_a_group_whose_device_the_module_did_not_give_says_so_and_the_screen_draws(settings: Settings) -> None:  # fmt: skip
    """The module silent for an embedded device: that widget says why (drawn on the next load), never a 500 or a blank."""
    reads: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/devices"):
            return httpx.Response(200, json={"devices": [{"name": "U_PLC", "source": "gateway"}]})  # fmt: skip
        reads.append(request.url.path)
        return httpx.Response(503, json={"detail": "busy"})

    _screens(settings, "title: T\ngroups: [{title: G, at: [1, 1], widgets: [{kind: group, bind: {device: U_PLC, group: settings}}, {kind: group, label: Again, bind: {device: U_PLC, group: settings}}, {kind: toggle, bind: {device: U_PLC, role: DO.Ch1}}]}]\n")  # fmt: skip
    module = ModuleClient("http://canopy.test", "k", transport=httpx.MockTransport(answer))  # fmt: skip
    with TestClient(create_app(settings, module)) as c:
        r = c.get("/screens/sim_bench")
    assert r.status_code == 200 and reads == [
        "/geecs/v1/devices/U_PLC"
    ]  # one read per device, however many widgets
    assert r.text.count("the module answered 503: reload the screen to draw it") == 2
    assert '<div class="w gw-missing" data-kind="group"><span class="lbl">Again · U_PLC</span>' in r.text  # fmt: skip
    assert (
        'data-kind="group" data-device' not in r.text
    )  # never polled or written: nothing drawn to write
    assert (
        'data-device="U_PLC" data-role="DO.Ch1"' in r.text
    )  # the rest of the screen drawn
