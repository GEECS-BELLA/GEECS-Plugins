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
    assert 'data-readback="rbv"' in r.text and 'data-hazard="1"' in r.text
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


def test_a_db_device_is_drawn_read_only_with_its_lamps_and_columns(client: TestClient, settings: Settings) -> None:  # fmt: skip
    _screens(settings, _DB_SCREEN)
    page = client.get("/screens/sim_bench").text
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in page
    assert re.search(r'<span class="ind" data-pv="x:u_plc:di_ch5">in <b>', page)
    assert re.search(r'<span class="ind" data-pv="x:u_plc:di_ch6">out <b>', page)
    assert (
        page.count('title="A DB device: its writes stay disabled') == 2
    )  # the flipper and the setpoint


def test_the_dense_layout_rules_hold() -> None:
    """Narrow cells: labels shrink and wrap, inputs fit the cell, notes keep their line breaks."""
    css = (Path(__file__).resolve().parents[1] / "geecs_control_room" / "static" / "screen.css").read_text()  # fmt: skip
    assert ".kit .w .lbl{flex:1 1 6rem;min-width:0;overflow-wrap:anywhere" in css
    assert "max-width:100%" in css.split(".kit .w .sp{", 1)[1].split("}", 1)[0]
    assert (
        "white-space:pre-line"
        in css.split(".kit .sgroup .note{", 1)[1].split("}", 1)[0]
    )
    assert ".kit .sgroup{display:grid;" in css


def _screens_again(settings: Settings, screen: str) -> None:
    (settings.experiment_dir / "screens" / "sim_bench.yaml").write_text(screen)
