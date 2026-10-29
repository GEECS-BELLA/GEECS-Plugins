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
