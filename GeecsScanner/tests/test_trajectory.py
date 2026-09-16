"""Preview uses the execution expansion without hardware and stays bounded."""

import json
import threading
import subprocess
from pathlib import Path

import pytest
from geecs_schemas import Sweep
from geecs_web_theme.testing import node_available

from geecs_scanner.service import trajectory
from geecs_scanner.service.errors import ScannerError


def curved_payload():
    return {
        "trajectory": {
            "kind": "spiral",
            "x": {"axis": "X"},
            "y": {"axis": "Y"},
            "x_center": 0,
            "y_center": 0,
            "x_range": 4,
            "y_range": 4,
            "dr": 1,
            "nth": 8,
        }
    }


@pytest.mark.parametrize("length", [None, "1", str(trajectory.MAX_INPUT_BYTES + 1)])
def test_oversized_raw_body_is_refused_before_validation(
    client, service, monkeypatch, length
):
    def unexpected(*a, **kw):
        pytest.fail("oversized request reached service validation")

    monkeypatch.setattr(service, "trajectory", unexpected)
    headers = {"content-type": "application/json"}
    if length is not None:
        headers["content-length"] = length
    response = client.post(
        "/api/trajectory",
        content=iter([b" " * trajectory.MAX_INPUT_BYTES, b"x"]),
        headers=headers,
    )
    assert response.status_code == 400
    assert "256 KiB" in response.json()["error"]["message"]


@pytest.mark.parametrize("body", [b"not json", b"[]", b"\xff"])
def test_invalid_raw_json_uses_error_taxonomy(client, body):
    response = client.post("/api/trajectory", content=body)
    assert response.status_code == 400
    assert response.json()["error"]["kind"] == "invalid_request"


def test_bounded_axis_preview_needs_no_child(monkeypatch):
    monkeypatch.setattr(
        trajectory.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("cold child for bounded axes"),
    )
    assert trajectory.preview(Sweep.model_validate(payload())).total_steps == 3


@pytest.mark.parametrize(
    "output,code", [("library chatter\n{}", 0), ("[]", 0), ('{"axes": []}', 0), ("", 1)]
)
def test_child_protocol_failures_are_legible_and_do_not_leak_stderr(
    monkeypatch, caplog, output, code
):
    import os

    class Child:
        pid = os.getpid()
        returncode = code

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def communicate(self, **kwargs):
            return output, "Traceback /private/host/path.py: secret detail"

        def poll(self):
            return self.returncode

    monkeypatch.setattr(trajectory.subprocess, "Popen", lambda *a, **kw: Child())
    with pytest.raises(ScannerError) as error:
        trajectory.preview(Sweep.model_validate(curved_payload()))
    assert error.value.kind == "invalid_request"
    assert "Trajectory calculation" in str(error.value)
    assert "/private" not in str(error.value)
    if code:
        assert "/private/host/path.py" in caplog.text


def test_cancelled_curved_preview_kills_child_and_releases_slot(monkeypatch):
    children = []
    popen = trajectory.subprocess.Popen

    def launch(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(trajectory.subprocess, "Popen", launch)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(ScannerError, match="cancelled"):
        trajectory.preview(Sweep.model_validate(curved_payload()), cancelled)
    assert children[0].poll() is not None
    assert trajectory._SLOTS.acquire(blocking=False)
    trajectory._SLOTS.release()


def test_preview_child_import_does_not_load_queue_client():
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import geecs_scanner.service.trajectory; assert 'geecs_scanner.service.scanner' not in sys.modules; assert 'geecs_bluesky.qs_client' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_long_thin_square_spiral_stays_behind_process_timeout(monkeypatch):
    body = curved_payload()
    body["trajectory"] = {
        "kind": "spiral_square",
        "x": {"axis": "X"},
        "y": {"axis": "Y"},
        "x_center": 0,
        "y_center": 0,
        "x_range": 4,
        "y_range": 4,
        "x_num": 62500,
        "y_num": 2,
    }
    monkeypatch.setattr(trajectory, "TIMEOUT", 0.001)
    with pytest.raises(ScannerError, match="timed out"):
        trajectory.preview(Sweep.model_validate(body))


def test_high_axis_count_stays_behind_process_timeout(monkeypatch):
    monkeypatch.setattr(trajectory, "TIMEOUT", 0.001)
    with pytest.raises(ScannerError, match="timed out"):
        trajectory.preview(Sweep.model_validate(payload(500, 1)))


def test_route_disconnect_cancels_service_preview(service, monkeypatch):
    import asyncio
    from fastapi import APIRouter
    from starlette.requests import Request
    from geecs_scanner.web.api import register

    def wait_for_disconnect(body, cancelled):
        assert body == payload()
        assert cancelled.wait(2), "disconnect was not propagated"
        return trajectory._expand(Sweep.model_validate(body))

    monkeypatch.setattr(service, "trajectory", wait_for_disconnect)
    router = APIRouter()
    register(router, service)
    endpoint = next(r.endpoint for r in router.routes if r.path == "/api/trajectory")
    messages = iter(
        [
            {
                "type": "http.request",
                "body": json.dumps(payload()).encode(),
                "more_body": False,
            },
            {"type": "http.disconnect"},
        ]
    )

    async def receive():
        return next(messages)

    async def run():
        request = Request({"type": "http", "headers": []}, receive)
        result = await endpoint(request)
        assert result.total_steps == 3

    asyncio.run(run())


def payload(axes=1, count=3):
    return {
        "trajectory": {
            "kind": "axes",
            "axes": [
                {
                    "kind": "range",
                    "axis": f"Axis{i}",
                    "start": -1,
                    "stop": 1,
                    "num": count,
                    "relative": i == 0,
                }
                for i in range(axes)
            ],
        }
    }


def test_five_axes_use_shared_python_coordinates(client):
    response = client.post("/api/trajectory", json=payload(5))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["total_steps"] == 3 and not result["sampled"]
    assert result["indices"] == [0, 1, 2]
    assert [a["axis"] for a in result["axes"]] == [f"Axis{i}" for i in range(5)]
    assert result["axes"][0]["relative"]
    assert all(a["positions"] == [-1, 0, 1] for a in result["axes"])


@pytest.mark.parametrize("stop", [2, 5])
def test_single_position_range_visits_only_start(client, stop):
    body = payload(count=1)
    body["trajectory"]["axes"][0].update(start=2, stop=stop)
    response = client.post("/api/trajectory", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["total_steps"] == 1
    assert result["axes"][0]["positions"] == [2]


def test_lists_preserve_repeats_and_reject_unequal_lengths(client):
    body = {
        "trajectory": {
            "kind": "axes",
            "axes": [
                {"kind": "list", "axis": "A", "positions": [2, 1, 2]},
                {"kind": "list", "axis": "B", "positions": [4, 5, 6]},
            ],
        }
    }
    assert client.post("/api/trajectory", json=body).json()["axes"][0]["positions"] == [
        2,
        1,
        2,
    ]
    body["trajectory"]["axes"][1]["positions"].pop()
    response = client.post("/api/trajectory", json=body)
    assert response.status_code == 400
    assert "equal point counts" in response.json()["error"]["message"]


def test_sampled_preview_discloses_count_and_retains_endpoints(client):
    result = client.post("/api/trajectory", json=payload(count=3000)).json()
    assert result["sampled"] and result["total_steps"] == 3000
    assert len(result["indices"]) <= 2000
    assert result["indices"][0] == 0 and result["indices"][-1] == 2999
    assert result["axes"][0]["positions"][0] == -1
    assert result["axes"][0]["positions"][-1] == 1


def test_product_budget_is_checked_before_expansion(monkeypatch):
    body = payload(5, 1000)
    body["trajectory"]["combine"] = "product"

    def no_process(*a, **kw):
        pytest.fail("should reject before allocation")

    monkeypatch.setattr(trajectory.subprocess, "Popen", no_process)
    with pytest.raises(ScannerError, match="coordinate preview budget"):
        trajectory.preview(Sweep.model_validate(body))


@pytest.mark.parametrize(
    "budget,value,message",
    [("TIMEOUT", 0.001, "timed out"), ("MAX_RSS", 1, "memory budget")],
)
def test_expansion_process_is_terminated_at_resource_budget(
    monkeypatch, budget, value, message
):
    monkeypatch.setattr(trajectory, budget, value)
    with pytest.raises(ScannerError, match=message):
        trajectory.preview(Sweep.model_validate(curved_payload()))


def test_busy_preview_refuses_without_waiting():
    trajectory._SLOTS.acquire()
    trajectory._SLOTS.acquire()
    try:
        with pytest.raises(ScannerError, match="busy"):
            trajectory.preview(Sweep.model_validate(payload()))
    finally:
        trajectory._SLOTS.release()
        trajectory._SLOTS.release()


def test_numeric_list_parser_accepts_delimiters_and_rejects_expressions():
    if not node_available():
        pytest.skip("Node is not installed")
    script = (
        Path(__file__).resolve().parents[1] / "geecs_scanner/static/sweep-composer.js"
    )
    harness = (
        "var window = {};\n"
        + script.read_text()
        + """
const parse = window.GEECS_SWEEP.parseList;
const good = parse("1, -2.5\\t3e2\\n1");
const bad = ["1,,2", "1,", "NaN", "Infinity", "np.arange(3)", "0x12", "1e999"].map(v => { try {parse(v); return false;} catch(e) {return true;} });
console.log(JSON.stringify({good, bad}));
"""
    )
    result = subprocess.run(
        ["node", "-"], input=harness, text=True, capture_output=True, check=True
    )
    data = json.loads(result.stdout)
    assert data["good"] == [1, -2.5, 300, 1]
    assert all(data["bad"])


@pytest.mark.parametrize(
    "kind,params",
    [
        (
            "spiral",
            {
                "x_center": 0,
                "y_center": 0,
                "x_range": 4,
                "y_range": 4,
                "dr": 1,
                "nth": 8,
            },
        ),
        (
            "spiral_fermat",
            {
                "x_center": 0,
                "y_center": 0,
                "x_range": 4,
                "y_range": 4,
                "dr": 1,
                "factor": 1,
            },
        ),
        (
            "spiral_square",
            {
                "x_center": 0,
                "y_center": 0,
                "x_range": 4,
                "y_range": 4,
                "x_num": 3,
                "y_num": 3,
            },
        ),
        ("x2x", {"start": -2, "stop": 2, "num": 3}),
    ],
)
def test_pattern_preview_matches_execution_coordinates(client, kind, params):
    from geecs_bluesky.trajectory import sweep_to_cycler

    body = {
        "trajectory": {
            "kind": kind,
            "x": {"axis": "X", "relative": True},
            "y": {"axis": "Y", "relative": True},
            **params,
        }
    }
    expected = sweep_to_cycler(Sweep.model_validate(body), str).by_key()
    response = client.post("/api/trajectory", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert not result["sampled"]
    assert result["total_steps"] == len(expected["X"])
    assert [a["positions"] for a in result["axes"]] == [expected["X"], expected["Y"]]
