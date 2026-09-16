"""Preview uses the execution expansion without hardware and stays bounded."""

import json
import subprocess
from pathlib import Path

import pytest
from geecs_schemas import Sweep

from geecs_scanner.service import trajectory
from geecs_scanner.service.errors import ScannerError


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
    [("TIMEOUT", 0.001, "time budget"), ("MAX_RSS", 1, "memory budget")],
)
def test_expansion_process_is_terminated_at_resource_budget(
    monkeypatch, budget, value, message
):
    monkeypatch.setattr(trajectory, budget, value)
    with pytest.raises(ScannerError, match=message):
        trajectory.preview(Sweep.model_validate(payload()))


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
