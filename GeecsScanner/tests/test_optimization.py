"""Optimizer configuration, preflight merge, live values and explicit best moves."""

from geecs_scanner.service.models import SetBestIn
from geecs_scanner.service.errors import ScannerError
import pytest


def test_config_and_required_merge(client):
    response = client.get("/api/configs/optimizer_configs/xopt_beam_charge")
    assert response.status_code == 200
    assert response.json()["required_devices"] == ["U_BCaveICT"]
    preset = {
        "name": "optimization",
        "devices": [{"device": "U_BCaveICT", "essential": False, "save_images": False}],
        "plan": {
            "name": "optimize",
            "kwargs": {"optimizer_config": "xopt_beam_charge"},
        },
    }
    response = client.post("/api/preflight", json=preset)
    assert response.status_code == 200
    plan = response.json()["plan"]
    assert plan["args"][0] == ["U_BCaveICT"]
    assert plan["kwargs"]["max_iterations"] == 10
    assert "non_essential" not in plan["kwargs"]
    assert "acquisition" not in plan["kwargs"]


def test_stream_values_best_move_and_stale_run_refusal(service, streams, manager):
    streams.on_document(
        "start",
        {
            "uid": "run",
            "plan_name": "optimize",
            "max_iterations": 2,
            "shots_per_step": 5,
            "optimization_move_targets": ["U_S1H:Current"],
        },
    )
    streams.on_document("descriptor", {"uid": "opt", "name": "optimization"})
    streams.on_document(
        "event",
        {
            "descriptor": "opt",
            "data": {
                "iteration": 2,
                "output:cam%2Echarge": float("nan"),
                "measured:Motor:axis%2Evalue": 0.2,
                "best:Motor:axis%2Evalue": 0.25,
                "best_move:U_S1H:Current": 0.25,
            },
        },
    )
    assert service.optimization().outputs == {"cam.charge": None}
    assert service.optimization().measured == {"Motor:axis.value": 0.2}
    assert service.optimization().best == {"Motor:axis.value": 0.25}
    with pytest.raises(ScannerError, match="completed"):
        service.set_optimization_best(SetBestIn(run_uid="run"))
    streams.on_document("stop", {"exit_status": "success"})
    with pytest.raises(ScannerError, match="latest"):
        service.set_optimization_best(SetBestIn(run_uid="older"))
    result = service.set_optimization_best(SetBestIn(run_uid="run"))
    assert result.plan == "mv"
    streams.on_document("start", {"uid": "next", "plan_name": "count"})
    assert service.optimization().run_uid is None


def test_optimization_sse_uses_json_null(client, streams):
    streams.on_document("start", {"uid": "run", "plan_name": "optimize"})
    streams.on_document("descriptor", {"uid": "opt", "name": "optimization"})
    streams.on_document(
        "event", {"descriptor": "opt", "data": {"output:x": float("nan")}}
    )
    response = client.get("/api/events?once=1")
    assert "event: optimization" in response.text
    assert '"x": null' in response.text
    assert "NaN" not in response.text
