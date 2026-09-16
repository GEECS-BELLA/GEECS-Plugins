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
                "output:cam~2Echarge": float("nan"),
                "measured:Motor:axis~2Evalue": 0.2,
                "best:Motor:axis~2Evalue": 0.25,
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
    manager.step()
    assert manager.history_items()[-1]["args"] == ["U_S1H.current", 0.25]
    assert service.optimization().invalidated_reason
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


@pytest.fixture
def completed_best(streams):
    streams.on_document(
        "start", {"uid": "run", "plan_name": "optimize", "scan_number": 42}
    )
    streams.on_document("descriptor", {"uid": "opt", "name": "optimization"})
    streams.on_document(
        "event",
        {
            "descriptor": "opt",
            "data": {
                "best_move:U_S1H:Current": 0.25,
                "best_move:U_S2H:Current": -0.5,
                "best:bump": 999,
            },
        },
    )
    streams.on_document("stop", {"run_start": "run", "exit_status": "success"})


def test_best_uses_every_physical_component(service, manager, completed_best, caplog):
    import logging

    with caplog.at_level(logging.INFO):
        service.set_optimization_best(
            SetBestIn(run_uid="run", operator="operator-test")
        )
    manager.step()
    assert manager.history_items()[-1]["args"] == [
        "U_S1H.current",
        0.25,
        "U_S2H.current",
        -0.5,
    ]
    assert "operator-test" in caplog.text


@pytest.mark.parametrize("status", ["abort", "fail"])
def test_failed_run_cannot_set_best(service, streams, completed_best, status):
    streams.on_document("stop", {"run_start": "run", "exit_status": status})
    assert service.optimization().exit_status == status
    with pytest.raises(ScannerError, match="successfully"):
        service.set_optimization_best(SetBestIn(run_uid="run"))


def test_expired_best_is_refused_and_reported(
    service, streams, completed_best, monkeypatch
):
    monkeypatch.setattr(streams, "_clock", lambda: 1900.0)
    result = service.optimization()
    assert result.expired and result.completed_at == 1000 and result.scan_number == 42
    with pytest.raises(ScannerError, match="expired"):
        service.set_optimization_best(SetBestIn(run_uid="run"))


@pytest.mark.parametrize("plan", ["mv", "run_action"])
def test_non_run_item_invalidates_best(service, manager, completed_best, plan):
    service._queue_item(
        plan, ["U_S1H.current", 0.0] if plan == "mv" else ["test"], {}, what="test"
    )
    manager.step()
    with pytest.raises(ScannerError, match="hardware changed"):
        service.set_optimization_best(SetBestIn(run_uid="run"))


def test_saving_optimizer_draft_does_not_resolve_or_merge(
    service, monkeypatch, tmp_path
):
    from geecs_scanner.service.models import SavePresetIn

    captured = []
    monkeypatch.setattr(
        service.resolver,
        "resolve_optimizer_config",
        lambda _: pytest.fail("saving resolved an optimizer"),
    )
    monkeypatch.setattr(
        service.resolver,
        "write_preset",
        lambda p, **kw: captured.append(p) or tmp_path / "draft.yaml",
    )
    document = {
        "name": "draft",
        "devices": [{"device": "Meter", "essential": False, "save_images": False}],
        "plan": {
            "name": "optimize",
            "kwargs": {"optimizer_config": "renamed", "acquisition": "strict"},
        },
    }
    service.save_preset("draft", SavePresetIn(preset=document))
    assert captured[0].plan.kwargs == document["plan"]["kwargs"]
    assert (
        not captured[0].devices[0].essential and not captured[0].devices[0].save_images
    )
