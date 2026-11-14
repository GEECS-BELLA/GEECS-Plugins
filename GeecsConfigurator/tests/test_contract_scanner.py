"""The configurator and the scanner, each against the other's real app (the scanner's demo manager).

A route either side calls, renamed or reshaped, fails here in the same PR:
the reopen, the trigger-profile dry run and the readback (configurator →
scanner), and Save as preset (scanner → configurator). The scanner is a
dev-only dependency; the package never imports it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from geecs_configurator.apply import ScannerClient, ServiceApplier
from geecs_configurator.log import Row

B = "?experiment=Bench"


@pytest.fixture
def scanner(tmp_path: Path) -> Any:
    """The scanner's app over its demo manager, stepped by hand."""
    from geecs_scanner.service import ScannerService, StreamCache
    from geecs_scanner.service.demo import DemoQueueClient, DemoReadback, DemoResolver, demo_preflight  # fmt: skip
    from geecs_scanner.web import create_app as scanner_app

    streams = StreamCache(clock=lambda: 1_000.0)
    manager = DemoQueueClient(streams, period=0.0, user="contract")
    service = ScannerService(manager, DemoResolver(), experiment="Demo", identity="contract", streams=streams, preflight=demo_preflight, readback=DemoReadback(manager), heartbeat_path=tmp_path / "hb.json")  # fmt: skip
    with TestClient(scanner_app(service)) as client:
        yield client, manager


def _row(kind: str, path: str) -> Row:
    return Row(source="saved", who="Ada", kind=kind, path=path, state="queued")


def _client(scanner: Any) -> ScannerClient:
    http, _ = scanner
    return ScannerClient(str(http.base_url), http)


def test_the_reopen_applies_through_the_scanners_route(
    scanner: Any, tmp_path: Path
) -> None:
    applier = ServiceApplier(tmp_path, scanners={"Demo": _client(scanner)})
    (answer,) = applier.settle([_row("trigger_profile", "scanner_configs/experiments/Demo/shot_control_configurations/x.yaml")])  # fmt: skip
    assert answer.state == "applied", answer.detail
    assert scanner[1].reopens == 1


def test_a_busy_scanner_keeps_the_apply_queued(scanner: Any, tmp_path: Path) -> None:
    http, manager = scanner
    manager._queue.append({"name": "count", "args": [], "kwargs": {}, "item_uid": "w"})
    applier = ServiceApplier(tmp_path, scanners={"Demo": _client(scanner)})
    (answer,) = applier.settle([_row("scan_variables", "scanner_configs/experiments/Demo/scan_devices/scan_variables.yaml")])  # fmt: skip
    assert answer.state == "queued" and "idle" in answer.detail and manager.reopens == 0


def test_the_trigger_profile_dry_run_is_the_scanners(scanner: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    monkeypatch.setattr("geecs_bluesky.devices.ca.liveness.probe_disconnected", lambda e, d, *, timeout: [])  # fmt: skip
    good = {"name": "p", "states": {"SCAN": [{"device": "U_DG645", "variable": "Amplitude.Ch AB", "value": "4.0"}]}}  # fmt: skip
    out = _client(scanner).trigger_profile_dry_run(good)
    assert out.ran and out.refusal is None and out.outcomes[0]["check"] == "validate"
    out = _client(scanner).trigger_profile_dry_run({"name": "p", "states": {}})
    assert out.refusal and "names no trigger device" in out.refusal


def test_the_readback_is_the_scanners(scanner: Any) -> None:
    status, body = _client(scanner).readback("U_S1H:current")
    assert status == 200 and "value" in body
    # the capture's word re-read keys on this flag (setups._read): renamed there, every word goes unread
    assert body["not_numeric"] is False


def test_save_as_preset_commits_through_the_configurators_route(committing: TestClient, origin: Path) -> None:  # fmt: skip
    """The scanner's saver against this app: a create, then a replace only with overwrite."""
    from conftest import git
    from geecs_scanner.service.errors import ScannerError
    from geecs_scanner.service.preset_saver import ConfiguratorPresets

    def send(method: str, url: str, data: bytes | None) -> tuple[int, bytes]:
        r = committing.request(method, url, content=data, headers={"Content-Type": "application/json"})  # fmt: skip
        return r.status_code, r.content

    saver = ConfiguratorPresets(str(committing.base_url), "Bench", send=send)
    doc = {"schema_version": 1, "name": "jet", "devices": [], "plan": {"name": "count", "kwargs": {"num": 2}}}  # fmt: skip
    out = saver.save("jet", doc, operator="Ada", overwrite=False)
    assert out.commit and git("log", "-1", "--format=%an", "main", cwd=origin) == "Ada via configurator"  # fmt: skip
    with pytest.raises(ScannerError) as err:
        saver.save(
            "jet", {**doc, "description": "again"}, operator="Bo", overwrite=False
        )
    assert err.value.extra.get("exists") is True
    out = saver.save(
        "jet", {**doc, "description": "again"}, operator="Bo", overwrite=True
    )
    assert out.commit and git("log", "-1", "--format=%an", "main", cwd=origin) == "Bo via configurator"  # fmt: skip
