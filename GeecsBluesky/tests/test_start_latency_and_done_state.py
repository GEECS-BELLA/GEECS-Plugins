"""Pin for the telemetry batch-connect QoL fix (0.33.0).

Telemetry devices connect **concurrently** (``telemetry_batch``) — the
sequential per-device connects cost ~9 s of start-to-execution latency
at ~87 devices (measured live 2026-07-13).

The bridge-side ``is_scanning_active()`` DONE-race pins that used to live
here died with the BlueskyScanner bridge (W5, issue #649).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import geecs_bluesky.session as session_module
from geecs_bluesky.scan_request_runner import build_telemetry_readables


class TestTelemetryBatchConnect:
    """Telemetry connects run concurrently, drop-on-failure per device."""

    def test_batch_connects_all_and_drops_failures(self, monkeypatch) -> None:
        pytest.importorskip("aioca")
        real = session_module.CaTelemetryReadable

        class Flaky(real):  # type: ignore[misc,valid-type]
            async def connect(self, **kwargs):
                if self._geecs_device_name == "bad":
                    raise RuntimeError("dead at scan start")
                return await super().connect(**kwargs)

        monkeypatch.setattr(session_module, "CaTelemetryReadable", Flaky)
        session = session_module.GeecsSession("Test", tiled=False, mock=True)
        connected = session.telemetry_batch({"a": ["X"], "bad": ["Y"], "c": ["Z"]})
        assert [d._geecs_device_name for d in connected] == ["a", "c"]

    def test_runner_uses_batch_and_records_only_connected(self) -> None:
        class BatchSession:
            def __init__(self) -> None:
                self.batch_calls: list[dict] = []

            def telemetry_batch(self, selected):
                self.batch_calls.append(dict(selected))
                return [
                    SimpleNamespace(_geecs_device_name=device)
                    for device in selected
                    if device != "dropped"
                ]

            def telemetry(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError(
                    "sequential telemetry factory used despite batch support"
                )

        policy = SimpleNamespace(
            subscribed_by_device=lambda: {
                "d1": ["v1"],
                "dropped": ["v2"],
                "d3": ["v3"],
            }
        )
        session = BatchSession()
        readables, recorded = build_telemetry_readables(session, None, policy)
        assert session.batch_calls  # the batch path was taken
        # Members arrive wrapped in one CaTelemetryGroup (one read Msg/row).
        assert len(readables) == 1
        assert [m._geecs_device_name for m in readables[0].members] == ["d1", "d3"]
        assert recorded == {"d1": ["v1"], "d3": ["v3"]}

    def test_runner_falls_back_without_batch(self) -> None:
        class LegacySession:
            def telemetry(self, device, variables, **kwargs):
                if device == "dead":
                    return None
                return SimpleNamespace(_geecs_device_name=device)

        policy = SimpleNamespace(
            subscribed_by_device=lambda: {"d1": ["v1"], "dead": ["v2"]}
        )
        readables, recorded = build_telemetry_readables(LegacySession(), None, policy)
        assert len(readables) == 1
        assert [m._geecs_device_name for m in readables[0].members] == ["d1"]
        assert recorded == {"d1": ["v1"]}
