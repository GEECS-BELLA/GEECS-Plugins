"""Pin for the telemetry batch-connect QoL fix (0.33.0).

Telemetry devices connect **concurrently** (``telemetry_batch``) — the
sequential per-device connects cost ~9 s of start-to-execution latency
at ~87 devices (measured live 2026-07-13).  The scan plan's in-plan
telemetry connect (one gather, drop-on-failure) is pinned in
``test_scan_request_plan.py``; the headless runner's builder over
``telemetry_batch`` went with the runner (Phase 2a).

The bridge-side ``is_scanning_active()`` DONE-race pins that used to live
here died with the BlueskyScanner bridge (W5, issue #649).
"""

from __future__ import annotations

import pytest

import geecs_bluesky.session as session_module


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
