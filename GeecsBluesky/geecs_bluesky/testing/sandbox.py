"""Runnable fake-hardware sandbox scans for GeecsBluesky."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bluesky import RunEngine

from geecs_bluesky.devices.generic_detector import GeecsGenericDetector
from geecs_bluesky.devices.motor import GeecsMotor
from geecs_bluesky.plans.step_scan import geecs_step_scan
from geecs_bluesky.testing.fake_device_server import FakeGeecsDevice
from geecs_bluesky.testing.run_engine import (
    BackgroundFakeServers,
    connect_devices,
    disconnect_devices,
)

Document = tuple[str, dict]


@dataclass(frozen=True)
class SandboxScanResult:
    """Captured output from a local fake-hardware RunEngine scan."""

    documents: list[Document]

    @property
    def start_doc(self) -> dict:
        """Return the scan start document."""
        return self._first_doc("start")

    @property
    def stop_doc(self) -> dict:
        """Return the scan stop document."""
        return self._first_doc("stop")

    @property
    def event_docs(self) -> list[dict]:
        """Return all event documents emitted by the scan."""
        return [doc for name, doc in self.documents if name == "event"]

    def _first_doc(self, target_name: str) -> dict:
        for name, doc in self.documents:
            if name == target_name:
                return doc
        raise KeyError(f"no {target_name!r} document captured")


def _update_sandbox_signal(devices: Sequence[FakeGeecsDevice]) -> None:
    """Advance the fake detector value and emit one simulated shot."""
    device = devices[0]
    position = float(device.variables["Position (mm)"])
    previous_timestamp = float(device.variables.get("acq_timestamp", 1000.0))
    next_shot = int(previous_timestamp - 999.0) + 1
    device.variables["Signal"] = 100.0 + 10.0 * position + next_shot
    device.fire_shot()


def run_fake_step_scan(
    *,
    positions: Sequence[float] = (0.0, 0.5, 1.0),
    shots_per_step: int = 2,
    fire_interval: float = 0.05,
) -> SandboxScanResult:
    """Run a real Bluesky RunEngine step scan against localhost fake hardware.

    This sandbox deliberately avoids the GEECS database, Tiled, and lab
    network.  It exercises the production transport, device classes, and
    ``geecs_step_scan`` plan, then returns the emitted Bluesky documents.
    """
    fake_device = FakeGeecsDevice(
        name="U_Sandbox",
        variables={
            "Position (mm)": 0.0,
            "Signal": 100.0,
            "acq_timestamp": 1000.0,
        },
    )

    with BackgroundFakeServers(
        fake_device,
        fire=_update_sandbox_signal,
        initial_delay=0.05,
        interval=fire_interval,
    ) as server:
        host, port = server.endpoint
        motor = GeecsMotor(
            "U_Sandbox",
            "Position (mm)",
            host,
            port,
            name="sandbox_motor",
            units="mm",
        )
        detector = GeecsGenericDetector(
            "U_Sandbox",
            ["Signal"],
            host,
            port,
            name="sandbox_detector",
        )
        detector.configure_shot_id(rep_rate_hz=1.0)

        documents: list[Document] = []
        re = RunEngine()
        re.subscribe(lambda name, doc: documents.append((name, doc)))

        connect_devices(re, motor, detector)
        try:
            re(
                geecs_step_scan(
                    motor=motor,
                    positions=positions,
                    detectors=[detector],
                    shots_per_step=shots_per_step,
                    md={
                        "sandbox": True,
                        "description": "Fake GEECS RunEngine sandbox scan",
                    },
                )
            )
        finally:
            disconnect_devices(re, motor, detector)

    return SandboxScanResult(documents=documents)
