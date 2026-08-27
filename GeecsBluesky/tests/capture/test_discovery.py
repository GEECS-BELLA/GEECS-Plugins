"""Discovery: devicetype-keyed camera selection from batched DB queries."""

from __future__ import annotations

import geecs_bluesky.capture.discovery as discovery
from geecs_bluesky.capture.discovery import CameraTarget, discover_capture_cameras


def test_discovery_selects_registry_devicetypes_only(monkeypatch) -> None:
    """Only registry devicetypes become targets; PVs compose via pv_naming."""
    monkeypatch.setattr(
        discovery.GeecsDb,
        "get_experiment_device_types",
        classmethod(
            lambda cls, experiment: {
                "UC_Amp4_IR_input": "Point Grey Camera",
                "U_S1H": "EMQ_TDK",
                "U_Haso": "HASO WFS",
            }
        ),
    )
    monkeypatch.setattr(
        discovery.GeecsDb,
        "get_experiment_devices",
        classmethod(
            lambda cls, experiment: {
                "UC_Amp4_IR_input": ("192.168.6.100", 5005),
                "U_S1H": ("192.168.6.20", 5001),
                "U_Haso": ("192.168.6.30", 5002),
            }
        ),
    )

    targets = discover_capture_cameras("Undulator")
    assert targets == [
        CameraTarget(
            device="UC_Amp4_IR_input",
            device_type="Point Grey Camera",
            pv="undulator:uc_amp4_ir_input:image",
            server_ip="192.168.6.100",
        )
    ]


def test_discovery_skips_missing_endpoint_row(monkeypatch) -> None:
    """A camera without an endpoint row is skipped, not fatal."""
    monkeypatch.setattr(
        discovery.GeecsDb,
        "get_experiment_device_types",
        classmethod(lambda cls, experiment: {"UC_Ghost": "Point Grey Camera"}),
    )
    monkeypatch.setattr(
        discovery.GeecsDb,
        "get_experiment_devices",
        classmethod(lambda cls, experiment: {}),
    )

    assert discover_capture_cameras("Undulator") == []
