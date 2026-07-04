"""Hermetic tests for GeecsSession (mock CA backends; no gateway, no network).

Full-scan behavior rides on already-tested pieces (CA devices, plans,
run_wrapper, ShotController) and is exercised live; here we pin the session's
construction, wiring, and validation surface.
"""

from __future__ import annotations

import pytest

pytest.importorskip("aioca")  # session is CA-only

from geecs_bluesky.session import GeecsSession, _positions  # noqa: E402
from geecs_bluesky.shot_controller import CaPutSetter, ShotController  # noqa: E402

SHOT_CONTROL = {
    "device": "U_DG645_ShotControl",
    "variables": {
        "Trigger.Source": {
            "OFF": "Single shot",
            "SCAN": "Internal",
            "ARMED": "Single shot",
        },
        "Trigger.ExecuteSingleShot": {"SINGLESHOT": "on"},
    },
}


def _session() -> GeecsSession:
    return GeecsSession("Undulator", tiled=False, mock=True)


def test_positions_math() -> None:
    """start/end/step expands like the scanner's _build_positions."""
    assert _positions(4.0, 5.0, 0.5) == [4.0, 4.5, 5.0]
    assert _positions(1.0, 1.0, 0.5) == [1.0]
    assert _positions(0.0, 1.0, 0.0) == [0.0]


def test_factories_build_connected_named_devices() -> None:
    """Factories yield connected devices with gateway PV sources and names."""
    s = _session()
    det = s.detector("UC_Amp2_IR_input", ["centroidx"])
    con = s.contributor("UC_TopView", ["centroidx"])
    snap = s.snapshot("U_S1H", ["Current"])
    jet = s.motor("U_ESP_JetXYZ", "Position.Axis 1")

    assert det.name == "uc_amp2_ir_input"
    assert det.centroidx.source.endswith("Undulator:UC_Amp2_IR_input:centroidx")
    assert con.acq_timestamp.source.endswith("Undulator:UC_TopView:acq_timestamp")
    assert snap.current.source.endswith("Undulator:U_S1H:Current")
    assert jet._setpoint.source.endswith("Undulator:U_ESP_JetXYZ:Position_Axis_1:SP")


def test_shot_control_attaches_ca_controller() -> None:
    """A dict config becomes a ShotController with gateway :SP setters."""
    s = _session()
    controller = s.shot_control(SHOT_CONTROL)
    assert isinstance(controller, ShotController)
    setter = controller._setters["Trigger.Source"]
    assert isinstance(setter, CaPutSetter)
    assert setter._pv == "Undulator:U_DG645_ShotControl:Trigger_Source:SP"
    assert s.shot_control(None) is None  # detaches


def test_scan_validation() -> None:
    """Bad requests fail loudly before touching any hardware."""
    s = _session()
    det = s.detector("UC_Amp2_IR_input", ["centroidx"])

    with pytest.raises(ValueError, match="at least one detector"):
        s.scan(detectors=[], shots_per_step=1)
    with pytest.raises(ValueError, match="mode="):
        s.scan(detectors=[det], mode="bogus")
    with pytest.raises(ValueError, match="start/end/step"):
        jet = s.motor("U_ESP_JetXYZ", "Position.Axis 1")
        s.scan(detectors=[det], motor=jet)  # no positions
    with pytest.raises(ValueError, match="shot_control"):
        s.scan(detectors=[det], mode="strict", save_data=False)
