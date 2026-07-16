"""Hermetic tests for GeecsSession (mock CA backends; no gateway, no network).

Full-scan behavior rides on already-tested pieces (CA devices, plans,
run_wrapper, ShotController) and is exercised live; here we pin the session's
construction, wiring, and validation surface.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from pathlib import Path

import pytest

pytest.importorskip("aioca")  # session is CA-only

from ophyd_async.core import callback_on_mock_put, set_mock_value  # noqa: E402

from geecs_bluesky.exceptions import GeecsConfigurationError  # noqa: E402
from geecs_bluesky.session import GeecsSession, _json_safe, _positions  # noqa: E402
from geecs_bluesky.shot_controller import CaPutSetter, ShotController  # noqa: E402
from tests.ca_mock_helpers import start_pacer  # noqa: E402

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
    """start/end/step expands to an inclusive position list."""
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


def test_construction_with_unreachable_tiled_server_is_prompt(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """tiled=True off-network: no hang, warning logged, session still usable.

    Before the reachability pre-check, ``GeecsSession(...)`` (tiled defaults
    true) synchronously called ``tiled.client.from_uri`` and blocked for the
    full HTTP connect timeout just to build a session off the lab network.
    """
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()
    uri = f"http://127.0.0.1:{dead_port}"
    monkeypatch.setattr(
        "geecs_bluesky.tiled_integration.read_tiled_config", lambda: (uri, None)
    )

    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        s = GeecsSession("Undulator", mock=True)  # tiled defaults to True
    elapsed = time.monotonic() - started

    assert elapsed < 3.0, f"construction took {elapsed:.1f}s — Tiled blocked it"
    assert f"Tiled server {uri} unreachable" in caplog.text
    # Session remains fully usable with Tiled disabled.
    det = s.detector("UC_Amp2_IR_input", ["centroidx"])
    assert det.name == "uc_amp2_ir_input"


def test_shot_control_attaches_ca_controller() -> None:
    """A dict config becomes a ShotController with gateway :SP setters."""
    s = _session()
    controller = s.shot_control(SHOT_CONTROL)
    assert isinstance(controller, ShotController)
    setter = controller._setters["Trigger.Source"]
    assert isinstance(setter, CaPutSetter)
    assert setter._pv == "Undulator:U_DG645_ShotControl:Trigger_Source:SP"
    assert s.shot_control(None) is None  # detaches


def test_shot_control_accepts_generalized_writes() -> None:
    """ShotControlWrites (multi-device ordered) builds the ordered controller."""
    from geecs_bluesky.models.shot_control import ShotControlWrites

    s = _session()
    writes = ShotControlWrites(
        name="spans",
        states={
            "SCAN": [
                ("U_DG645_ShotControl", "Amplitude.Ch AB", "4.0"),
                ("U_PLC", "DO.Ch9", "on"),
            ]
        },
    )
    controller = s.shot_control(writes)
    assert isinstance(controller, ShotController)
    assert controller.defines_state("SCAN")
    # One CA setter per (device, variable), on each device's own :SP PV.
    setters = controller._setters
    assert {
        setters[("U_DG645_ShotControl", "Amplitude.Ch AB")]._pv,
        setters[("U_PLC", "DO.Ch9")]._pv,
    } == {
        "Undulator:U_DG645_ShotControl:Amplitude_Ch_AB:SP",
        "Undulator:U_PLC:DO_Ch9:SP",
    }
    # Empty writes detach, like an empty legacy config.
    assert s.shot_control(ShotControlWrites(name="empty", states={})) is None


def test_action_signal_factory_builds_cached_connected_signals() -> None:
    """The production SettableFactory: CA-native :SP puts, dtype-inferred reads."""
    import asyncio

    s = _session()
    factory = s.action_signal_factory()
    settable = factory.get_settable("U_PLC", "DO.Ch9")
    assert settable._pv.endswith("Undulator:U_PLC:DO_Ch9:SP")
    # Cached per (device, variable): the same wrapper comes back.
    assert factory.get_settable("U_PLC", "DO.Ch9") is settable
    readable = factory.get_readable("U_PLC", "DI.Ch17")
    assert readable.source.endswith("Undulator:U_PLC:DI_Ch17")
    assert factory.get_readable("U_PLC", "DI.Ch17") is readable

    # Values are coerced to wire strings on set (mock session records the
    # put — raw CA in production, per the CaPutSetter convention, so numeric
    # :SP PVs accept the put).  set() is awaited inside the RE loop.
    async def _set() -> None:
        await settable.set(4.0)

    asyncio.run_coroutine_threadsafe(_set(), s.RE._loop).result(timeout=5.0)
    assert settable.last_mock_put == "4.0"

    # The factory rides the scan cleanup path uniformly.
    s.disconnect(factory)
    assert factory._settables == {}


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


# ---------------------------------------------------------------------------
# Strict-mode fail-fast validation (scan + optimize preambles)
# ---------------------------------------------------------------------------

# ARMED missing entirely — strict must refuse before claiming a folder.
SHOT_CONTROL_NO_ARMED = {
    "device": "U_DG645_ShotControl",
    "variables": {
        "Trigger.Source": {"OFF": "Single shot", "SCAN": "Internal"},
        "Trigger.ExecuteSingleShot": {"SINGLESHOT": "on"},
    },
}

# ARMED present but SINGLESHOT all-empty — fire_shot would be a silent no-op.
SHOT_CONTROL_EMPTY_SINGLESHOT = {
    "device": "U_DG645_ShotControl",
    "variables": {
        "Trigger.Source": {"ARMED": "Single shot", "SINGLESHOT": ""},
        "Trigger.ExecuteSingleShot": {"SINGLESHOT": ""},
    },
}


class _NeverAskedSuggester:
    """Suggester stand-in for validation tests (must never be reached)."""

    def suggest(self):
        raise AssertionError("suggest() reached despite failed validation")

    def observe(self, inputs, objective, bin_data):
        raise AssertionError("observe() reached despite failed validation")


def _no_claim(monkeypatch: pytest.MonkeyPatch) -> list:
    """Patch claim_scan_number to record calls (none expected)."""
    claims: list = []
    monkeypatch.setattr(
        "geecs_bluesky.session.claim_scan_number",
        lambda experiment: claims.append(experiment) or (None, None),
    )
    return claims


def test_strict_optimize_without_armed_fails_before_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict optimize with no ARMED state raises before any folder claim."""
    s = _session()
    det = s.detector("UC_Amp2_IR_input", ["centroidx"])
    s.shot_control(SHOT_CONTROL_NO_ARMED)
    claims = _no_claim(monkeypatch)

    with pytest.raises(GeecsConfigurationError, match="non-empty ARMED"):
        s.optimize(
            variables={},
            detectors=[det],
            objective=lambda b: 0.0,
            suggester=_NeverAskedSuggester(),
            mode="strict",
        )
    assert claims == [], "scan folder was claimed despite failed validation"


def test_strict_scan_with_empty_singleshot_fails_before_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-empty SINGLESHOT entry fails validation (silent no-op fire)."""
    s = _session()
    det = s.detector("UC_Amp2_IR_input", ["centroidx"])
    s.shot_control(SHOT_CONTROL_EMPTY_SINGLESHOT)
    claims = _no_claim(monkeypatch)

    with pytest.raises(GeecsConfigurationError, match="non-empty SINGLESHOT"):
        s.scan(detectors=[det], mode="strict")
    assert claims == []


def test_shot_control_unreachable_device_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo'd shot-control device fails at attach, not mid-plan."""

    async def _never_connects(pvs, timeout=None):
        raise TimeoutError(f"channels never connected: {pvs}")

    monkeypatch.setattr("aioca.connect", _never_connects)
    s = GeecsSession("Undulator", tiled=False, mock=False)  # real CA path
    with pytest.raises(GeecsConfigurationError, match="unreachable"):
        s.shot_control(SHOT_CONTROL)
    assert s._shot_controller is None, "failed attach must not leave a controller"


def test_shot_control_reachability_check_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reachable device attaches; the check sees every setter PV."""
    seen: list = []

    async def _connects(pvs, timeout=None):
        seen.append(sorted(pvs))

    monkeypatch.setattr("aioca.connect", _connects)
    s = GeecsSession("Undulator", tiled=False, mock=False)
    controller = s.shot_control(SHOT_CONTROL)
    assert isinstance(controller, ShotController)
    assert seen == [
        [
            "Undulator:U_DG645_ShotControl:Trigger_ExecuteSingleShot:SP",
            "Undulator:U_DG645_ShotControl:Trigger_Source:SP",
        ]
    ]


def test_shot_control_empty_dict_detaches() -> None:
    """shot_control({}) behaves like shot_control(None): clean detach."""
    s = _session()
    assert s.shot_control(SHOT_CONTROL) is not None
    assert s.shot_control({}) is None
    assert s._shot_controller is None


# ---------------------------------------------------------------------------
# Pre-claimed scan_number/scan_folder pairing
# ---------------------------------------------------------------------------


def test_scan_number_without_folder_raises_clearly() -> None:
    """scan_number without scan_folder is a config error, not a TypeError."""
    s = _session()
    det = s.detector("UC_Amp2_IR_input", ["centroidx"])

    with pytest.raises(GeecsConfigurationError, match="scan_folder"):
        s.scan(detectors=[det], scan_number=123)
    with pytest.raises(GeecsConfigurationError, match="scan_folder"):
        s.optimize(
            variables={},
            detectors=[det],
            objective=lambda b: 0.0,
            suggester=_NeverAskedSuggester(),
            scan_number=123,
        )


# ---------------------------------------------------------------------------
# optimization.json serialization (failed objectives -> null)
# ---------------------------------------------------------------------------


def test_json_safe_maps_nonfinite_floats_to_none() -> None:
    """NaN/inf are nulled recursively; finite values pass through."""
    history = [
        {"iteration": 1, "inputs": {"x": float("nan")}, "objective": float("inf")},
        {"iteration": 2, "inputs": {"x": 0.5}, "objective": -1.25, "tags": [1.0]},
    ]
    safe = _json_safe(history)
    assert safe[0] == {"iteration": 1, "inputs": {"x": None}, "objective": None}
    assert safe[1] == history[1]


def test_optimization_json_with_failed_iteration_is_valid(tmp_path: Path) -> None:
    """A failed objective lands as null in optimization.json (parseable)."""
    s = _session()
    cam = s.detector("UC_Cam", ["Sig"], name="cam")
    knob = s.settable("U_S1H", "Current", name="s1h")
    callback_on_mock_put(
        knob._setpoint, lambda v, **kw: set_mock_value(knob.readback, v)
    )
    set_mock_value(cam.acq_timestamp, 1000.0)
    s._export_scalar_files = lambda scan_number: None  # no Tiled in mock tests

    points = [{"s1h": 0.1}, {"s1h": 0.4}]

    class _Suggester:
        def suggest(self):
            return points.pop(0) if points else None

        def observe(self, inputs, objective, bin_data):
            pass

    calls = {"n": 0}

    def objective(bin_data) -> float:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("evaluator blew up")  # -> NaN in history
        return 1.5

    pacer = start_pacer(s.RE, [(cam, 1000.0)], initial_delay=0.8, interval=0.15)
    try:
        _uid, history = s.optimize(
            variables={"s1h": knob},
            detectors=[cam],
            objective=objective,
            suggester=_Suggester(),
            shots_per_iteration=1,
            max_iterations=5,
            scan_number=42,
            scan_folder=str(tmp_path),
        )
    finally:
        pacer.cancel()

    text = (tmp_path / "optimization.json").read_text()
    assert "NaN" not in text
    data = json.loads(text)  # strict parser accepts it
    assert data[0]["objective"] is None
    assert data[1]["objective"] == pytest.approx(1.5)
    assert len(data) == len(history) == 2


# ---------------------------------------------------------------------------
# ScanInfo scanner stamp
# ---------------------------------------------------------------------------


def test_scan_info_stamps_bluesky_scanner(tmp_path: Path) -> None:
    """ScanInfo ini carries a Scanner = "bluesky" metadata key."""
    s = _session()
    s._write_scan_info(
        7,
        str(tmp_path),
        motor=None,
        positions=[None],
        shots_per_step=3,
        description="stamp test",
    )
    content = (tmp_path / f"ScanInfo{tmp_path.name}.ini").read_text()
    assert 'Scanner = "bluesky"' in content
    assert "[Scan Info]" in content  # legacy section intact
