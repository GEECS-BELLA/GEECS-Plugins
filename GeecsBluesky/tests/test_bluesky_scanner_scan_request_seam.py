"""Tests for the BlueskyScanner ScanRequest entry (the acceptance seam).

``reinitialize`` duck-detects a :class:`~geecs_schemas.ScanRequest` and maps
it onto the same internal machinery as the legacy ``exec_config`` path — the
GUI keeps using ``exec_config`` untouched.  The **parity pin**: a ScanRequest
noscan drives the exact same fake-session ``scan()`` call as the equivalent
exec_config (at the legacy identity rep rate of 1 Hz, where the legacy
``shots = rep_rate × wait_time`` derivation is invertible).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.scan_request_runner import MULTI_AXIS_MESSAGE
from geecs_bluesky.scanner_bridge import bluesky_scanner
from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner
from geecs_schemas import (
    ActionPlan,
    SaveSet,
    SaveSetEntry,
    ScanRequest,
    ScanVariable,
    TriggerProfile,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSyncDevice:
    def __init__(self, device: str, name: str, factory: str) -> None:
        self.name = name
        self._geecs_device_name = device
        self._last_acq = None
        self.factory = factory

    def trigger(self) -> None:
        raise NotImplementedError


class _FakeSnapshotDevice:
    def __init__(self, device: str, name: str) -> None:
        self.name = name
        self._geecs_device_name = device
        self.factory = "snapshot"


class _FakeMovable:
    def __init__(self, device: str, variable: str, kind: str, name: str) -> None:
        self.name = name
        self._geecs_device_name = device
        self.variable = variable
        self.kind = kind


class _FakeSession:
    """Records factory + scan calls (mirrors the pre-flight test fake)."""

    def __init__(self) -> None:
        self.rep_rate_hz = 1.0
        self.scan_kwargs: dict | None = None
        self.shot_control_config = "unset"
        self.movables: list[_FakeMovable] = []

    def detector(self, device, variables, *, save_images=False, name=None):
        return _FakeSyncDevice(device, name or device, "detector")

    def contributor(self, device, variables, *, save_images=False, name=None):
        return _FakeSyncDevice(device, name or device, "contributor")

    def snapshot(self, device, variables, *, name=None):
        return _FakeSnapshotDevice(device, name or device)

    def motor(self, device, variable, *, name=None, **kwargs):
        movable = _FakeMovable(device, variable, "motor", name or device)
        self.movables.append(movable)
        return movable

    def settable(self, device, variable, *, name=None):
        movable = _FakeMovable(device, variable, "settable", name or device)
        self.movables.append(movable)
        return movable

    def shot_control(self, config):
        self.shot_control_config = config

    def scan(self, **kwargs):
        self.scan_kwargs = kwargs
        return "uid"


class _StubResolver:
    """In-memory ConfigResolver for the seam tests."""

    def __init__(
        self,
        save_sets: dict | None = None,
        trigger_profiles: dict | None = None,
        variables: dict | None = None,
        plans: dict | None = None,
    ) -> None:
        self.save_sets = save_sets or {}
        self.trigger_profiles = trigger_profiles or {}
        self.variables = variables or {}
        self.plans = plans or {}

    def resolve_save_set(self, name):
        try:
            return self.save_sets[name]
        except KeyError:
            raise GeecsConfigurationError(f"save set {name!r} not found") from None

    def resolve_trigger_profile(self, name):
        try:
            return self.trigger_profiles[name]
        except KeyError:
            raise GeecsConfigurationError(
                f"trigger profile {name!r} not found"
            ) from None

    def resolve_scan_variable(self, name):
        try:
            return self.variables[name]
        except KeyError:
            raise GeecsConfigurationError(f"scan variable {name!r} not found") from None

    def resolve_action_plan(self, name):
        try:
            return self.plans[name]
        except KeyError:
            raise GeecsConfigurationError(f"action plan {name!r} not found") from None


def _make_scanner(session: _FakeSession) -> BlueskyScanner:
    scanner = BlueskyScanner.__new__(BlueskyScanner)
    scanner._session = session
    scanner._experiment_dir = "TestExp"
    scanner._devices_config = {}
    scanner._acquisition_mode = "free_run_time_sync"
    scanner._shot_control = None
    scanner._shots_per_step = 1
    scanner._rep_rate_hz = 1.0
    scanner._detectors = []
    scanner._motor = None
    scanner._device_lock = threading.Lock()
    scanner._on_event = None
    scanner._current_state = None
    scanner._total_shots = 0
    scanner._total_steps = 0
    scanner._completed_shots = 0
    scanner._abort_requested = False
    scanner._optimization_loader = None
    scanner._scan_request = None
    scanner._request_step = None
    scanner._scan_config = None
    scanner._RE = SimpleNamespace(
        state="idle", abort=lambda reason=None: None, _loop=None
    )
    return scanner


_SAVE_SET = SaveSet(
    name="baseline",
    entries=[
        SaveSetEntry(device="U_Ref", scalars=["Sig"]),
        SaveSetEntry(device="U_Cam2", scalars=["Val"], images=True),
        SaveSetEntry(device="U_Stage", scalars=["Pos"], role="snapshot"),
    ],
)


def _resolver(**overrides) -> _StubResolver:
    base = dict(save_sets={"baseline": _SAVE_SET})
    base.update(overrides)
    return _StubResolver(**base)


@pytest.fixture(autouse=True)
def _no_stale_recheck_wait(monkeypatch):
    """Skip the 2 s staleness-recheck grace (fresh frames are irrelevant here)."""
    monkeypatch.setattr(bluesky_scanner, "_STALE_RECHECK_WAIT_S", 0.0)


def _patch_claim(monkeypatch, claims: list) -> None:
    monkeypatch.setattr(
        bluesky_scanner,
        "claim_scan_number",
        lambda experiment: claims.append(experiment) or (None, None),
    )


def _normalize_scan_kwargs(kwargs: dict) -> dict:
    normalized = dict(kwargs)
    normalized["detectors"] = [
        (d._geecs_device_name, d.factory) for d in kwargs["detectors"]
    ]
    if kwargs.get("motor") is not None:
        normalized["motor"] = (
            kwargs["motor"]._geecs_device_name,
            kwargs["motor"].variable,
            kwargs["motor"].kind,
        )
    return normalized


# ---------------------------------------------------------------------------
# THE parity pin: ScanRequest noscan ≡ exec_config noscan
# ---------------------------------------------------------------------------


def test_scan_request_noscan_parity_with_exec_config(monkeypatch) -> None:
    """The two entry shapes drive identical fake-session scan() calls."""
    monkeypatch.delenv("GEECS_BLUESKY_ACQUISITION_MODE", raising=False)
    claims: list = []
    _patch_claim(monkeypatch, claims)

    # --- legacy exec_config path (1 Hz × 3 s wait → 3 shots) ---------------
    exec_session = _FakeSession()
    exec_scanner = _make_scanner(exec_session)
    exec_config = SimpleNamespace(
        scan_config=SimpleNamespace(
            scan_mode="noscan",
            device_var=None,
            start=0.0,
            end=0.0,
            step=0.0,
            wait_time=3.0,
            additional_description="stats run",
            background=False,
        ),
        options=SimpleNamespace(rep_rate_hz=1.0, acquisition_mode="free_run_time_sync"),
        save_config=SimpleNamespace(
            Devices={
                "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
                "U_Cam2": {
                    "synchronous": True,
                    "save_nonscalar_data": True,
                    "variable_list": ["Val"],
                },
                "U_Stage": {"synchronous": False, "variable_list": ["Pos"]},
            }
        ),
    )
    assert exec_scanner.reinitialize(exec_config) is True
    exec_scanner._run_scan(exec_scanner._scan_config)

    # --- ScanRequest path ---------------------------------------------------
    request_session = _FakeSession()
    request_scanner = _make_scanner(request_session)
    request = ScanRequest.model_validate(
        {
            "mode": "noscan",
            "shots_per_step": 3,
            "acquisition": "free_run",
            "save_set": "baseline",
            "description": "stats run",
        }
    )
    assert request_scanner.reinitialize(request, resolver=_resolver()) is True
    request_scanner._run_scan(request_scanner._scan_config)

    assert exec_session.scan_kwargs is not None
    assert request_session.scan_kwargs is not None
    assert _normalize_scan_kwargs(request_session.scan_kwargs) == (
        _normalize_scan_kwargs(exec_session.scan_kwargs)
    )
    assert claims == ["TestExp", "TestExp"]


# ---------------------------------------------------------------------------
# Step scans through the seam
# ---------------------------------------------------------------------------


def test_scan_request_step_uses_resolved_variable_kind(monkeypatch) -> None:
    """A motor-kind variable builds session.motor; positions come verbatim."""
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession()
    scanner = _make_scanner(session)
    resolver = _resolver(
        variables={
            "jet_z": ScanVariable(target="U_ESP_JetXYZ:Position.Axis 3", kind="motor")
        }
    )
    request = ScanRequest.model_validate(
        {
            "mode": "step",
            "shots_per_step": 2,
            "acquisition": "free_run",
            "save_set": "baseline",
            "axes": [{"variable": "jet_z", "positions": {"values": [4.0, 4.5, 6.0]}}],
        }
    )
    assert scanner.reinitialize(request, resolver=resolver) is True
    scanner._run_scan(scanner._scan_config)

    kwargs = session.scan_kwargs
    assert kwargs is not None
    assert kwargs["motor"].kind == "motor"
    assert kwargs["motor"]._geecs_device_name == "U_ESP_JetXYZ"
    assert kwargs["positions"] == [4.0, 4.5, 6.0]
    assert kwargs["shots_per_step"] == 2
    # ScanInfo fields synthesized from the resolved axis
    assert kwargs["scan_info"]["scan_parameter"] == "U_ESP_JetXYZ:Position.Axis 3"
    assert kwargs["scan_info"]["start"] == 4.0
    assert kwargs["scan_info"]["end"] == 6.0
    assert kwargs["md"]["device_var"] == "U_ESP_JetXYZ:Position.Axis 3"


def test_scan_request_trigger_profile_becomes_shot_control(monkeypatch) -> None:
    profile = TriggerProfile(
        name="HTU",
        states={
            "SCAN": [
                {
                    "device": "U_DG645_ShotControl",
                    "variable": "Trigger.Source",
                    "value": "External rising edges",
                }
            ]
        },
    )
    session = _FakeSession()
    scanner = _make_scanner(session)
    request = ScanRequest.model_validate(
        {
            "mode": "noscan",
            "shots_per_step": 1,
            "acquisition": "free_run",
            "save_set": "baseline",
            "trigger_profile": "HTU",
        }
    )
    scanner.reinitialize(request, resolver=_resolver(trigger_profiles={"HTU": profile}))
    # Stored as generalized ordered writes (ShotControlWrites) — the shape
    # GeecsSession.shot_control builds the ordered controller from.
    assert scanner._shot_control is not None
    assert scanner._shot_control.devices == ["U_DG645_ShotControl"]
    assert scanner._shot_control.writes_for_state("SCAN") == [
        ("U_DG645_ShotControl", "Trigger.Source", "External rising edges")
    ]


# ---------------------------------------------------------------------------
# Documented v1 refusals through the seam (fail at reinitialize, pre-thread)
# ---------------------------------------------------------------------------


def test_multi_axis_request_refused_at_reinitialize() -> None:
    scanner = _make_scanner(_FakeSession())
    request = ScanRequest.model_validate(
        {
            "mode": "step",
            "save_set": "baseline",
            "axes": [
                {"variable": "a", "positions": {"start": 0, "end": 1, "step": 1}},
                {"variable": "b", "positions": {"start": 0, "end": 1, "step": 1}},
            ],
        }
    )
    with pytest.raises(NotImplementedError, match=MULTI_AXIS_MESSAGE):
        scanner.reinitialize(request, resolver=_resolver())


def test_actions_validated_then_refused_at_reinitialize() -> None:
    plan = ActionPlan.model_validate({"steps": [{"do": "wait", "seconds": 1.0}]})
    scanner = _make_scanner(_FakeSession())
    request = ScanRequest.model_validate(
        {
            "mode": "noscan",
            "save_set": "baseline",
            "actions": {"setup": ["prep"]},
        }
    )
    with pytest.raises(NotImplementedError, match="prep"):
        scanner.reinitialize(request, resolver=_resolver(plans={"prep": plan}))
    with pytest.raises(GeecsConfigurationError, match="prep"):
        scanner.reinitialize(request, resolver=_resolver())  # name unknown


def test_optimize_request_refused_at_reinitialize() -> None:
    scanner = _make_scanner(_FakeSession())
    request = ScanRequest.model_validate(
        {
            "mode": "optimize",
            "save_set": "baseline",
            "optimization": {
                "variables": {"jet_z": [0.0, 1.0]},
                "evaluator": {"module": "m", "class": "C"},
                "generator": {"name": "random"},
            },
        }
    )
    with pytest.raises(NotImplementedError, match="GeecsSession.run"):
        scanner.reinitialize(request, resolver=_resolver())


def test_exec_config_path_clears_request_state(monkeypatch) -> None:
    """Switching back to exec_config after a request forgets the request."""
    monkeypatch.delenv("GEECS_BLUESKY_ACQUISITION_MODE", raising=False)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession()
    scanner = _make_scanner(session)
    request = ScanRequest.model_validate(
        {
            "mode": "step",
            "shots_per_step": 1,
            "save_set": "baseline",
            "axes": [{"variable": "v", "positions": {"start": 0, "end": 1, "step": 1}}],
        }
    )
    scanner.reinitialize(
        request,
        resolver=_resolver(variables={"v": ScanVariable(target="U_X:Pos")}),
    )
    assert scanner._request_step is not None

    exec_config = SimpleNamespace(
        scan_config=SimpleNamespace(
            scan_mode="noscan",
            device_var=None,
            start=0.0,
            end=0.0,
            step=0.0,
            wait_time=1.0,
            additional_description="",
            background=False,
        ),
        options=SimpleNamespace(rep_rate_hz=1.0, acquisition_mode="free_run_time_sync"),
        save_config=SimpleNamespace(Devices={}),
    )
    scanner.reinitialize(exec_config)
    assert scanner._request_step is None
    assert scanner._scan_request is None
