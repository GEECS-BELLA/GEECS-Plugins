"""Tests for the BlueskyScanner ScanRequest entry (the delegation seam).

``reinitialize`` duck-detects a :class:`~geecs_schemas.ScanRequest`,
validates it fail-fast, and stores it; the scan thread then delegates to
:func:`~geecs_bluesky.scan_request_runner.run_scan_request` so the full
schema surface (actions, entry rituals, multi-axis grids) runs through the
one engine definition.  The **parity pin**: a request submitted through the
bridge produces ``session.scan`` kwargs identical to calling
``run_scan_request`` headless on an equivalent fake session.  The bridge
must not pre-claim a scan number on this path (``session.scan`` claims).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlWrites
from geecs_bluesky.scan_request_runner import run_scan_request
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


class _FakeActionSignal:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeActionFactory:
    """Recording SettableFactory for compiled action plans."""

    def __init__(self) -> None:
        self.settables: dict[tuple[str, str], _FakeActionSignal] = {}
        self.readables: dict[tuple[str, str], _FakeActionSignal] = {}

    def get_settable(self, device: str, variable: str) -> _FakeActionSignal:
        return self.settables.setdefault(
            (device, variable), _FakeActionSignal(f"{device}-{variable}")
        )

    def get_readable(self, device: str, variable: str) -> _FakeActionSignal:
        return self.readables.setdefault(
            (device, variable), _FakeActionSignal(f"{device}-{variable}")
        )


class _FakeSession:
    """Records factory + scan calls (no ``experiment`` attr → no DB policy,
    so telemetry stays off — M3b explicit-only behavior)."""

    def __init__(self) -> None:
        self.rep_rate_hz = 1.0
        self.scan_kwargs: dict | None = None
        self.shot_control_config = "unset"
        self.movables: list[_FakeMovable] = []
        self.disconnected: list = []
        self.action_factories: list[_FakeActionFactory] = []

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

    def action_signal_factory(self):
        factory = _FakeActionFactory()
        self.action_factories.append(factory)
        return factory

    def shot_control(self, config):
        self.shot_control_config = config

    def scan(self, **kwargs):
        self.scan_kwargs = kwargs
        return "uid"

    def disconnect(self, *devices):
        self.disconnected.extend(devices)


class _StubResolver:
    """In-memory ConfigResolver for the seam tests."""

    def __init__(
        self,
        save_sets: dict | None = None,
        trigger_profiles: dict | None = None,
        variables: dict | None = None,
        plans: dict | None = None,
        defaults=None,
    ) -> None:
        self.save_sets = save_sets or {}
        self.trigger_profiles = trigger_profiles or {}
        self.variables = variables or {}
        self.plans = plans or {}
        self.defaults = defaults
        self.action_plan_resolutions: list[str] = []

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
        self.action_plan_resolutions.append(name)
        try:
            return self.plans[name]
        except KeyError:
            raise GeecsConfigurationError(f"action plan {name!r} not found") from None

    def resolve_experiment_defaults(self):
        return self.defaults


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
    scanner._request_resolver = None
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

_WAIT_PLAN = ActionPlan.model_validate({"steps": [{"do": "wait", "seconds": 0.01}]})


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
    motor = kwargs.get("motor")
    if isinstance(motor, list):
        normalized["motor"] = [
            (m._geecs_device_name, m.variable, m.kind) for m in motor
        ]
    elif motor is not None:
        normalized["motor"] = (
            motor._geecs_device_name,
            motor.variable,
            motor.kind,
        )
    # Compiled plan-stub callables can't be compared by value — compare
    # their presence/None-ness (the plans behind them are pinned elsewhere).
    for slot in ("setup", "per_step", "closeout"):
        if slot in normalized:
            normalized[slot] = normalized[slot] is not None
    return normalized


def _noscan_request(**overrides) -> ScanRequest:
    base = dict(
        mode="noscan",
        shots_per_step=3,
        acquisition="free_run",
        save_sets=["baseline"],
        description="stats run",
    )
    base.update(overrides)
    return ScanRequest.model_validate(base)


# ---------------------------------------------------------------------------
# THE parity pin: bridge-delegated request ≡ headless run_scan_request
# ---------------------------------------------------------------------------


def test_scan_request_delegation_parity_with_headless(monkeypatch) -> None:
    """The bridge's delegated run produces session.scan kwargs identical to
    calling run_scan_request headless on an equivalent fresh fake session —
    and the bridge never pre-claims (session.scan owns the claim)."""
    claims: list = []
    _patch_claim(monkeypatch, claims)
    request = _noscan_request()

    bridge_session = _FakeSession()
    scanner = _make_scanner(bridge_session)
    assert scanner.reinitialize(request, resolver=_resolver()) is True
    scanner._run_scan(scanner._scan_config)

    headless_session = _FakeSession()
    run_scan_request(headless_session, request, _resolver())

    assert bridge_session.scan_kwargs is not None
    assert headless_session.scan_kwargs is not None
    assert _normalize_scan_kwargs(bridge_session.scan_kwargs) == (
        _normalize_scan_kwargs(headless_session.scan_kwargs)
    )
    # The delegated path must NOT pre-claim: session.scan claims the number
    # itself (and self-attaches scan.log when it claimed).
    assert claims == []
    # The runner's finally disconnects what it created, on both paths.
    assert len(bridge_session.disconnected) == len(headless_session.disconnected) == 3


# ---------------------------------------------------------------------------
# Step scans through the seam (delegated)
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
            "save_sets": ["baseline"],
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
    assert kwargs["scan_info"]["scan_parameter"] == "U_ESP_JetXYZ:Position.Axis 3"
    assert kwargs["md"]["scan_variable"] == "jet_z"
    assert claims == []


def test_multi_axis_request_delegates(monkeypatch) -> None:
    """A 2-axis step request runs as an outer-product grid through the bridge."""
    _patch_claim(monkeypatch, [])
    session = _FakeSession()
    scanner = _make_scanner(session)
    resolver = _resolver(
        variables={
            "a": ScanVariable(target="U_X:Pos"),
            "b": ScanVariable(target="U_Y:Pos"),
        }
    )
    request = ScanRequest.model_validate(
        {
            "mode": "step",
            "shots_per_step": 1,
            "acquisition": "free_run",
            "save_sets": ["baseline"],
            "axes": [
                {"variable": "a", "positions": {"start": 0, "end": 1, "step": 1}},
                {"variable": "b", "positions": {"values": [4.0, 5.0]}},
            ],
        }
    )
    assert scanner.reinitialize(request, resolver=resolver) is True
    scanner._run_scan(scanner._scan_config)

    kwargs = session.scan_kwargs
    assert kwargs is not None
    assert [m._geecs_device_name for m in kwargs["motor"]] == ["U_X", "U_Y"]
    assert kwargs["positions"] == [
        (0.0, 4.0),
        (0.0, 5.0),
        (1.0, 4.0),
        (1.0, 5.0),
    ]
    assert kwargs["md"]["scan_axes"] == ["a", "b"]


def test_scan_request_trigger_profile_reaches_session(monkeypatch) -> None:
    """The delegated run attaches the profile via session.shot_control."""
    _patch_claim(monkeypatch, [])
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
    request = _noscan_request(shots_per_step=1, trigger_profile="HTU")
    scanner.reinitialize(request, resolver=_resolver(trigger_profiles={"HTU": profile}))
    scanner._run_scan(scanner._scan_config)

    writes = session.shot_control_config
    assert isinstance(writes, ShotControlWrites)
    assert writes.devices == ["U_DG645_ShotControl"]
    assert writes.writes_for_state("SCAN") == [
        ("U_DG645_ShotControl", "Trigger.Source", "External rising edges")
    ]


def test_scan_request_without_profile_detaches_shot_control(monkeypatch) -> None:
    _patch_claim(monkeypatch, [])
    session = _FakeSession()
    scanner = _make_scanner(session)
    scanner.reinitialize(_noscan_request(), resolver=_resolver())
    scanner._run_scan(scanner._scan_config)
    assert session.shot_control_config is None


# ---------------------------------------------------------------------------
# Actions through the seam (delegated + fail-fast validation retained)
# ---------------------------------------------------------------------------


def test_actions_delegate_and_compile(monkeypatch) -> None:
    """A setup action compiles and reaches session.scan as a non-None hook."""
    _patch_claim(monkeypatch, [])
    session = _FakeSession()
    scanner = _make_scanner(session)
    request = _noscan_request(actions={"setup": ["prep"]})
    resolver = _resolver(plans={"prep": _WAIT_PLAN})
    assert scanner.reinitialize(request, resolver=resolver) is True
    scanner._run_scan(scanner._scan_config)

    kwargs = session.scan_kwargs
    assert kwargs is not None
    assert kwargs["setup"] is not None
    assert kwargs["per_step"] is None
    assert kwargs["closeout"] is None
    assert kwargs["md"]["action_plans"] == {"setup": ["prep"]}


def test_unknown_action_name_still_fails_at_reinitialize() -> None:
    scanner = _make_scanner(_FakeSession())
    request = _noscan_request(actions={"setup": ["prep"]})
    with pytest.raises(GeecsConfigurationError, match="prep"):
        scanner.reinitialize(request, resolver=_resolver())  # name unknown


def test_defaults_not_applied_twice(monkeypatch) -> None:
    """reinitialize validates a post-defaults COPY but stores the original —
    run_scan_request applies defaults itself, exactly once."""
    _patch_claim(monkeypatch, [])
    session = _FakeSession()
    scanner = _make_scanner(session)
    resolver = _resolver(
        plans={"default_prep": _WAIT_PLAN},
        defaults={"actions": {"setup": ["default_prep"]}},
    )
    assert scanner.reinitialize(_noscan_request(), resolver=resolver) is True
    # The stored request is the pre-defaults original.
    assert list(scanner._scan_request.actions.setup) == []
    scanner._run_scan(scanner._scan_config)

    kwargs = session.scan_kwargs
    assert kwargs is not None
    assert kwargs["md"]["action_plans"]["setup"] == ["default_prep"]
    assert kwargs["md"]["applied_defaults"] == {"actions.setup": ["default_prep"]}


# ---------------------------------------------------------------------------
# Bridge seams: preflight abort + progress totals
# ---------------------------------------------------------------------------


def test_delegated_preflight_abort(monkeypatch) -> None:
    """Preflight returning None aborts pre-claim: no scan, abort flag set."""
    _patch_claim(monkeypatch, [])
    session = _FakeSession()
    scanner = _make_scanner(session)
    scanner.reinitialize(_noscan_request(), resolver=_resolver())
    scanner._preflight_check_sync_liveness = (
        lambda detectors, strict=False, disconnect_on_drop=True: None
    )
    scanner._run_scan(scanner._scan_config)

    assert session.scan_kwargs is None
    assert scanner._abort_requested is True
    # The runner's finally still disconnects the devices it created.
    assert len(session.disconnected) == 3


def test_delegated_preflight_receives_assembled_detectors(monkeypatch) -> None:
    """The bridge preflight hook sees the full detector list + strict flag."""
    _patch_claim(monkeypatch, [])
    session = _FakeSession()
    scanner = _make_scanner(session)
    scanner.reinitialize(_noscan_request(acquisition="strict"), resolver=_resolver())
    seen: dict = {}

    def _fake_preflight(detectors, *, strict=False, disconnect_on_drop=True):
        seen["devices"] = [d._geecs_device_name for d in detectors]
        seen["strict"] = strict
        seen["disconnect_on_drop"] = disconnect_on_drop
        return detectors

    scanner._preflight_check_sync_liveness = _fake_preflight
    scanner._run_scan(scanner._scan_config)

    assert seen["devices"] == ["U_Ref", "U_Cam2", "U_Stage"]
    assert seen["strict"] is True
    assert seen["disconnect_on_drop"] is False
    assert session.scan_kwargs is not None


def test_delegated_totals_hook(monkeypatch) -> None:
    """A 3-position step scan × 2 shots primes the GUI progress totals."""
    _patch_claim(monkeypatch, [])
    session = _FakeSession()
    scanner = _make_scanner(session)
    resolver = _resolver(variables={"v": ScanVariable(target="U_X:Pos")})
    request = ScanRequest.model_validate(
        {
            "mode": "step",
            "shots_per_step": 2,
            "acquisition": "free_run",
            "save_sets": ["baseline"],
            "axes": [{"variable": "v", "positions": {"values": [0.0, 1.0, 2.0]}}],
        }
    )
    scanner.reinitialize(request, resolver=resolver)
    scanner._run_scan(scanner._scan_config)

    assert scanner._total_steps == 3
    assert scanner._total_shots == 6


# ---------------------------------------------------------------------------
# Retained refusal (optimize awaits GUI-submission step (iii)) + state reset
# ---------------------------------------------------------------------------


def test_optimize_request_refused_at_reinitialize() -> None:
    scanner = _make_scanner(_FakeSession())
    request = ScanRequest.model_validate(
        {
            "mode": "optimize",
            "save_sets": ["baseline"],
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
            "save_sets": ["baseline"],
            "axes": [{"variable": "v", "positions": {"start": 0, "end": 1, "step": 1}}],
        }
    )
    resolver = _resolver(variables={"v": ScanVariable(target="U_X:Pos")})
    scanner.reinitialize(request, resolver=resolver)
    assert scanner._scan_request is not None
    assert scanner._request_resolver is resolver

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
    assert scanner._scan_request is None
    assert scanner._request_resolver is None
