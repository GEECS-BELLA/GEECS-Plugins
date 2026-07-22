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

from pathlib import Path
from types import SimpleNamespace

import pytest

from geecs_bluesky import scan_request_runner
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
    PseudoScanVariable,
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
        self.optimize_kwargs: dict | None = None
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

    def optimize(self, **kwargs):
        self.optimize_kwargs = kwargs
        return "uid-opt", []

    def disconnect(self, *devices):
        self.disconnected.extend(devices)


class _FakeOptimizationBridge:
    """Loader-returned bridge fake: records bind()/finish() interactions."""

    def __init__(self, spec) -> None:
        self.spec = spec
        self.bind_kwargs: dict | None = None
        self.finished = False
        self.objective = object()
        self.suggester = object()

    def bind(self, *, devices, scan_tag, scan_folder=None):
        self.bind_kwargs = {
            "devices": list(devices),
            "scan_tag": scan_tag,
            "scan_folder": scan_folder,
        }
        return self.objective, self.suggester

    def finish(self) -> None:
        self.finished = True


class _FakeOptimizationLoader:
    """optimization_loader fake: records what it was called with."""

    def __init__(self) -> None:
        self.calls: list = []
        self.bridges: list[_FakeOptimizationBridge] = []

    def __call__(self, spec) -> _FakeOptimizationBridge:
        self.calls.append(spec)
        bridge = _FakeOptimizationBridge(spec)
        self.bridges.append(bridge)
        return bridge


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
    scanner._on_event = None
    scanner._current_state = None
    scanner._total_shots = 0
    scanner._total_steps = 0
    scanner._completed_shots = 0
    scanner._scan_number = None
    scanner._abort_requested = False
    scanner._optimization_loader = None
    scanner._active_run_uid = None
    scanner._active_descriptor_uids = set()
    scanner._scan_request = None
    scanner._request_resolver = None
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
    """Pin that a bridge-side claim is structurally impossible (G3).

    The bridge module no longer imports any claim entry point, so *claims*
    can only ever stay empty.  If a claim import returns to the bridge,
    this fails loudly — re-patch it here so the never-pre-claims
    assertions in these tests become meaningful again.
    """
    for name in ("claim_scan_number", "claim_scan"):
        assert not hasattr(bluesky_scanner, name), (
            f"the bridge module regained {name}; monkeypatch it here so "
            "the never-pre-claims assertions stay meaningful"
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
    # The stop probe is a bridge-vs-headless seam by design: the bridge
    # supplies its abort-flag lambda, a headless run supplies nothing
    # (issue #571) — drop it from the parity comparison.
    normalized.pop("should_abort", None)
    # The bridge injects a pause supervisor (#552) the headless path does
    # not — a bridge-only seam like should_abort, not a scan-content field.
    normalized.pop("pause_supervisor", None)
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
    scanner._run_scan()

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
    scanner._run_scan()

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
    scanner._run_scan()

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
    scanner._run_scan()

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
    scanner._run_scan()
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
    scanner._run_scan()

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


def test_unknown_trigger_variant_fails_at_reinitialize() -> None:
    """An unknown trigger_variant fails at reinitialize, not in the scan thread."""
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
    scanner = _make_scanner(_FakeSession())
    request = _noscan_request(trigger_profile="HTU", trigger_variant="typo")
    with pytest.raises(GeecsConfigurationError, match="typo"):
        scanner.reinitialize(
            request, resolver=_resolver(trigger_profiles={"HTU": profile})
        )


def test_pseudo_scan_variable_validates_at_reinitialize() -> None:
    """A pseudo (composite) variable now passes reinitialize validation; a
    bad forward formula is still refused there, not in the thread."""
    pseudo = PseudoScanVariable(
        kind="pseudo",
        targets=[{"target": "U_X:Pos", "forward": "composite_var"}],
        mode="absolute",
    )
    scanner = _make_scanner(_FakeSession())
    request = _noscan_request(
        mode="step",
        axes=[{"variable": "combo", "positions": {"values": [0.0, 1.0]}}],
    )
    scanner.reinitialize(request, resolver=_resolver(variables={"combo": pseudo}))

    bad = PseudoScanVariable(
        kind="pseudo",
        targets=[{"target": "U_X:Pos", "forward": "import_os()"}],
        mode="absolute",
    )
    with pytest.raises(GeecsConfigurationError, match="import_os"):
        scanner.reinitialize(request, resolver=_resolver(variables={"combo": bad}))


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
    scanner._run_scan()

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
    scanner._run_scan()

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

    def _fake_preflight(detectors, *, strict=False):
        seen["devices"] = [d._geecs_device_name for d in detectors]
        seen["strict"] = strict
        return detectors

    scanner._preflight_check_sync_liveness = _fake_preflight
    scanner._run_scan()

    assert seen["devices"] == ["U_Ref", "U_Cam2", "U_Stage"]
    assert seen["strict"] is True
    assert session.scan_kwargs is not None


def test_delegated_lifecycle_carries_engine_claimed_scan_number(monkeypatch) -> None:
    """The delegated path picks the scan number up from the start document.

    The bridge never pre-claims here — ``session.scan`` claims inside the
    engine, after INITIALIZING/RUNNING were emitted — so those first
    emissions carry ``None`` and the bridge re-emits RUNNING with the
    number once the start document (which ``geecs_run_wrapper`` stamps
    with ``scan_number``) flows through ``_on_document``.  DONE then
    carries it too.
    """
    from geecs_bluesky.events import ScanLifecycleEvent, ScanState

    _patch_claim(monkeypatch, [])
    session = _FakeSession()
    scanner = _make_scanner(session)
    events: list = []
    scanner._on_event = events.append
    scanner.reinitialize(_noscan_request(), resolver=_resolver())
    # Pass-through preflight: the operator-dialog pipeline is pinned
    # elsewhere and would otherwise dialog on the frameless fakes.
    scanner._preflight_check_sync_liveness = (
        lambda detectors, strict=False, disconnect_on_drop=True: detectors
    )

    original_scan = session.scan

    def scan_with_engine_claim(**kwargs):
        # The engine claims inside session.scan and stamps the number into
        # the run start document; the bridge sees it via _on_document.
        scanner._on_document("start", {"uid": "u1", "scan_number": 33})
        # A duplicate start-doc pickup must not re-emit.
        scanner._on_document("start", {"uid": "u1", "scan_number": 33})
        return original_scan(**kwargs)

    session.scan = scan_with_engine_claim
    scanner._run_scan()

    lifecycle = [
        (e.state, e.scan_number) for e in events if isinstance(e, ScanLifecycleEvent)
    ]
    assert lifecycle == [
        (ScanState.INITIALIZING, None),  # pre-claim: number not known yet
        (ScanState.RUNNING, None),
        (ScanState.RUNNING, 33),  # re-emitted once the engine claimed
        (ScanState.DONE, 33),
    ]


def test_start_document_from_foreign_run_is_ignored(monkeypatch) -> None:
    """A start document while this scanner is not RUNNING changes nothing.

    The session's RunEngine is shared: a headless run driven directly on
    ``scanner._session`` must not flip the idle GUI to RUNNING or plant a
    stale scan number.
    """
    session = _FakeSession()
    scanner = _make_scanner(session)
    events: list = []
    scanner._on_event = events.append

    scanner._on_document("start", {"uid": "u1", "scan_number": 99})

    assert scanner._scan_number is None
    assert events == []


def test_foreign_run_events_while_idle_do_not_mutate_progress() -> None:
    """A whole foreign run while idle leaves GUI progress untouched (#511).

    The session's RunEngine is shared: a headless run driven directly on
    ``scanner._session`` emits start, descriptor, and event documents
    through the same ``_on_document`` subscription.  None of them may
    mutate ``_completed_shots`` or emit a ``ScanStepEvent``.
    """
    from geecs_bluesky.events import ScanStepEvent

    session = _FakeSession()
    scanner = _make_scanner(session)
    events: list = []
    scanner._on_event = events.append

    scanner._on_document("start", {"uid": "f1", "scan_number": 99})
    scanner._on_document("descriptor", {"uid": "fd1", "run_start": "f1"})
    scanner._on_document("event", {"descriptor": "fd1", "data": {"bin_number": 1}})
    scanner._on_document("event", {"descriptor": "fd1", "data": {"bin_number": 2}})
    scanner._on_document("stop", {"run_start": "f1"})

    assert scanner._completed_shots == 0
    assert scanner._scan_number is None
    assert not any(isinstance(e, ScanStepEvent) for e in events)
    assert events == []


def test_foreign_run_events_after_completion_do_not_mutate_progress() -> None:
    """Foreign events after this scanner's run completed are ignored (#511).

    Simulates a full owned run (RUNNING → start/descriptor/event/stop →
    DONE, as the scan thread does), then a foreign run on the shared
    RunEngine: the completed counters must stay frozen and no further
    ``ScanStepEvent`` may reach the GUI.
    """
    from geecs_bluesky.events import ScanStepEvent

    session = _FakeSession()
    scanner = _make_scanner(session)
    events: list = []
    scanner._on_event = events.append
    scanner._total_shots = 1
    scanner._total_steps = 1

    # This scanner's own run: RUNNING is set before the plan reaches the RE.
    scanner._current_state = scanner._scan_state("RUNNING")
    scanner._on_document("start", {"uid": "own1", "scan_number": 12})
    scanner._on_document("descriptor", {"uid": "own-d1", "run_start": "own1"})
    scanner._on_document("event", {"descriptor": "own-d1", "data": {"bin_number": 1}})
    scanner._on_document("stop", {"run_start": "own1"})
    scanner._current_state = scanner._scan_state("DONE")
    assert scanner._completed_shots == 1
    own_step_events = [e for e in events if isinstance(e, ScanStepEvent)]
    assert len(own_step_events) == 1

    # Foreign run after completion: nothing may move.
    scanner._on_document("start", {"uid": "f2", "scan_number": 99})
    scanner._on_document("descriptor", {"uid": "fd2", "run_start": "f2"})
    scanner._on_document("event", {"descriptor": "fd2", "data": {"bin_number": 1}})
    scanner._on_document("stop", {"run_start": "f2"})

    assert scanner._completed_shots == 1
    assert scanner._scan_number == 12
    assert [e for e in events if isinstance(e, ScanStepEvent)] == own_step_events


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
    scanner._run_scan()

    assert scanner._total_steps == 3
    assert scanner._total_shots == 6


# ---------------------------------------------------------------------------
# Optimize through the seam (delegated via the injected optimization_loader)
# ---------------------------------------------------------------------------


def _optimize_request(**overrides) -> ScanRequest:
    base = dict(
        mode="optimize",
        shots_per_step=4,
        acquisition="free_run",
        save_sets=["baseline"],
        optimization={
            "variables": {"jet_z": [0.0, 1.0], "U_S1H:Current": [-2.0, 2.0]},
            "objectives": {"counts": "MAXIMIZE"},
            "evaluator": {"module": "m", "class": "C"},
            "generator": {"name": "bayes_default"},
            "max_iterations": 5,
            "move_to_best_on_finish": True,
        },
    )
    base.update(overrides)
    return ScanRequest.model_validate(base)


def _optimize_resolver(**overrides) -> _StubResolver:
    base = dict(
        variables={"jet_z": ScanVariable(target="U_ESP_JetXYZ:Position.Axis 3")}
    )
    base.update(overrides)
    return _resolver(**base)


def _patch_runner_claim(monkeypatch, tag) -> list:
    """Patch the runner's claim_scan; return the recorded experiment names."""
    claims: list = []
    folder = None if tag is None else f"/nonexistent/scans/Scan{tag.number:03d}"
    monkeypatch.setattr(
        scan_request_runner,
        "claim_scan",
        lambda experiment: claims.append(experiment) or (tag, folder),
    )
    return claims


def test_optimize_request_runs_end_to_end_through_bridge(monkeypatch) -> None:
    """The refusal is gone: an optimize request delegates through the runner.

    The loader receives the request's resolved ``OptimizationSpec``, the
    bridge's ``bind`` gets the connected movables + detectors and the
    runner-claimed scan tag/folder, and its (objective, suggester) pair
    reaches ``session.optimize`` along with the pre-claimed number/folder.
    ``finish()`` bookkeeping runs after the successful run.
    """
    tag = SimpleNamespace(number=41)
    _patch_runner_claim(monkeypatch, tag)
    session = _FakeSession()
    scanner = _make_scanner(session)
    loader = _FakeOptimizationLoader()
    scanner._optimization_loader = loader
    request = _optimize_request()

    assert scanner.reinitialize(request, resolver=_optimize_resolver()) is True
    scanner._run_scan()

    # The loader received the request's optimization field group, verbatim.
    assert loader.calls == [request.optimization]
    (bridge,) = loader.bridges
    assert bridge.bind_kwargs is not None
    assert bridge.bind_kwargs["scan_tag"] is tag
    assert bridge.bind_kwargs["scan_folder"] == "/nonexistent/scans/Scan041"
    bound_names = {
        getattr(d, "_geecs_device_name", None) for d in bridge.bind_kwargs["devices"]
    }
    # Movables (both VOCS variables) + the save set's detectors.
    assert {"U_ESP_JetXYZ", "U_S1H", "U_Ref", "U_Cam2", "U_Stage"} <= bound_names
    assert bridge.finished is True

    kwargs = session.optimize_kwargs
    assert kwargs is not None
    assert kwargs["objective"] is bridge.objective
    assert kwargs["suggester"] is bridge.suggester
    assert kwargs["scan_number"] == 41
    assert kwargs["scan_folder"] == "/nonexistent/scans/Scan041"
    assert kwargs["max_iterations"] == 5
    assert kwargs["shots_per_iteration"] == 4
    assert kwargs["on_finish"] == "best"
    assert kwargs["mode"] == "free_run"
    # Catalog name resolved to its target; Device:Variable passes through.
    assert set(kwargs["variables"]) == {"jet_z", "U_S1H:Current"}
    assert kwargs["md"]["scan_request_mode"] == "optimize"
    assert kwargs["md"]["db_scan_runtime"] == {
        "db_scalars": "applied",
        "background_telemetry": "not_run_in_optimize",
    }
    # The runner's finally still disconnects everything it created.
    assert len(session.disconnected) == 5


def test_optimize_bridge_device_requirements_auto_provisioned(monkeypatch) -> None:
    """The loader-returned bridge's ``device_requirements`` (duck-typed, like
    ``finish``) reach the runner and provision the objective's diagnostic
    into the effective device set — the #520 reversal (field incident
    2026-07-15: NaN objectives because the diagnostic never saved)."""
    _patch_runner_claim(monkeypatch, SimpleNamespace(number=42))
    session = _FakeSession()
    scanner = _make_scanner(session)

    class _RequirementsBridge(_FakeOptimizationBridge):
        device_requirements = {
            "Devices": {
                "UC_TopView": {
                    "synchronous": True,
                    "save_nonscalar_data": True,
                    "variable_list": ["acq_timestamp"],
                }
            }
        }

    loader = _FakeOptimizationLoader()
    loader_calls = loader.calls

    def _loader(spec):
        loader_calls.append(spec)
        bridge = _RequirementsBridge(spec)
        loader.bridges.append(bridge)
        return bridge

    scanner._optimization_loader = _loader
    scanner.reinitialize(_optimize_request(), resolver=_optimize_resolver())
    scanner._run_scan()

    (bridge,) = loader.bridges
    bound_names = {
        getattr(d, "_geecs_device_name", None) for d in bridge.bind_kwargs["devices"]
    }
    assert "UC_TopView" in bound_names  # provisioned, connected, bound
    md = session.optimize_kwargs["md"]
    assert md["provisioned_device_requirements"] == {
        "UC_TopView": {
            "synchronous": True,
            "save_nonscalar_data": True,
            "variable_list": ["acq_timestamp"],
        }
    }


def test_optimize_bridge_without_requirements_attribute_is_no_op(
    monkeypatch,
) -> None:
    """A loader bridge exposing no ``device_requirements`` changes nothing —
    the effective device set is exactly the save sets'."""
    _patch_runner_claim(monkeypatch, None)
    session = _FakeSession()
    scanner = _make_scanner(session)
    loader = _FakeOptimizationLoader()  # its bridge has no such attribute
    scanner._optimization_loader = loader
    scanner.reinitialize(_optimize_request(), resolver=_optimize_resolver())
    scanner._run_scan()

    assert "provisioned_device_requirements" not in session.optimize_kwargs["md"]


def test_optimize_zero_save_sets_accepted_at_reinitialize_and_runs(
    monkeypatch,
) -> None:
    """Optimize mode no longer needs save sets at reinitialize: the optimizer
    provisions its own diagnostics.  Other modes keep the requirement."""
    _patch_runner_claim(monkeypatch, None)
    session = _FakeSession()
    scanner = _make_scanner(session)

    class _RequirementsBridge(_FakeOptimizationBridge):
        device_requirements = {
            "Devices": {
                "UC_TopView": {
                    "synchronous": True,
                    "save_nonscalar_data": True,
                    "variable_list": ["acq_timestamp"],
                }
            }
        }

    scanner._optimization_loader = lambda spec: _RequirementsBridge(spec)
    request = _optimize_request(save_sets=[])
    assert scanner.reinitialize(request, resolver=_optimize_resolver()) is True
    scanner._run_scan()

    md = session.optimize_kwargs["md"]
    assert "save_sets" not in md
    assert list(md["provisioned_device_requirements"]) == ["UC_TopView"]
    # Non-optimize modes still refuse an empty save_sets at reinitialize.
    noscan = ScanRequest.model_validate(
        {"mode": "noscan", "shots_per_step": 3, "save_sets": []}
    )
    with pytest.raises(GeecsConfigurationError, match="save set"):
        _make_scanner(_FakeSession()).reinitialize(noscan, resolver=_resolver())


def test_optimize_totals_hook_primes_gui_progress(monkeypatch) -> None:
    """Optimize primes the GUI totals with the max_iterations upper bound."""
    _patch_runner_claim(monkeypatch, None)
    session = _FakeSession()
    scanner = _make_scanner(session)
    scanner._optimization_loader = _FakeOptimizationLoader()
    scanner.reinitialize(_optimize_request(), resolver=_optimize_resolver())
    scanner._run_scan()

    assert scanner._total_steps == 5
    assert scanner._total_shots == 20  # 5 iterations × 4 shots


def test_optimize_unclaimed_scan_still_runs(monkeypatch) -> None:
    """claim failure (off-network) binds with scan_tag=None and still runs."""
    _patch_runner_claim(monkeypatch, None)
    session = _FakeSession()
    scanner = _make_scanner(session)
    loader = _FakeOptimizationLoader()
    scanner._optimization_loader = loader
    scanner.reinitialize(_optimize_request(), resolver=_optimize_resolver())
    scanner._run_scan()

    (bridge,) = loader.bridges
    assert bridge.bind_kwargs["scan_tag"] is None
    kwargs = session.optimize_kwargs
    assert kwargs is not None
    assert kwargs["scan_number"] is None
    assert kwargs["scan_folder"] is None


def test_optimize_actions_skipped_and_recorded_through_bridge(monkeypatch) -> None:
    """Actions on optimize keep the skip-with-warning behavior via the bridge:
    names still validate fail-fast at reinitialize, nothing compiles/executes,
    and the skip is recorded in the run metadata."""
    _patch_runner_claim(monkeypatch, None)
    session = _FakeSession()
    scanner = _make_scanner(session)
    scanner._optimization_loader = _FakeOptimizationLoader()
    resolver = _optimize_resolver(plans={"prep": _WAIT_PLAN})
    request = _optimize_request(actions={"setup": ["prep"]})
    assert scanner.reinitialize(request, resolver=resolver) is True
    scanner._run_scan()

    kwargs = session.optimize_kwargs
    assert kwargs is not None
    assert kwargs["md"]["skipped_action_plans"] == {"setup": ["prep"]}
    assert session.action_factories == []  # nothing compiled


def test_optimize_unknown_action_still_fails_at_reinitialize() -> None:
    scanner = _make_scanner(_FakeSession())
    scanner._optimization_loader = _FakeOptimizationLoader()
    request = _optimize_request(actions={"setup": ["prep"]})
    with pytest.raises(GeecsConfigurationError, match="prep"):
        scanner.reinitialize(request, resolver=_optimize_resolver())


def test_optimize_unknown_variable_fails_at_reinitialize() -> None:
    """A catalog VOCS name that does not resolve fails at reinitialize."""
    scanner = _make_scanner(_FakeSession())
    scanner._optimization_loader = _FakeOptimizationLoader()
    with pytest.raises(GeecsConfigurationError, match="jet_z"):
        scanner.reinitialize(_optimize_request(), resolver=_resolver())


def test_optimize_pseudo_variable_validates_at_reinitialize() -> None:
    """An optimize VOCS catalog name resolving to a pseudo variable is valid."""
    pseudo = PseudoScanVariable(
        kind="pseudo",
        targets=[{"target": "U_X:Pos", "forward": "composite_var"}],
        mode="absolute",
    )
    scanner = _make_scanner(_FakeSession())
    scanner._optimization_loader = _FakeOptimizationLoader()
    scanner.reinitialize(
        _optimize_request(), resolver=_resolver(variables={"jet_z": pseudo})
    )


def test_optimize_request_without_loader_refused_at_reinitialize() -> None:
    """No injected optimization_loader → clear refusal (the GUI half being
    built in parallel relies on this message staying explicit)."""
    scanner = _make_scanner(_FakeSession())  # _optimization_loader is None
    with pytest.raises(NotImplementedError, match="optimization_loader"):
        scanner.reinitialize(_optimize_request(), resolver=_optimize_resolver())


def test_dependency_direction_no_geecs_scanner_import() -> None:
    """geecs_bluesky must never import geecs_scanner (the loader stays
    injected; the Xopt/evaluator stack lives GUI-side).  AST-level check so
    docstring usage examples don't count — only real import statements."""
    import ast

    import geecs_bluesky

    def _imports_geecs_scanner(tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == "geecs_scanner" for a in node.names):
                    return True
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "geecs_scanner":
                    return True
        return False

    package_root = Path(geecs_bluesky.__file__).parent
    offenders = [
        str(path.relative_to(package_root))
        for path in sorted(package_root.rglob("*.py"))
        if _imports_geecs_scanner(ast.parse(path.read_text()))
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# The one submission shape: anything but a ScanRequest is refused (G3)
# ---------------------------------------------------------------------------


def test_non_scan_request_submission_raises_type_error() -> None:
    """A legacy exec_config-shaped object is refused with a clear TypeError.

    The duck-typed exec_config path was deleted root-and-stem (G3); the
    error must name the removal and point at ScanRequest so any stale
    exec_config-shaped submission fails loudly, not mysteriously.  (The
    legacy GUI itself never reaches this: it dies earlier, at engine
    construction, on the deleted ``shot_control_information`` kwarg.)
    """
    scanner = _make_scanner(_FakeSession())
    exec_config = SimpleNamespace(
        scan_config=SimpleNamespace(scan_mode="noscan"),
        options=SimpleNamespace(rep_rate_hz=1.0),
        save_config=SimpleNamespace(Devices={}),
    )
    with pytest.raises(TypeError, match="exec_config path was removed"):
        scanner.reinitialize(exec_config)
    with pytest.raises(TypeError, match="ScanRequest"):
        scanner.reinitialize({"mode": "noscan"})
    # State is untouched by a refused submission.
    assert scanner._scan_request is None
    assert scanner._request_resolver is None
