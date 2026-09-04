"""Tests for scan_request_runner: resolver, adapters, and the plan preamble.

Covers the configs-repo resolver (new-schema YAML loads directly, legacy YAML
converts — the whole existing corpus is usable immediately), the SaveSet →
devices_config derivation rules, the TriggerProfile → ShotControlWrites
adapter (ordered, multi-device), action slot assembly + compilation +
wiring, multi-axis grid execution, and how a request maps onto the scan
plan's preamble — driven here against a fake GeecsSession *without* a
RunEngine (:func:`run_request` steps ``geecs_scan_request_plan`` by hand
and records the ``build_claimed_scan_plan`` call the preamble ends in).
Optimize-mode execution, telemetry connects, and document parity run on a
real mock RunEngine in ``test_scan_request_plan.py``.  The documented v1
gaps (``all_scalars``, optimize without a loader) refuse loudly.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from bluesky.utils import Msg

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlWrites
from geecs_bluesky.plans.scan_request_plan import geecs_scan_request_plan
from geecs_bluesky.scan_request_runner import (
    ConfigResolver,
    ConfigsRepoResolver,
    apply_experiment_defaults,
    assemble_action_slots,
    build_action_registry,
    collect_save_set_rituals,
    merge_optimizer_device_requirements,
    merge_save_sets,
    resolve_save_sets_and_rituals,
    save_set_to_devices_config,
    snapshot_images_ignored,
    trigger_writes_from_profile,
)
from geecs_schemas import (
    ActionPlan,
    ExperimentDefaults,
    PseudoScanVariable,
    SaveSet,
    SaveSetEntry,
    ScanRequest,
    ScanVariable,
    TriggerProfile,
)

# ---------------------------------------------------------------------------
# Fake session (records factory + scan calls; no RunEngine / CA)
# ---------------------------------------------------------------------------


class _FakeDevice:
    def __init__(self, device: str, kind: str) -> None:
        self._geecs_device_name = device
        self.kind = kind


class _FakeActionSignal:
    """Named stand-in for a CA action signal (message-level assertions only)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeActionFactory:
    """Recording SettableFactory: named fakes, cached per (device, variable)."""

    def __init__(self) -> None:
        self.settables: dict[tuple[str, str], _FakeActionSignal] = {}
        self.readables: dict[tuple[str, str], _FakeActionSignal] = {}
        self.disconnected = False

    def get_settable(self, device: str, variable: str) -> _FakeActionSignal:
        return self.settables.setdefault(
            (device, variable), _FakeActionSignal(f"{device}-{variable}")
        )

    def get_readable(self, device: str, variable: str) -> _FakeActionSignal:
        return self.readables.setdefault(
            (device, variable), _FakeActionSignal(f"{device}-{variable}")
        )

    async def disconnect(self) -> None:
        self.disconnected = True


class PlanSeams:
    """What ``geecs_scan_request_plan`` reads from a session, as a mixin.

    ``experiment`` (empty → the DB-backed policy/served-set/liveness
    providers stay inert, hermetic), ``rep_rate_hz``, ``_mock``, the
    manual-move gate, and ``build_claimed_scan_plan`` — the one call the
    preamble ends in, recorded as :attr:`scan_kwargs`; its returned "inner
    plan" yields nothing and returns ``"uid-scan"``.  Shared by every fake
    session that drives the preamble (this suite, the preflight suite).
    """

    experiment = ""
    rep_rate_hz = 1.0
    _mock = True
    scan_kwargs: dict | None = None

    def _refuse_if_manual_move(self, verb: str) -> None:
        pass

    def build_claimed_scan_plan(self, **kwargs):
        self.scan_kwargs = kwargs
        return _immediately("uid-scan")


class _FakeSession(PlanSeams):
    """Records the preamble's device builds and its final claimed-plan call.

    Devices are plain records (no connect), so the plan's in-plan connect
    stage has nothing to do; the action-signal factory does ride the plan's
    finalize disconnect (see :func:`_drive`).
    """

    def __init__(self) -> None:
        self.devices: list[tuple[str, str]] = []  # (device, factory)
        self.scan_kwargs = None
        self.disconnected: list = []
        self.action_factories: list[_FakeActionFactory] = []
        self.confirm_settable_calls: list = []
        self.pseudo_calls: list = []

    def _make(self, device: str, kind: str) -> _FakeDevice:
        self.devices.append((device, kind))
        return _FakeDevice(device, kind)

    def detector(
        self,
        device,
        variables,
        *,
        save_images=False,
        save_control_only=False,
        name=None,
    ):
        return self._make(device, "detector")

    def contributor(
        self,
        device,
        variables,
        *,
        save_images=False,
        save_control_only=False,
        name=None,
    ):
        return self._make(device, "contributor")

    def snapshot(self, device, variables, *, save_control_only=False, name=None):
        return self._make(device, "snapshot")

    def motor(self, device, variable, *, name=None, **kwargs):
        return self._make(f"{device}:{variable}", "motor")

    def settable(self, device, variable, *, name=None):
        return self._make(f"{device}:{variable}", "settable")

    def confirm_settable(
        self, device, variable, *, confirm_device, confirm_variable, **kwargs
    ):
        self.confirm_settable_calls.append(
            (device, variable, confirm_device, confirm_variable)
        )
        return self._make(f"{device}:{variable}", "confirm_settable")

    def pseudo_movable(self, variable_name, components, mode, *, name=None):
        self.pseudo_calls.append(
            (variable_name, [(d, v, f.source) for d, v, f in components], mode)
        )
        return self._make(f"pseudo:{variable_name}", "pseudo_movable")

    def action_signal_factory(self):
        factory = _FakeActionFactory()
        self.action_factories.append(factory)
        return factory

    def disconnect(self, *devices):
        self.disconnected.extend(devices)


def _immediately(value):
    """A plan that yields nothing and returns *value* (the fake inner plan)."""
    return value
    yield  # pragma: no cover — makes this a generator


class _Done:
    """A settled future stand-in for the driver's ``wait_for`` responses."""

    def __init__(self, value) -> None:
        self._value = value

    def result(self):
        return self._value


def _drive(plan):
    """Step a plan generator without a RunEngine; return its return value.

    The preamble's only RE-dependent messages are ``wait_for`` (in-plan
    connects, the finalize disconnect): their coroutine functions run to
    completion here so the same fail-fast and cleanup semantics hold.
    Every other message gets ``None`` back (nothing in the preamble reads
    a response before the claimed inner plan, which the fake session
    replaces).
    """
    response = None
    try:
        while True:
            msg = plan.send(response)
            response = None
            if msg.command == "wait_for":
                response = [_Done(asyncio.run(fn())) for fn in msg.args[0]]
    except StopIteration as stop:
        return stop.value


@pytest.fixture(autouse=True)
def _no_scan_number_claim(monkeypatch):
    """No data tree in these tests: the plan's claim yields no number/folder."""
    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.claim_scan_number",
        lambda experiment: (None, None),
    )
    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.claim_scan",
        lambda experiment: (None, None),
    )


def run_request(session, request, resolver, *, submission=None, **plan_kwargs):
    """Run *request* through the scan plan's preamble on a fake session.

    The headless-door call shape (``failed_move_policy="raise"``, as
    ``GeecsSession.run`` passes); returns the plan's return value — the
    fake inner plan's ``"uid-scan"`` on success.
    """
    return _drive(
        geecs_scan_request_plan(
            request,
            submission=submission,
            session=session,
            resolver=resolver,
            failed_move_policy="raise",
            **plan_kwargs,
        )
    )


def _collect_messages(plan) -> list[Msg]:
    """Drive a plan-stub generator without a RunEngine (no responses needed)."""
    messages: list[Msg] = []
    try:
        message = plan.send(None)
        while True:
            messages.append(message)
            message = plan.send(None)
    except StopIteration:
        pass
    return messages


def _set_targets(plan) -> list[tuple[str, object]]:
    """The (signal name, value) sequence of a plan's 'set' messages."""
    return [
        (m.obj.name, m.args[0]) for m in _collect_messages(plan) if m.command == "set"
    ]


# ---------------------------------------------------------------------------
# Configs-repo fixture: one experiment with legacy files, one with new-schema
# ---------------------------------------------------------------------------


LEGACY_SAVE_ELEMENT = """\
Devices:
  U_Cam:
    synchronous: true
    save_nonscalar_data: true
    variable_list: [acq_timestamp, MaxCounts]
  U_Cam2:
    synchronous: true
    variable_list: [Val]
  U_Slow:
    synchronous: false
    variable_list: [Pressure]
"""

NEW_SAVE_SET = """\
schema_version: 1
name: NewSet
entries:
  - device: U_New
    scalars: [counts]
    images: true
"""

LEGACY_SHOT_CONTROL = """\
device: U_DG645_ShotControl
variables:
  Trigger.Source:
    "OFF": "Single shot external rising edges"
    SCAN: "External rising edges"
    STANDBY: "External rising edges"
    SINGLESHOT: ""
  Amplitude.Ch AB:
    SCAN: "4.0"
    STANDBY: "0.5"
"""

NEW_TRIGGER_PROFILE = """\
schema_version: 2
name: NewProfile
states:
  SCAN:
    - {device: U_DG645_ShotControl, variable: Trigger.Source, value: External rising edges}
"""

STRICT_TRIGGER_PROFILE = """\
schema_version: 1
name: Strict
states:
  SCAN:
    - {device: U_DG645_ShotControl, variable: Trigger.Source, value: External rising edges}
  STANDBY:
    - {device: U_DG645_ShotControl, variable: Trigger.Source, value: External rising edges}
  ARMED:
    - {device: U_DG645_ShotControl, variable: Trigger.Source, value: Single shot external rising edges}
  SINGLESHOT:
    - {device: U_DG645_ShotControl, variable: Trigger.Source, value: "*TRG"}
"""

# Two asynchronous (snapshot-role) cameras that a save set marks as image
# savers — the #702 shapes: one with scalars, one with none at all.
LEGACY_ASYNC_CAMERAS = """\
Devices:
  U_AsyncCam:
    synchronous: false
    save_nonscalar_data: true
    variable_list: [Pressure]
  U_AsyncBare:
    synchronous: false
    save_nonscalar_data: true
    variable_list: []
"""

LEGACY_ELEMENT_WITH_ACTIONS = """\
Devices:
  U_Cam:
    synchronous: true
    variable_list: [MaxCounts]
setup_action:
  steps:
    - action: set
      device: U_PLC
      variable: DO.Ch1
      value: 'on'
"""

LEGACY_SCAN_DEVICES = """\
single_scan_devices:
  jet_z: "U_ESP_JetXYZ:Position.Axis 3"
  jet_x: "U_ESP_JetXYZ:Position.Axis 1"
"""

NEW_SCAN_VARIABLES = """\
schema_version: 1
variables:
  jet_z: {target: "U_ESP_JetXYZ:Position.Axis 3", kind: motor}
  hexapod_y: {target: "U_Hexapod:ypos"}
  emq1_current:
    target: "U_EMQTripletBipolar:Current_Limit.Ch1"
    confirm: "U_EMQTripletBipolar:Current.Ch1"
  combo:
    kind: pseudo
    mode: absolute
    targets:
      - {target: "U_S1H:Current", forward: "composite_var * 2"}
  bump_x:
    kind: pseudo
    mode: relative
    targets:
      - {target: "U_S3H:Current", forward: "x * 1"}
      - {target: "U_S4H:Current", forward: "x * -2"}
  badcombo:
    kind: pseudo
    mode: absolute
    targets:
      - {target: "U_S1H:Current", forward: "nonsense_name * 2"}
"""

LEGACY_ACTIONS = """\
actions:
  close_shutters:
    steps:
      - action: set
        device: U_PLC
        variable: DO.Ch9
        value: 'on'
  default_prep:
    steps:
      - action: set
        device: U_PLC
        variable: DO.Ch1
        value: 'on'
  scan_prep:
    steps:
      - action: set
        device: U_PLC
        variable: DO.Ch2
        value: 'on'
  between_steps:
    steps:
      - action: set
        device: U_PLC
        variable: DO.Ch3
        value: 'on'
  scan_cleanup:
    steps:
      - action: set
        device: U_PLC
        variable: DO.Ch4
        value: 'off'
  default_cleanup:
    steps:
      - action: set
        device: U_PLC
        variable: DO.Ch5
        value: 'off'
  cam_ritual:
    steps:
      - action: set
        device: U_Cam
        variable: Analysis
        value: 'on'
  cam_park:
    steps:
      - action: set
        device: U_Cam
        variable: Analysis
        value: 'off'
"""

# Second save set in the LegacyExp experiment for the multi-save-set (M4)
# union tests: a fresh device (U_Aux) plus one device (U_Cam) that overlaps
# UC_Test — merged per the documented union rule (scalars unioned, images
# OR'd True, entry ritual unioned once).
AUX_SAVE_SET = """\
schema_version: 1
name: UC_Aux
entries:
  - device: U_Aux
    scalars: [Aux1]
  - device: U_Cam
    scalars: [Extra]
    images: true
    setup: [cam_ritual]
"""

# New-schema save set whose entries carry setup/closeout rituals (shared
# ritual named by both entries — must run once).
RITUAL_SAVE_SET = """\
schema_version: 1
name: RitualSet
entries:
  - device: U_Cam
    scalars: [MaxCounts]
    setup: [cam_ritual]
    closeout: [cam_park]
  - device: U_Cam2
    scalars: [Val]
    setup: [cam_ritual]
"""


@pytest.fixture
def configs_root(tmp_path):
    legacy = tmp_path / "LegacyExp"
    (legacy / "save_devices").mkdir(parents=True)
    (legacy / "save_devices" / "UC_Test.yaml").write_text(LEGACY_SAVE_ELEMENT)
    (legacy / "save_devices" / "UC_WithActions.yaml").write_text(
        LEGACY_ELEMENT_WITH_ACTIONS
    )
    (legacy / "save_devices" / "UC_AsyncCams.yaml").write_text(LEGACY_ASYNC_CAMERAS)
    (legacy / "shot_control_configurations").mkdir()
    (legacy / "shot_control_configurations" / "HTU-Normal.yaml").write_text(
        LEGACY_SHOT_CONTROL
    )
    (legacy / "shot_control_configurations" / "Empty.yaml").write_text("")
    (legacy / "shot_control_configurations" / "Strict.yaml").write_text(
        STRICT_TRIGGER_PROFILE
    )
    (legacy / "scan_devices").mkdir()
    (legacy / "scan_devices" / "scan_devices.yaml").write_text(LEGACY_SCAN_DEVICES)
    (legacy / "action_library").mkdir()
    (legacy / "action_library" / "actions.yaml").write_text(LEGACY_ACTIONS)
    (legacy / "save_devices" / "RitualSet.yaml").write_text(RITUAL_SAVE_SET)
    (legacy / "save_devices" / "UC_Aux.yaml").write_text(AUX_SAVE_SET)

    modern = tmp_path / "ModernExp"
    (modern / "save_devices").mkdir(parents=True)
    (modern / "save_devices" / "NewSet.yaml").write_text(NEW_SAVE_SET)
    (modern / "shot_control_configurations").mkdir()
    (modern / "shot_control_configurations" / "NewProfile.yaml").write_text(
        NEW_TRIGGER_PROFILE
    )
    (modern / "scan_devices").mkdir()
    (modern / "scan_devices" / "scan_variables.yaml").write_text(NEW_SCAN_VARIABLES)
    return tmp_path


@pytest.fixture
def legacy_resolver(configs_root):
    return ConfigsRepoResolver("LegacyExp", experiments_root=configs_root)


@pytest.fixture
def modern_resolver(configs_root):
    return ConfigsRepoResolver("ModernExp", experiments_root=configs_root)


# ---------------------------------------------------------------------------
# ConfigsRepoResolver
# ---------------------------------------------------------------------------


def test_resolver_satisfies_the_protocol(legacy_resolver) -> None:
    assert isinstance(legacy_resolver, ConfigResolver)


def test_legacy_save_element_converts(legacy_resolver) -> None:
    save_set = legacy_resolver.resolve_save_set("UC_Test")
    assert isinstance(save_set, SaveSet)
    by_device = {e.device: e for e in save_set.entries}
    # acq_timestamp is implicit — the converter drops it.
    assert by_device["U_Cam"].scalars == ["MaxCounts"]
    assert by_device["U_Cam"].images is True
    assert by_device["U_Slow"].role is not None  # synchronous: false → snapshot
    assert by_device["U_Slow"].role.value == "snapshot"


def test_new_schema_save_set_loads_directly(modern_resolver) -> None:
    save_set = modern_resolver.resolve_save_set("NewSet")
    assert save_set.name == "NewSet"
    assert save_set.entries[0].device == "U_New"


def test_missing_save_set_raises_with_path(legacy_resolver) -> None:
    with pytest.raises(GeecsConfigurationError, match="save set 'Nope'"):
        legacy_resolver.resolve_save_set("Nope")


def test_legacy_shot_control_converts(legacy_resolver) -> None:
    profile = legacy_resolver.resolve_trigger_profile("HTU-Normal")
    assert isinstance(profile, TriggerProfile)
    assert profile.devices == ["U_DG645_ShotControl"]
    writes = profile.writes_for("SCAN")
    assert {(w.device, w.variable): w.value for w in writes} == {
        ("U_DG645_ShotControl", "Trigger.Source"): "External rising edges",
        ("U_DG645_ShotControl", "Amplitude.Ch AB"): "4.0",
    }
    # legacy empty-string no-op → the state simply has no write
    assert not profile.defines_state("SINGLESHOT")


def test_empty_shot_control_raises(legacy_resolver) -> None:
    with pytest.raises(GeecsConfigurationError, match="names no\n?.*device|names no"):
        legacy_resolver.resolve_trigger_profile("Empty")


def test_new_schema_trigger_profile_loads_directly(modern_resolver) -> None:
    profile = modern_resolver.resolve_trigger_profile("NewProfile")
    assert profile.schema_version == 2
    assert [(w.device, w.variable, w.value) for w in profile.writes_for("SCAN")] == [
        ("U_DG645_ShotControl", "Trigger.Source", "External rising edges")
    ]


def test_legacy_scan_variable_resolves_as_setpoint(legacy_resolver) -> None:
    spec = legacy_resolver.resolve_scan_variable("jet_z")
    assert isinstance(spec, ScanVariable)
    assert spec.target == "U_ESP_JetXYZ:Position.Axis 3"
    assert spec.kind == "setpoint"


def test_new_schema_scan_variables_load(modern_resolver) -> None:
    assert modern_resolver.resolve_scan_variable("jet_z").kind == "motor"
    assert isinstance(
        modern_resolver.resolve_scan_variable("combo"), PseudoScanVariable
    )


def test_public_scan_variable_catalog_accessor(modern_resolver) -> None:
    """The public catalog accessor (0.49.0, for GEECS-Console's panel)."""
    catalog = modern_resolver.scan_variable_catalog()
    assert "combo" in catalog.variables
    assert catalog.variables["jet_z"].kind == "motor"


def test_unknown_scan_variable_lists_known_names(legacy_resolver) -> None:
    with pytest.raises(GeecsConfigurationError, match="jet_z"):
        legacy_resolver.resolve_scan_variable("nope")


def test_action_plan_resolution(legacy_resolver) -> None:
    plan = legacy_resolver.resolve_action_plan("close_shutters")
    assert plan.steps[0].device == "U_PLC"
    with pytest.raises(GeecsConfigurationError, match="close_shutters"):
        legacy_resolver.resolve_action_plan("open_shutters")


# ---------------------------------------------------------------------------
# TriggerProfile → ShotControlWrites adapter (ordered, multi-device)
# ---------------------------------------------------------------------------


def test_trigger_adapter_preserves_state_semantics(legacy_resolver) -> None:
    """Per-state writes and defines_state agree between profile and writes."""
    profile = legacy_resolver.resolve_trigger_profile("HTU-Normal")
    writes = trigger_writes_from_profile(profile)
    assert isinstance(writes, ShotControlWrites)
    assert writes.name == profile.name
    assert writes.devices == profile.devices
    for state in ("OFF", "SCAN", "STANDBY", "SINGLESHOT", "ARMED"):
        expected = [(w.device, w.variable, w.value) for w in profile.writes_for(state)]
        assert writes.writes_for_state(state) == expected, state
        assert writes.defines_state(state) == profile.defines_state(state), state


# ---------------------------------------------------------------------------
# SaveSet → devices_config derivation
# ---------------------------------------------------------------------------


def test_devices_config_derivation_rules() -> None:
    save_set = SaveSet(
        name="s",
        entries=[
            SaveSetEntry(device="U_A", scalars=["x"], images=True),
            SaveSetEntry(device="U_B", scalars=[]),
            SaveSetEntry(device="U_Slow", scalars=["p"], role="snapshot"),
        ],
    )
    config = save_set_to_devices_config(save_set)
    assert list(config) == ["U_A", "U_B", "U_Slow"]
    assert config["U_A"] == {
        "synchronous": True,
        "save_nonscalar_data": True,
        "variable_list": ["x"],
    }
    assert config["U_B"]["synchronous"] is True
    assert config["U_Slow"]["synchronous"] is False


def test_devices_config_reference_override_moves_first() -> None:
    save_set = SaveSet(
        name="s",
        entries=[
            SaveSetEntry(device="U_A", scalars=["x"]),
            SaveSetEntry(device="U_Pin", scalars=["y"], role="reference"),
        ],
    )
    assert list(save_set_to_devices_config(save_set)) == ["U_Pin", "U_A"]


def test_devices_config_contributor_override_demotes() -> None:
    save_set = SaveSet(
        name="s",
        entries=[
            SaveSetEntry(device="U_NotPacemaker", scalars=["x"], role="contributor"),
            SaveSetEntry(device="U_A", scalars=["y"]),
        ],
    )
    assert list(save_set_to_devices_config(save_set)) == ["U_A", "U_NotPacemaker"]


def test_devices_config_two_references_rejected() -> None:
    save_set = SaveSet(
        name="s",
        entries=[
            SaveSetEntry(device="U_A", scalars=["x"], role="reference"),
            SaveSetEntry(device="U_B", scalars=["y"], role="reference"),
        ],
    )
    with pytest.raises(GeecsConfigurationError, match="more than one"):
        save_set_to_devices_config(save_set)


def test_devices_config_all_contributors_rejected() -> None:
    save_set = SaveSet(
        name="s",
        entries=[SaveSetEntry(device="U_A", scalars=["x"], role="contributor")],
    )
    with pytest.raises(GeecsConfigurationError, match="pacemaker"):
        save_set_to_devices_config(save_set)


def test_devices_config_all_scalars_is_a_documented_gap() -> None:
    save_set = SaveSet(name="s", entries=[SaveSetEntry(device="U_A", all_scalars=True)])
    with pytest.raises(NotImplementedError, match="all_scalars"):
        save_set_to_devices_config(save_set)


def _snapshot_images_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "IGNORED for snapshot-role" in r.getMessage()
    ]


def test_devices_config_snapshot_images_warns_loudly(caplog) -> None:
    """#754: images: true on a snapshot-role entry is a no-op — say so, DB-free."""
    save_set = SaveSet(
        name="s754",
        entries=[
            SaveSetEntry(device="U_Cam", scalars=["x"], images=True),
            SaveSetEntry(device="U_Haso", scalars=["p"], images=True, role="snapshot"),
            SaveSetEntry(device="U_Gauge", scalars=["q"], role="snapshot"),
        ],
    )
    with caplog.at_level(logging.WARNING):
        config = save_set_to_devices_config(save_set)
    warnings = _snapshot_images_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert "'s754'" in warnings[0]
    assert "U_Haso" in warnings[0]
    assert "U_Cam" not in warnings[0] and "U_Gauge" not in warnings[0]
    assert "#754" in warnings[0]
    # Not rejected: the entry is still derived (scalars recorded, images inert).
    assert config["U_Haso"] == {
        "synchronous": False,
        "save_nonscalar_data": True,
        "variable_list": ["p"],
    }
    assert snapshot_images_ignored(config) == ["U_Haso"]


def test_devices_config_synchronous_images_do_not_warn(caplog) -> None:
    save_set = SaveSet(
        name="s",
        entries=[
            SaveSetEntry(device="U_Cam", scalars=["x"], images=True),
            SaveSetEntry(device="U_Pin", scalars=["y"], images=True, role="reference"),
            SaveSetEntry(device="U_Slow", scalars=["p"], role="snapshot"),
        ],
    )
    with caplog.at_level(logging.WARNING):
        config = save_set_to_devices_config(save_set)
    assert _snapshot_images_warnings(caplog) == []
    assert snapshot_images_ignored(config) == []


# ---------------------------------------------------------------------------
# The plan preamble on a fake session
# ---------------------------------------------------------------------------


def _noscan_request(**overrides) -> ScanRequest:
    base = dict(
        mode="noscan",
        shots_per_step=3,
        acquisition="free_run",
        save_sets=["UC_Test"],
        description="stats",
    )
    base.update(overrides)
    return ScanRequest.model_validate(base)


def test_noscan_request_maps_onto_session_scan(legacy_resolver) -> None:
    session = _FakeSession()
    uid = run_request(session, _noscan_request(), legacy_resolver)

    assert uid == "uid-scan"
    # free-run roles by position: first sync = reference detector, second =
    # contributor, async = snapshot.
    assert session.devices == [
        ("U_Cam", "detector"),
        ("U_Cam2", "contributor"),
        ("U_Slow", "snapshot"),
    ]
    kwargs = session.scan_kwargs
    assert kwargs["motor"] is None
    assert kwargs["positions"] == [None]
    assert kwargs["shots_per_step"] == 3
    assert kwargs["strict"] is False
    assert kwargs["description"] == "stats"
    assert kwargs["scan_info_overrides"]["scan_mode"] == "noscan"
    assert kwargs["scan_info_overrides"]["background"] is False
    # no trigger profile named → no shot controller built
    assert kwargs["controller"] is None


def test_failed_move_policy_seam_reaches_the_claimed_plan(legacy_resolver) -> None:
    """The one door-specific seam, pinned by behaviour: the headless call
    shape hands ``"raise"`` to the claimed plan, the queue's plain call
    shape (no seam) hands the ``"pause"`` default — and a typo is refused
    pre-claim rather than silently selecting raise semantics downstream."""
    session = _FakeSession()
    run_request(session, _noscan_request(), legacy_resolver)
    assert session.scan_kwargs["failed_move_policy"] == "raise"

    queue_shaped = _FakeSession()
    _drive(
        geecs_scan_request_plan(
            _noscan_request().model_dump(),
            session=queue_shaped,
            resolver=legacy_resolver,
        )
    )
    assert queue_shaped.scan_kwargs["failed_move_policy"] == "pause"

    typo = _FakeSession()
    with pytest.raises(GeecsConfigurationError, match="failed_move_policy='Pause'"):
        _drive(
            geecs_scan_request_plan(
                _noscan_request().model_dump(),
                session=typo,
                resolver=legacy_resolver,
                failed_move_policy="Pause",
            )
        )
    assert typo.devices == []  # refused before any hardware was touched


def test_strict_request_builds_all_sync_as_detectors(legacy_resolver) -> None:
    session = _FakeSession()
    run_request(
        session,
        _noscan_request(acquisition="strict", trigger_profile="Strict"),
        legacy_resolver,
    )
    assert session.devices == [
        ("U_Cam", "detector"),
        ("U_Cam2", "detector"),
        ("U_Slow", "snapshot"),
    ]
    assert session.scan_kwargs["strict"] is True


def test_trigger_profile_is_attached_via_the_adapter(legacy_resolver) -> None:
    session = _FakeSession()
    run_request(session, _noscan_request(trigger_profile="HTU-Normal"), legacy_resolver)
    writes = session.scan_kwargs["controller"]._writes
    assert isinstance(writes, ShotControlWrites)
    assert writes.devices == ["U_DG645_ShotControl"]


def test_step_request_setpoint_variable_uses_settable(legacy_resolver) -> None:
    session = _FakeSession()
    request = _noscan_request(
        mode="step",
        axes=[{"variable": "jet_z", "positions": {"start": 0, "end": 1, "step": 0.5}}],
    )
    run_request(session, request, legacy_resolver)
    kwargs = session.scan_kwargs
    assert kwargs["positions"] == [0.0, 0.5, 1.0]
    assert kwargs["motor"].kind == "settable"  # legacy entries default setpoint
    assert (
        kwargs["scan_info_overrides"]["scan_parameter"]
        == "U_ESP_JetXYZ:Position.Axis 3"
    )
    assert kwargs["md"]["scan_variable"] == "jet_z"


def test_step_request_motor_kind_and_position_list(modern_resolver) -> None:
    session = _FakeSession()
    request = _noscan_request(
        mode="step",
        save_sets=["NewSet"],
        axes=[{"variable": "jet_z", "positions": {"values": [4.0, 4.5, 6.0]}}],
    )
    run_request(session, request, modern_resolver)
    kwargs = session.scan_kwargs
    assert kwargs["motor"].kind == "motor"
    assert kwargs["positions"] == [4.0, 4.5, 6.0]


def test_step_request_confirm_variable_uses_confirm_settable(modern_resolver) -> None:
    """A scan variable with ``confirm`` set builds a confirm-settable movable.

    ``confirm`` takes precedence over ``kind`` (topology-C, session.confirm_
    device.md #5): the request-facing behavior of resolve_movable_target +
    build_movable.
    """
    session = _FakeSession()
    request = _noscan_request(
        mode="step",
        save_sets=["NewSet"],
        axes=[{"variable": "emq1_current", "positions": {"values": [2.0, 2.5]}}],
    )
    run_request(session, request, modern_resolver)
    kwargs = session.scan_kwargs
    assert kwargs["motor"].kind == "confirm_settable"
    assert session.confirm_settable_calls == [
        (
            "U_EMQTripletBipolar",
            "Current_Limit.Ch1",
            "U_EMQTripletBipolar",
            "Current.Ch1",
        )
    ]


# ---------------------------------------------------------------------------
# Multi-axis grid execution (outer product, first axis outermost)
# ---------------------------------------------------------------------------


def test_two_axis_request_runs_as_outer_product_grid(legacy_resolver) -> None:
    session = _FakeSession()
    request = _noscan_request(
        mode="step",
        axes=[
            {"variable": "jet_z", "positions": {"start": 0, "end": 1, "step": 1}},
            {"variable": "jet_x", "positions": {"values": [4.0, 5.0, 6.0]}},
        ],
    )
    run_request(session, request, legacy_resolver)

    kwargs = session.scan_kwargs
    # N movables, outermost axis first.
    assert [m._geecs_device_name for m in kwargs["motor"]] == [
        "U_ESP_JetXYZ:Position.Axis 3",
        "U_ESP_JetXYZ:Position.Axis 1",
    ]
    # Outer product in list order: first axis outermost/slowest.
    assert kwargs["positions"] == [
        (0.0, 4.0),
        (0.0, 5.0),
        (0.0, 6.0),
        (1.0, 4.0),
        (1.0, 5.0),
        (1.0, 6.0),
    ]
    # ScanInfo carries both targets; its 1-D fields describe the outer axis.
    info = kwargs["scan_info_overrides"]
    assert info["scan_parameter"] == (
        "U_ESP_JetXYZ:Position.Axis 3,U_ESP_JetXYZ:Position.Axis 1"
    )
    assert (info["start"], info["end"], info["step"]) == (0.0, 1.0, 1.0)
    # Run metadata carries the axes and grid shape.
    md = kwargs["md"]
    assert md["scan_axes"] == ["jet_z", "jet_x"]
    assert md["grid_shape"] == [2, 3]
    assert md["num_grid_points"] == 6
    assert md["scan_variable"] == "jet_z,jet_x"


def test_single_axis_request_shape_is_unchanged_by_grid_support(
    legacy_resolver,
) -> None:
    """Regression: one axis still passes a bare motor + flat float positions."""
    session = _FakeSession()
    request = _noscan_request(
        mode="step",
        axes=[{"variable": "jet_z", "positions": {"start": 0, "end": 1, "step": 0.5}}],
    )
    run_request(session, request, legacy_resolver)
    kwargs = session.scan_kwargs
    assert not isinstance(kwargs["motor"], list)
    assert kwargs["positions"] == [0.0, 0.5, 1.0]
    assert "scan_axes" not in kwargs["md"]


# ---------------------------------------------------------------------------
# Action execution wiring (setup / per_step / closeout compiled + passed)
# ---------------------------------------------------------------------------


def test_actions_compile_into_session_scan_hooks(legacy_resolver) -> None:
    session = _FakeSession()
    request = _noscan_request(
        actions={
            "setup": ["scan_prep"],
            "per_step": ["between_steps"],
            "closeout": ["scan_cleanup"],
        }
    )
    run_request(session, request, legacy_resolver)

    kwargs = session.scan_kwargs
    # Each hook is a plan-stub callable yielding the compiled steps.
    assert _set_targets(kwargs["setup"]()) == [("U_PLC-DO.Ch2", "on")]
    assert _set_targets(kwargs["per_step"]()) == [("U_PLC-DO.Ch3", "on")]
    assert _set_targets(kwargs["closeout"]()) == [("U_PLC-DO.Ch4", "off")]
    # Reusable: per_step must produce a fresh generator per step boundary.
    assert _set_targets(kwargs["per_step"]()) == [("U_PLC-DO.Ch3", "on")]
    # Provenance: the assembled slot order lands in the run metadata.
    assert kwargs["md"]["action_plans"] == {
        "setup": ["scan_prep"],
        "per_step": ["between_steps"],
        "closeout": ["scan_cleanup"],
    }
    # Signals were prefetched (connected pre-claim) on the session's factory,
    # and the factory rides the plan's finalize disconnect.
    (factory,) = session.action_factories
    assert set(factory.settables) == {
        ("U_PLC", "DO.Ch2"),
        ("U_PLC", "DO.Ch3"),
        ("U_PLC", "DO.Ch4"),
    }
    assert factory.disconnected is True


def test_request_without_actions_passes_no_hooks(legacy_resolver) -> None:
    session = _FakeSession()
    run_request(session, _noscan_request(), legacy_resolver)
    kwargs = session.scan_kwargs
    assert kwargs["setup"] is None
    assert kwargs["per_step"] is None
    assert kwargs["closeout"] is None
    assert session.action_factories == []  # no factory built for nothing
    assert "action_plans" not in kwargs["md"]


def test_unknown_action_name_fails_validation_first(legacy_resolver) -> None:
    session = _FakeSession()
    request = _noscan_request(actions={"closeout": ["not_a_plan"]})
    with pytest.raises(GeecsConfigurationError, match="not_a_plan"):
        run_request(session, request, legacy_resolver)
    assert session.devices == []  # failed before any hardware was touched


def test_pseudo_variable_builds_movable_and_records_metadata(modern_resolver) -> None:
    """A pseudo step axis executes: compiled components, movable, provenance."""
    session = _FakeSession()
    request = _noscan_request(
        mode="step",
        save_sets=["NewSet"],
        axes=[{"variable": "bump_x", "positions": {"start": 0, "end": 1, "step": 1}}],
    )
    run_request(session, request, modern_resolver)
    assert session.pseudo_calls == [
        (
            "bump_x",
            [("U_S3H", "Current", "x * 1"), ("U_S4H", "Current", "x * -2")],
            "relative",
        )
    ]
    kwargs = session.scan_kwargs
    assert kwargs["motor"].kind == "pseudo_movable"
    assert kwargs["scan_info_overrides"]["scan_parameter"] == "bump_x"
    assert kwargs["md"]["pseudo_variables"] == {
        "bump_x": {
            "mode": "relative",
            "targets": [
                {"target": "U_S3H:Current", "forward": "x * 1"},
                {"target": "U_S4H:Current", "forward": "x * -2"},
            ],
        }
    }


def test_pseudo_variable_bad_expression_fails_preclaim(modern_resolver) -> None:
    """A forward formula that fails to compile aborts before any hardware."""
    session = _FakeSession()
    request = _noscan_request(
        mode="step",
        save_sets=["NewSet"],
        axes=[{"variable": "badcombo", "positions": {"start": 0, "end": 1, "step": 1}}],
    )
    with pytest.raises(GeecsConfigurationError, match="nonsense_name"):
        run_request(session, request, modern_resolver)
    assert session.devices == []  # failed before any hardware was touched


def test_noscan_without_save_set_is_rejected(legacy_resolver) -> None:
    with pytest.raises(GeecsConfigurationError, match="save set"):
        run_request(_FakeSession(), _noscan_request(save_sets=[]), legacy_resolver)


def _optimize_request(**overrides) -> ScanRequest:
    base = dict(
        mode="optimize",
        shots_per_step=5,
        acquisition="free_run",
        save_sets=["UC_Test"],
        optimization={
            "variables": {"jet_z": [0.0, 1.0], "U_S1H:Current": [-2.0, 2.0]},
            "objectives": {"counts": "MAXIMIZE"},
            "evaluator": {"module": "m", "class": "C"},
            "generator": {"name": "bayes_default"},
            "max_iterations": 7,
            "move_to_best_on_finish": True,
        },
    )
    base.update(overrides)
    return ScanRequest.model_validate(base)


# ---------------------------------------------------------------------------
# Optimizer device_requirements auto-provisioning (the #520 reversal)
# ---------------------------------------------------------------------------


_TOPVIEW_REQUIREMENTS = {
    "Devices": {
        "UC_TopView": {
            "add_all_variables": False,
            "save_nonscalar_data": True,
            "synchronous": True,
            "variable_list": ["acq_timestamp"],
        }
    }
}


class TestMergeOptimizerDeviceRequirements:
    """The pure merge: union semantics, casefold matching, provenance."""

    def test_none_and_empty_are_no_ops(self) -> None:
        config = {"U_Cam": {"variable_list": ["a"]}}
        for requirements in (None, {}, {"Devices": {}}, "not-a-mapping"):
            before = {k: dict(v) for k, v in config.items()}
            assert merge_optimizer_device_requirements(config, requirements) == {}
            assert config == before

    def test_new_device_appended_after_save_set_devices(self) -> None:
        config = {
            "U_Ref": {
                "synchronous": True,
                "save_nonscalar_data": True,
                "variable_list": ["acq_timestamp"],
            }
        }
        provisioned = merge_optimizer_device_requirements(config, _TOPVIEW_REQUIREMENTS)
        # Appended after the save-set devices: the pacemaker is unchanged.
        assert list(config) == ["U_Ref", "UC_TopView"]
        assert config["UC_TopView"] == {
            "synchronous": True,
            "save_nonscalar_data": True,
            "variable_list": ["acq_timestamp"],
        }
        assert provisioned == {
            "UC_TopView": {
                "synchronous": True,
                "save_nonscalar_data": True,
                "variable_list": ["acq_timestamp"],
            }
        }

    def test_existing_device_unions_variables_and_ors_save_flag(self) -> None:
        config = {
            "UC_TopView": {
                "synchronous": True,
                "save_nonscalar_data": False,
                "variable_list": ["MaxCounts"],
            }
        }
        requirements = {
            "Devices": {
                "UC_TopView": {
                    "synchronous": True,
                    "save_nonscalar_data": True,
                    "variable_list": ["acq_timestamp", "MaxCounts"],
                }
            }
        }
        provisioned = merge_optimizer_device_requirements(config, requirements)
        # Save-set variables stay first; only the missing ones append.
        assert config["UC_TopView"]["variable_list"] == ["MaxCounts", "acq_timestamp"]
        assert config["UC_TopView"]["save_nonscalar_data"] is True
        assert provisioned == {
            "UC_TopView": {
                "variable_list": ["acq_timestamp"],
                "save_nonscalar_data": True,
            }
        }

    def test_existing_device_keeps_save_set_synchronous_semantics(self) -> None:
        # The save sets own the acquisition-role semantics: a snapshot
        # (asynchronous) entry is never flipped synchronous by a requirement.
        config = {
            "U_Slow": {
                "synchronous": False,
                "save_nonscalar_data": False,
                "variable_list": ["Pressure"],
            }
        }
        requirements = {
            "Devices": {"U_Slow": {"synchronous": True, "variable_list": ["Pressure"]}}
        }
        provisioned = merge_optimizer_device_requirements(config, requirements)
        assert config["U_Slow"]["synchronous"] is False
        assert provisioned == {}  # nothing was actually added

    def test_case_insensitive_match_merges_under_configured_spelling(self) -> None:
        config = {
            "UC_TopView": {
                "synchronous": True,
                "save_nonscalar_data": True,
                "variable_list": ["MaxCounts"],
            }
        }
        requirements = {"Devices": {"uc_topview": {"variable_list": ["acq_timestamp"]}}}
        provisioned = merge_optimizer_device_requirements(config, requirements)
        assert list(config) == ["UC_TopView"]  # no wrong-case duplicate entry
        assert config["UC_TopView"]["variable_list"] == ["MaxCounts", "acq_timestamp"]
        assert provisioned == {"UC_TopView": {"variable_list": ["acq_timestamp"]}}

    def test_duck_typed_attribute_requirements(self) -> None:
        # Requirement entries may be objects, not dicts (duck-typed).
        entry = SimpleNamespace(
            synchronous=True,
            save_nonscalar_data=True,
            variable_list=["acq_timestamp"],
        )
        config: dict = {}
        provisioned = merge_optimizer_device_requirements(
            config, {"Devices": {"UC_TopView": entry}}
        )
        assert config["UC_TopView"]["variable_list"] == ["acq_timestamp"]
        assert provisioned["UC_TopView"]["synchronous"] is True


# ---------------------------------------------------------------------------
# Multi-device trigger profiles: ordered writes, single-device regression
# ---------------------------------------------------------------------------


def test_single_device_profile_adapts_unchanged() -> None:
    """Regression: one-device profiles carry the same writes as before."""
    profile = TriggerProfile(
        name="new-shape",
        states={
            "SCAN": [
                {"device": "U_DG645", "variable": "Trigger.Source", "value": "Ext"},
                {"device": "U_DG645", "variable": "Amplitude.Ch AB", "value": "4.0"},
            ],
            "STANDBY": [
                {"device": "U_DG645", "variable": "Amplitude.Ch AB", "value": "0.5"},
            ],
        },
    )
    writes = trigger_writes_from_profile(profile)
    assert writes.devices == ["U_DG645"]
    assert writes.writes_for_state("SCAN") == [
        ("U_DG645", "Trigger.Source", "Ext"),
        ("U_DG645", "Amplitude.Ch AB", "4.0"),
    ]
    assert writes.writes_for_state("STANDBY") == [("U_DG645", "Amplitude.Ch AB", "0.5")]
    assert not writes.defines_state("OFF")


def test_multi_device_writes_preserve_declared_order() -> None:
    """A transition spanning devices keeps the profile's write order."""
    profile = TriggerProfile(
        name="spans-devices",
        states={
            "SCAN": [
                {"device": "U_DG645", "variable": "Trigger.Source", "value": "Ext"},
                {"device": "U_PLC", "variable": "DO.Ch9", "value": "on"},
            ],
            "STANDBY": [
                {"device": "U_PLC", "variable": "DO.Ch9", "value": "off"},
                {"device": "U_DG645", "variable": "Trigger.Source", "value": "Int"},
            ],
        },
    )
    writes = trigger_writes_from_profile(profile)
    assert set(writes.devices) == {"U_DG645", "U_PLC"}
    assert writes.writes_for_state("SCAN") == [
        ("U_DG645", "Trigger.Source", "Ext"),
        ("U_PLC", "DO.Ch9", "on"),
    ]
    # STANDBY declares the reverse device order — preserved verbatim.
    assert writes.writes_for_state("STANDBY") == [
        ("U_PLC", "DO.Ch9", "off"),
        ("U_DG645", "Trigger.Source", "Int"),
    ]


def test_multi_device_profile_runs_through_the_request(legacy_resolver) -> None:
    """A request naming a multi-device profile executes with zero refusals."""
    session = _FakeSession()
    profile = TriggerProfile(
        name="spans",
        states={
            "SCAN": [
                {"device": "U_DG645", "variable": "Amplitude.Ch AB", "value": "4.0"},
                {"device": "U_PLC", "variable": "DO.Ch9", "value": "on"},
            ],
        },
    )

    class _Resolver:
        def resolve_save_set(self, name):
            return legacy_resolver.resolve_save_set(name)

        def resolve_trigger_profile(self, name):
            return profile

        def resolve_action_plan(self, name):
            raise GeecsConfigurationError(name)

        def resolve_scan_variable(self, name):
            return legacy_resolver.resolve_scan_variable(name)

    run_request(session, _noscan_request(trigger_profile="spans"), _Resolver())
    writes = session.scan_kwargs["controller"]._writes
    assert writes.devices == ["U_DG645", "U_PLC"]


def test_profile_without_any_device_is_rejected() -> None:
    profile = TriggerProfile(name="empty", states={})
    with pytest.raises(GeecsConfigurationError, match="names no trigger device"):
        trigger_writes_from_profile(profile)


# ---------------------------------------------------------------------------
# Experiment defaults: fill unset fields, record provenance
# ---------------------------------------------------------------------------


def test_apply_experiment_defaults_fills_unset_fields() -> None:
    request = _noscan_request()  # no trigger_profile, no actions
    defaults = {
        "trigger_profile": "HTU-Normal",
        "actions": {"setup": ["prep"], "closeout": ["shutdown"]},
    }
    updated, applied = apply_experiment_defaults(request, defaults)
    assert updated.capture.trigger_profile == "HTU-Normal"
    assert updated.actions.setup == ["prep"]
    assert updated.actions.closeout == ["shutdown"]
    assert applied == {
        "trigger_profile": "HTU-Normal",
        "actions.setup": ["prep"],
        "actions.closeout": ["shutdown"],
    }
    # The original request is never mutated.
    assert request.capture.trigger_profile is None
    assert request.actions.setup == []


def test_apply_experiment_defaults_brackets_the_scans_own_plans() -> None:
    """Mirrored merge: defaults prepend to setup, append to closeout."""
    request = _noscan_request(
        actions={"setup": ["scan_prep"], "closeout": ["scan_cleanup"]}
    )
    defaults = {"actions": {"setup": ["default_prep"], "closeout": ["default_cleanup"]}}
    updated, applied = apply_experiment_defaults(request, defaults)
    assert updated.actions.setup == ["default_prep", "scan_prep"]
    assert updated.actions.closeout == ["scan_cleanup", "default_cleanup"]
    assert applied == {
        "actions.setup": ["default_prep"],
        "actions.closeout": ["default_cleanup"],
    }


def test_assemble_action_slots_layers_nest_like_context_managers() -> None:
    """setup: defaults → rituals → scan's own; closeout: exact reverse."""
    request = _noscan_request(
        actions={
            "setup": ["scan_prep"],
            "per_step": ["between_steps"],
            "closeout": ["scan_cleanup"],
        }
    )
    defaults = {"actions": {"setup": ["default_prep"], "closeout": ["default_cleanup"]}}
    merged, applied = apply_experiment_defaults(request, defaults)
    rituals = {"setup": ["cam_ritual"], "closeout": ["cam_park"]}
    slots = assemble_action_slots(merged.actions, applied, rituals)
    assert slots == {
        "setup": ["default_prep", "cam_ritual", "scan_prep"],
        "per_step": ["between_steps"],
        "closeout": ["scan_cleanup", "cam_park", "default_cleanup"],
    }


def test_assemble_action_slots_without_defaults_or_rituals() -> None:
    request = _noscan_request(actions={"setup": ["scan_prep"]})
    slots = assemble_action_slots(request.actions, {}, {"setup": [], "closeout": []})
    assert slots == {"setup": ["scan_prep"], "per_step": [], "closeout": []}


def test_apply_experiment_defaults_none_is_a_noop() -> None:
    request = _noscan_request()
    updated, applied = apply_experiment_defaults(request, None)
    assert updated is request
    assert applied == {}


def test_resolver_defaults_absent_file_returns_none(legacy_resolver) -> None:
    assert legacy_resolver.resolve_experiment_defaults() is None


def test_resolver_defaults_validate_against_the_model(
    configs_root, legacy_resolver
) -> None:
    (configs_root / "LegacyExp" / "experiment_defaults.yaml").write_text(
        "trigger_profile: HTU-Normal\nactions:\n  setup: [close_shutters]\n"
    )
    defaults = legacy_resolver.resolve_experiment_defaults()
    assert isinstance(defaults, ExperimentDefaults)
    assert defaults.trigger_profile == "HTU-Normal"
    assert defaults.actions.setup == ["close_shutters"]


def test_run_applies_defaults_and_records_provenance(
    configs_root, legacy_resolver
) -> None:
    """A defaults file supplies the trigger profile; the run records it."""
    (configs_root / "LegacyExp" / "experiment_defaults.yaml").write_text(
        "trigger_profile: HTU-Normal\n"
    )
    session = _FakeSession()
    run_request(session, _noscan_request(), legacy_resolver)

    writes = session.scan_kwargs["controller"]._writes
    assert isinstance(writes, ShotControlWrites)
    assert writes.devices == ["U_DG645_ShotControl"]
    assert session.scan_kwargs["md"]["applied_defaults"] == [
        {"field": "trigger_profile", "value": "HTU-Normal"}
    ]


def test_default_actions_execute_bracketing_the_scans_own(
    configs_root, legacy_resolver
) -> None:
    """Defaults-supplied plans run first on setup and last on closeout."""
    (configs_root / "LegacyExp" / "experiment_defaults.yaml").write_text(
        "actions:\n  setup: [default_prep]\n  closeout: [default_cleanup]\n"
    )
    session = _FakeSession()
    request = _noscan_request(
        actions={"setup": ["scan_prep"], "closeout": ["scan_cleanup"]}
    )
    run_request(session, request, legacy_resolver)

    kwargs = session.scan_kwargs
    assert _set_targets(kwargs["setup"]()) == [
        ("U_PLC-DO.Ch1", "on"),  # default_prep first
        ("U_PLC-DO.Ch2", "on"),  # then the scan's own
    ]
    assert _set_targets(kwargs["closeout"]()) == [
        ("U_PLC-DO.Ch4", "off"),  # the scan's own first
        ("U_PLC-DO.Ch5", "off"),  # defaults last (outermost bracket)
    ]
    assert kwargs["md"]["action_plans"] == {
        "setup": ["default_prep", "scan_prep"],
        "closeout": ["scan_cleanup", "default_cleanup"],
    }


# ---------------------------------------------------------------------------
# SaveSet entry-level setup/closeout rituals (collected, de-duplicated, run)
# ---------------------------------------------------------------------------


class _SaveSetResolver:
    """Stub resolver serving one duck-typed save set + one known plan."""

    def __init__(self, save_set, known_plans=("prep_cam",)) -> None:
        self._save_set = save_set
        self._known = set(known_plans)

    def resolve_save_set(self, name):
        return self._save_set

    def resolve_action_plan(self, name):
        if name not in self._known:
            raise GeecsConfigurationError(f"action plan {name!r} not found")
        return ActionPlan.model_validate({"steps": [{"do": "wait", "seconds": 1.0}]})


def test_entry_level_actions_collected() -> None:
    save_set = SaveSet(
        name="s",
        entries=[
            SaveSetEntry(
                device="U_A", scalars=["x"], setup=["prep_cam"], closeout=["park"]
            ),
            SaveSetEntry(device="U_B", scalars=["y"]),
        ],
    )
    assert collect_save_set_rituals(save_set) == {
        "setup": ["prep_cam"],
        "closeout": ["park"],
    }
    # Entries without references contribute nothing.
    plain = SaveSet(name="s", entries=[SaveSetEntry(device="U_A", scalars=["x"])])
    assert collect_save_set_rituals(plain) == {"setup": [], "closeout": []}


def test_entry_rituals_deduplicate_across_entries() -> None:
    """Two entries naming the same ritual run it once (schema contract)."""
    save_set = SaveSet(
        name="s",
        entries=[
            SaveSetEntry(device="U_A", scalars=["x"], setup=["prep", "align"]),
            SaveSetEntry(device="U_B", scalars=["y"], setup=["prep"]),
        ],
    )
    assert collect_save_set_rituals(save_set)["setup"] == ["prep", "align"]


def _entry_action_save_set(**entry_overrides) -> SaveSet:
    entry = dict(device="U_A", scalars=["x"])
    entry.update(entry_overrides)
    return SaveSet.model_validate({"name": "s", "entries": [entry]})


def test_resolve_save_sets_and_rituals_validates_names() -> None:
    resolver = _SaveSetResolver(_entry_action_save_set(setup=["prep_cam"]))
    save_set, rituals = resolve_save_sets_and_rituals(resolver, ["s"])
    assert rituals == {"setup": ["prep_cam"], "closeout": []}


def test_entry_level_unknown_action_fails_validation_first() -> None:
    resolver = _SaveSetResolver(_entry_action_save_set(setup=["nope"]))
    with pytest.raises(GeecsConfigurationError, match="nope"):
        resolve_save_sets_and_rituals(resolver, ["s"])


def test_entry_rituals_execute_between_defaults_and_request(
    legacy_resolver,
) -> None:
    """Rituals from RitualSet land between defaults and the request's own
    plans on setup, and mirrored on closeout — with the shared ritual
    de-duplicated (both entries name cam_ritual; it runs once)."""
    session = _FakeSession()
    request = _noscan_request(
        save_sets=["RitualSet"],
        actions={"setup": ["scan_prep"], "closeout": ["scan_cleanup"]},
    )
    run_request(session, request, legacy_resolver)

    kwargs = session.scan_kwargs
    assert kwargs["md"]["action_plans"] == {
        "setup": ["cam_ritual", "scan_prep"],
        "closeout": ["scan_cleanup", "cam_park"],
    }
    assert _set_targets(kwargs["setup"]()) == [
        ("U_Cam-Analysis", "on"),  # the entries' ritual, once
        ("U_PLC-DO.Ch2", "on"),  # then the scan's own setup
    ]
    assert _set_targets(kwargs["closeout"]()) == [
        ("U_PLC-DO.Ch4", "off"),  # the scan's own closeout first
        ("U_Cam-Analysis", "off"),  # then the entries' ritual
    ]


def test_converted_element_actions_execute(legacy_resolver) -> None:
    """A legacy element's setup_action converts to an entry ritual that the
    runner compiles and executes (the extracted plan resolves by name)."""
    session = _FakeSession()
    request = _noscan_request(save_sets=["UC_WithActions"])
    run_request(session, request, legacy_resolver)

    kwargs = session.scan_kwargs
    assert kwargs["md"]["action_plans"]["setup"] == ["UC_WithActions_setup"]
    assert _set_targets(kwargs["setup"]()) == [("U_PLC-DO.Ch1", "on")]
    assert kwargs["closeout"] is None


# ---------------------------------------------------------------------------
# End to end: axes + actions + multi-device profile, zero NotImplementedError
# ---------------------------------------------------------------------------

MULTI_DEVICE_PROFILE = """\
schema_version: 1
name: spans
states:
  SCAN:
    - {device: U_DG645, variable: Amplitude.Ch AB, value: "4.0"}
    - {device: U_PLC, variable: DO.Ch7, value: "on"}
  STANDBY:
    - {device: U_PLC, variable: DO.Ch7, value: "off"}
    - {device: U_DG645, variable: Amplitude.Ch AB, value: "0.5"}
"""


def test_full_fake_session_flow_axes_actions_multi_device_trigger(
    configs_root, legacy_resolver
) -> None:
    """The M3b acceptance flow: a ScanRequest carrying a 2-axis grid, all
    three action slots, entry rituals, experiment defaults, and a
    multi-device trigger profile drives the whole fake-session flow with
    zero NotImplementedErrors."""
    (
        configs_root / "LegacyExp" / "shot_control_configurations" / "Spans.yaml"
    ).write_text(MULTI_DEVICE_PROFILE)
    (configs_root / "LegacyExp" / "experiment_defaults.yaml").write_text(
        "actions:\n  setup: [default_prep]\n  closeout: [default_cleanup]\n"
    )
    session = _FakeSession()
    request = ScanRequest.model_validate(
        {
            "mode": "step",
            "shots_per_step": 2,
            "acquisition": "free_run",
            "save_sets": ["RitualSet"],
            "trigger_profile": "Spans",
            "axes": [
                {"variable": "jet_z", "positions": {"start": 0, "end": 1, "step": 1}},
                {"variable": "jet_x", "positions": {"values": [4.0, 5.0]}},
            ],
            "actions": {
                "setup": ["scan_prep"],
                "per_step": ["between_steps"],
                "closeout": ["scan_cleanup"],
            },
            "description": "m3b acceptance",
        }
    )

    uid = run_request(session, request, legacy_resolver)

    assert uid == "uid-scan"
    # Multi-device trigger built worker-side as ordered writes.
    writes = session.scan_kwargs["controller"]._writes
    assert isinstance(writes, ShotControlWrites)
    assert set(writes.devices) == {"U_DG645", "U_PLC"}
    assert writes.writes_for_state("STANDBY") == [
        ("U_PLC", "DO.Ch7", "off"),
        ("U_DG645", "Amplitude.Ch AB", "0.5"),
    ]
    kwargs = session.scan_kwargs
    # 2-axis grid: 2 × 2 grid points, tuples, both movables.
    assert len(kwargs["positions"]) == 4
    assert kwargs["md"]["grid_shape"] == [2, 2]
    assert [m.kind for m in kwargs["motor"]] == ["settable", "settable"]
    # All four layers assembled in nesting order (defaults outermost).
    assert kwargs["md"]["action_plans"] == {
        "setup": ["default_prep", "cam_ritual", "scan_prep"],
        "per_step": ["between_steps"],
        "closeout": ["scan_cleanup", "cam_park", "default_cleanup"],
    }
    assert _set_targets(kwargs["setup"]()) == [
        ("U_PLC-DO.Ch1", "on"),
        ("U_Cam-Analysis", "on"),
        ("U_PLC-DO.Ch2", "on"),
    ]
    assert _set_targets(kwargs["closeout"]()) == [
        ("U_PLC-DO.Ch4", "off"),
        ("U_Cam-Analysis", "off"),
        ("U_PLC-DO.Ch5", "off"),
    ]
    # Provenance of the applied defaults.
    assert kwargs["md"]["applied_defaults"] == [
        {"field": "actions.setup", "value": ["default_prep"]},
        {"field": "actions.closeout", "value": ["default_cleanup"]},
    ]
    # Cleanup: the action signal factory rides the plan's finalize disconnect.
    (factory,) = session.action_factories
    assert factory.disconnected is True


# ---------------------------------------------------------------------------
# Lazy action-plan registry: a real fault must not masquerade as "not found"
# ---------------------------------------------------------------------------


def test_lazy_registry_propagates_unexpected_resolver_faults() -> None:
    """A non-"not found" fault propagates; only an unknown name is a miss.

    The lazy registry converts a genuine "plan not in the library"
    (``GeecsConfigurationError``) into ``KeyError`` for the compiler, but any
    other fault (a resolver bug, transient IO) must surface — masking it as a
    miss would misdirect debugging to "plan not found" with no candidates.
    """

    class _BoomResolver:
        def resolve_action_plan(self, name: str):
            if name == "missing":
                raise GeecsConfigurationError("not in library")
            raise RuntimeError("resolver exploded")

    registry = build_action_registry(_BoomResolver())
    # Unknown name → KeyError (the compiler's "not found" path).
    with pytest.raises(KeyError):
        registry["missing"]
    # Any other fault propagates unchanged.
    with pytest.raises(RuntimeError, match="exploded"):
        registry["anything_else"]
    assert registry.get("missing", "default") == "default"
    with pytest.raises(RuntimeError, match="exploded"):
        registry.get("anything_else")


# ---------------------------------------------------------------------------
# M3c: DB-integration runtime (get-side: db_scalars + telemetry; set-side
# disabled — reserved fields warn and are not applied)
# ---------------------------------------------------------------------------


class _M3cPolicy:
    """In-memory get-side ScalarPolicyProvider for the runner integration tests."""

    def __init__(self, subscribed=None, all_vars=None) -> None:
        self._subscribed = subscribed or {}
        self._all = all_vars or {}

    def get_variables(self, device):
        return list(self._subscribed.get(device, []))

    def all_variables(self, device):
        return list(self._all.get(device, []))

    def subscribed_by_device(self):
        return dict(self._subscribed)


class _M3cSession(_FakeSession):
    """Fake session exposing an experiment name (a DB-policy session shape).

    The policy itself is installed by :func:`_install_policy`; telemetry
    connects are in-plan (real CA-mock devices) and are pinned in
    ``test_scan_request_plan.py``.
    """

    experiment = "TestExp"


def _install_policy(monkeypatch, policy) -> None:
    """Force the preamble to use *policy* instead of a real GeecsDb.

    Also stubs out the served-set provider (the unserved-variables
    pre-flight): a session exposing ``experiment`` would otherwise reach for
    the real DB — these tests must stay hermetic (and never stall on an
    off-network MySQL timeout).  The plan module binds ``make_scalar_policy``
    by name, so both modules are patched.
    """
    import geecs_bluesky.plans.scan_request_plan as plan_module
    import geecs_bluesky.scan_request_runner as runner

    monkeypatch.setattr(runner, "make_scalar_policy", lambda session: policy)
    monkeypatch.setattr(plan_module, "make_scalar_policy", lambda session: policy)
    monkeypatch.setattr(runner, "make_served_set_provider", lambda session: None)


def _db_noscan_request(**overrides):
    base = dict(
        mode="noscan",
        shots_per_step=2,
        acquisition="strict",
        trigger_profile="Strict",
        save_sets=["UC_Test"],
    )
    base.update(overrides)
    return ScanRequest.model_validate(base)


def test_db_scalars_union_reaches_devices_config(monkeypatch, legacy_resolver) -> None:
    # UC_Test's converted element pins db_scalars=False per device, so its
    # recorded list stays explicit-only.  Verify the resolver-level policy is
    # threaded by checking a save set with db_scalars left at the True default.
    policy = _M3cPolicy(subscribed={"U_Cam": ["MaxCounts", "centroidx"]})
    save_set = SaveSet(
        name="s",
        entries=[SaveSetEntry(device="U_Cam", scalars=["Extra"])],  # db_scalars=True
    )
    config = save_set_to_devices_config(save_set, policy)
    assert config["U_Cam"]["variable_list"] == ["MaxCounts", "centroidx", "Extra"]


def test_converted_legacy_element_pins_db_scalars_false(legacy_resolver) -> None:
    # The legacy converter sets db_scalars=False, so even with a policy the
    # recorded scalars are exactly the element's explicit variable_list.
    policy = _M3cPolicy(subscribed={"U_Cam": ["ShouldNotAppear"]})
    save_set = legacy_resolver.resolve_save_set("UC_Test")
    config = save_set_to_devices_config(save_set, policy)
    assert "ShouldNotAppear" not in config["U_Cam"]["variable_list"]
    assert config["U_Cam"]["variable_list"] == ["MaxCounts"]


def test_reserved_boundary_fields_warn_and_are_not_applied(
    monkeypatch, legacy_resolver, caplog
) -> None:
    """A SaveSet entry that sets the reserved set-side fields is inert + warned.

    The DB set-side (scan start/end writes) is disabled in this version.  An
    entry that still carries ``at_scan_start`` / ``at_scan_end`` must NOT
    produce any boundary write, must NOT chain anything into the
    setup/closeout hooks, and must NOT record ``db_scan_writes`` metadata —
    but the operator gets exactly one WARNING naming the device so they know
    the values are inert.  The scan itself still runs.
    """
    _install_policy(monkeypatch, _M3cPolicy())
    session = _M3cSession()
    save_set = SaveSet(
        name="ReservedSet",
        entries=[
            SaveSetEntry(
                device="U_DG645_ShotControl",
                scalars=["x"],
                at_scan_start={"Trigger.Source": "External"},
                at_scan_end={"Amplitude.Ch AB": "0"},
            )
        ],
    )
    monkeypatch.setattr(legacy_resolver, "resolve_save_set", lambda name: save_set)

    with caplog.at_level(logging.WARNING):
        run_request(session, _db_noscan_request(save_sets=["X"]), legacy_resolver)

    # Exactly one reserved-not-honored warning, naming the device.
    reserved = [
        r
        for r in caplog.records
        if "reserved DB scan start/end fields" in r.getMessage()
    ]
    assert len(reserved) == 1
    assert "U_DG645_ShotControl" in reserved[0].getMessage()

    # The scan ran, but nothing was chained and no set-side metadata recorded.
    kwargs = session.scan_kwargs
    assert kwargs["setup"] is None
    assert kwargs["closeout"] is None
    assert "db_scan_writes" not in kwargs["md"]
    assert "db_scan_runtime" not in kwargs["md"]


def test_no_provider_leaves_m3b_behavior_unchanged(legacy_resolver) -> None:
    # A session with no experiment name → no policy → no DB writes, no
    # telemetry, explicit-only scalars (the M3b path, still green).
    session = _FakeSession()
    run_request(session, _db_noscan_request(), legacy_resolver)
    assert "db_scan_writes" not in session.scan_kwargs["md"]
    assert "background_telemetry" not in session.scan_kwargs["md"]


# ---------------------------------------------------------------------------
# M4: multiple save sets union into one effective device set
# ---------------------------------------------------------------------------


def test_merge_save_sets_unions_devices_and_merges_overlap() -> None:
    a = SaveSet(
        name="A",
        entries=[
            SaveSetEntry(device="U_Cam", scalars=["MaxCounts"], setup=["r1"]),
            SaveSetEntry(device="U_Slow", role="snapshot"),
        ],
    )
    b = SaveSet(
        name="B",
        entries=[
            SaveSetEntry(
                device="U_Cam",
                scalars=["Extra"],
                images=True,
                all_scalars=True,
                setup=["r1", "r2"],
            ),
            SaveSetEntry(device="U_Aux", scalars=["Aux1"]),
        ],
    )
    merged = merge_save_sets([a, b], name="merged")
    by_device = {e.device: e for e in merged.entries}
    # union of devices, first-appearance order across the list
    assert [e.device for e in merged.entries] == ["U_Cam", "U_Slow", "U_Aux"]
    cam = by_device["U_Cam"]
    # scalars union order-preserving/deduped, images + all_scalars OR True,
    # entry rituals unioned once (deduped)
    assert cam.scalars == ["MaxCounts", "Extra"]
    assert cam.images is True
    assert cam.all_scalars is True
    assert cam.setup == ["r1", "r2"]
    # role: first non-None kept (U_Slow's snapshot survives; U_Cam has none)
    assert cam.role is None
    assert by_device["U_Slow"].role.value == "snapshot"


def test_merge_save_sets_single_element_is_identity() -> None:
    only = SaveSet(name="s", entries=[SaveSetEntry(device="U_A", scalars=["x"])])
    assert merge_save_sets([only]) is only


def test_merge_save_sets_conflicting_roles_raise() -> None:
    """Same device with different explicit roles across sets is an error.

    Role wires the reference/contributor/snapshot semantics, so resolving it by
    save_sets list order would silently give the scan the wrong synchronization
    for a required device — refuse instead (review finding on #479).
    """
    a = SaveSet(name="A", entries=[SaveSetEntry(device="U_Cam", role="reference")])
    b = SaveSet(name="B", entries=[SaveSetEntry(device="U_Cam", role="snapshot")])
    with pytest.raises(GeecsConfigurationError, match="conflicting"):
        merge_save_sets([a, b])
    # Order must not change the outcome: the reverse also raises.
    with pytest.raises(GeecsConfigurationError, match="conflicting"):
        merge_save_sets([b, a])
    # Same role, or one side unset, is fine — no raise, the role survives.
    unset = SaveSet(name="C", entries=[SaveSetEntry(device="U_Cam")])
    merged = merge_save_sets([a, unset])
    assert merged.entries[0].role.value == "reference"


def test_two_save_sets_record_union_of_devices(legacy_resolver) -> None:
    session = _FakeSession()
    request = _noscan_request(save_sets=["UC_Test", "UC_Aux"])
    run_request(session, request, legacy_resolver)
    # UC_Test = {U_Cam(sync), U_Cam2(sync), U_Slow(async)}, UC_Aux adds U_Aux
    # (sync) and overlaps U_Cam (merged).  Free-run roles by position: first
    # sync = reference detector, later sync = contributor, async = snapshot.
    assert session.devices == [
        ("U_Cam", "detector"),
        ("U_Cam2", "contributor"),
        ("U_Aux", "contributor"),
        ("U_Slow", "snapshot"),
    ]
    # provenance: both named sets recorded
    assert session.scan_kwargs["md"]["save_sets"] == ["UC_Test", "UC_Aux"]


def test_two_save_sets_merge_overlapping_device_config(legacy_resolver) -> None:
    # U_Cam is in both UC_Test (MaxCounts, images) and UC_Aux (Extra, images):
    # the merged devices_config unions its scalars and keeps images on.
    merged, _rituals = resolve_save_sets_and_rituals(
        legacy_resolver, ["UC_Test", "UC_Aux"]
    )
    config = save_set_to_devices_config(merged)
    assert config["U_Cam"]["variable_list"] == ["MaxCounts", "Extra"]
    assert config["U_Cam"]["save_nonscalar_data"] is True
    assert "U_Aux" in config


def test_two_save_sets_ritual_deduped_once(legacy_resolver) -> None:
    # UC_Aux's U_Cam entry names cam_ritual; RitualSet also names it.  Across
    # the two named sets the ritual is collected once (deduped by plan name).
    _merged, rituals = resolve_save_sets_and_rituals(
        legacy_resolver, ["RitualSet", "UC_Aux"]
    )
    assert rituals["setup"].count("cam_ritual") == 1


class _SaveRecordingSession(_FakeSession):
    """FakeSession that also records each detector's save_images flag.

    Deliberately carries no ``experiment`` name: that keeps the
    DB-backed preflights and providers inert (hermetic), so the toggle
    wiring is pinned by monkeypatching the selection seam — which is
    unit-tested against a fake provider in test_native_image_save.py.

    The plan's deferred-connect facade rebinds these factory methods to
    itself, so they call the base explicitly rather than through
    ``super()`` (whose ``self`` would be the facade).
    """

    def __init__(self) -> None:
        super().__init__()
        self.save_flags: dict[str, bool] = {}
        self.control_only_flags: dict[str, bool] = {}

    def detector(
        self,
        device,
        variables,
        *,
        save_images=False,
        save_control_only=False,
        name=None,
    ):
        self.save_flags[device] = save_images
        self.control_only_flags[device] = save_control_only
        return _FakeSession.detector(self, device, variables, save_images=save_images)

    def contributor(
        self,
        device,
        variables,
        *,
        save_images=False,
        save_control_only=False,
        name=None,
    ):
        self.save_flags[device] = save_images
        self.control_only_flags[device] = save_control_only
        return _FakeSession.contributor(
            self, device, variables, save_images=save_images
        )

    def snapshot(self, device, variables, *, save_control_only=False, name=None):
        self.control_only_flags[device] = save_control_only
        return _FakeSession.snapshot(self, device, variables)


def _select_u_cam(experiment, devices_config, *, provider=None):
    """Selection stand-in: U_Cam is the one capture-eligible camera."""
    return [d for d in devices_config if d == "U_Cam"]


def test_native_image_save_off_wires_through_runner(
    legacy_resolver, monkeypatch
) -> None:
    """Toggle-off end-to-end: only the registry-devicetype camera loses its
    native save; md carries the capture list; nothing else changes."""
    import geecs_bluesky.scan_request_runner as runner_mod

    monkeypatch.setattr(  # daemon heartbeat absent in tests — bypass refusal
        runner_mod, "preflight_capture_liveness", lambda *a, **k: None
    )
    monkeypatch.setattr(runner_mod, "select_capture_devices", _select_u_cam)
    session = _SaveRecordingSession()
    run_request(
        session,
        _noscan_request(
            acquisition="strict",
            trigger_profile="Strict",
            native_image_save=False,
        ),
        legacy_resolver,
    )
    # U_Cam is Point Grey → suppressed; U_Cam2 keeps whatever the save set said.
    assert session.save_flags["U_Cam"] is False
    assert session.control_only_flags["U_Cam"] is True  # active off-write surface
    md = session.scan_kwargs["md"]
    assert md["capture_devices"] == ["U_Cam"]
    assert md["native_image_save"] is False
    # Role/order untouched: same device list as the plain strict test.
    assert [d for d, _k in session.devices] == ["U_Cam", "U_Cam2", "U_Slow"]


def test_native_image_save_off_wires_contributor_branch(
    legacy_resolver, monkeypatch
) -> None:
    """Free-run: the non-reference contributor branch threads the flag too."""
    import geecs_bluesky.scan_request_runner as runner_mod

    def _select_u_cam2(experiment, devices_config, *, provider=None):
        return [d for d in devices_config if d == "U_Cam2"]

    monkeypatch.setattr(  # daemon heartbeat absent in tests — bypass refusal
        runner_mod, "preflight_capture_liveness", lambda *a, **k: None
    )
    monkeypatch.setattr(runner_mod, "select_capture_devices", _select_u_cam2)
    session = _SaveRecordingSession()
    run_request(
        session,
        _noscan_request(acquisition="free_run", native_image_save=False),
        legacy_resolver,
    )
    # U_Cam is the reference (detector); U_Cam2 becomes a contributor and
    # must carry the control-only flag through that branch.
    assert dict(session.devices)["U_Cam2"] == "contributor"
    assert session.control_only_flags["U_Cam2"] is True
    assert session.save_flags["U_Cam2"] is False


def _select_through_real_seam(monkeypatch, runner_mod, types: dict[str, str]):
    """Route the REAL selection seam through a fake devicetype provider.

    Unlike the ``_select_u_cam`` stand-ins this keeps the seam's own policy
    (devicetype + save flag + sync role) in the loop, so the async-role drop
    (#702) is exercised end to end through the runner, hermetically.
    """
    real = runner_mod.select_capture_devices

    class _Provider:
        @staticmethod
        def by_device():
            return dict(types)

    monkeypatch.setattr(
        runner_mod,
        "select_capture_devices",
        lambda experiment, devices_config, *, provider=None: real(
            experiment, devices_config, provider=_Provider()
        ),
    )


def _capture_drop_warnings(caplog, device: str) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
        and r.getMessage().startswith(f"{device} is NOT capture-owned")
    ]


def test_async_camera_without_scalars_is_dropped_from_capture(
    legacy_resolver, monkeypatch, caplog
) -> None:
    """#702 shape 1: an async capture-eligible camera with an EMPTY
    variable_list builds no device (the "no scalars" skip), so nothing could
    ever command its save off — it must land in neither ``capture_devices``
    nor the save_control_only config, loudly, and the toggle-off scan goes
    ahead on the sync camera alone."""
    import geecs_bluesky.scan_request_runner as runner_mod

    monkeypatch.setattr(  # daemon heartbeat absent in tests — bypass refusal
        runner_mod, "preflight_capture_liveness", lambda *a, **k: None
    )
    _select_through_real_seam(
        monkeypatch,
        runner_mod,
        {"U_Cam": "Point Grey Camera", "U_AsyncBare": "Point Grey Camera"},
    )
    session = _SaveRecordingSession()
    with caplog.at_level(logging.WARNING):
        run_request(
            session,
            _noscan_request(
                acquisition="strict",
                trigger_profile="Strict",
                save_sets=["UC_Test", "UC_AsyncCams"],
                native_image_save=False,
            ),
            legacy_resolver,
        )
    md = session.scan_kwargs["md"]
    assert md["capture_devices"] == ["U_Cam"]
    assert md["native_image_save"] is False
    assert "U_AsyncBare" not in dict(session.devices)  # still skipped: no scalars
    assert "U_AsyncBare" not in session.control_only_flags
    (message,) = _capture_drop_warnings(caplog, "U_AsyncBare")
    assert "Point Grey Camera" in message and "#702" in message
    assert "neither commands nor suppresses" in message
    assert "role: snapshot" in message


def test_async_camera_with_scalars_is_a_plain_snapshot_not_capture_owned(
    legacy_resolver, monkeypatch, caplog
) -> None:
    """#702 shape 2: an async capture-eligible camera WITH scalars used to
    get the save child but no acq_timestamp join column (an orphaned
    stack). It is now a plain snapshot: no control-only flag, absent from
    ``capture_devices``, warned about by name; the sync camera is still
    captured."""
    import geecs_bluesky.scan_request_runner as runner_mod

    monkeypatch.setattr(  # daemon heartbeat absent in tests — bypass refusal
        runner_mod, "preflight_capture_liveness", lambda *a, **k: None
    )
    _select_through_real_seam(
        monkeypatch,
        runner_mod,
        {"U_Cam": "Point Grey Camera", "U_AsyncCam": "Point Grey Camera"},
    )
    session = _SaveRecordingSession()
    with caplog.at_level(logging.WARNING):
        run_request(
            session,
            _noscan_request(
                acquisition="strict",
                trigger_profile="Strict",
                save_sets=["UC_Test", "UC_AsyncCams"],
                native_image_save=False,
            ),
            legacy_resolver,
        )
    assert dict(session.devices)["U_AsyncCam"] == "snapshot"
    assert session.control_only_flags["U_AsyncCam"] is False
    md = session.scan_kwargs["md"]
    assert md["capture_devices"] == ["U_Cam"]
    assert session.control_only_flags["U_Cam"] is True
    assert len(_capture_drop_warnings(caplog, "U_AsyncCam")) == 1


def test_toggle_off_with_only_async_eligible_camera_is_inert_and_unrefused(
    legacy_resolver, monkeypatch, caplog
) -> None:
    """When the ONLY capture-eligible camera is async, toggle-off resolves
    to no capture devices: the liveness preflight is never consulted (no
    daemon needed for a scan that captures nothing), no ``capture_devices``
    key is published, and the inert request stays visible as
    ``native_image_save: false`` alongside the existing no-eligible warning."""
    import geecs_bluesky.scan_request_runner as runner_mod

    preflight_calls: list = []
    monkeypatch.setattr(
        runner_mod,
        "preflight_capture_liveness",
        lambda *a, **k: preflight_calls.append(a),
    )
    _select_through_real_seam(
        monkeypatch, runner_mod, {"U_AsyncCam": "Point Grey Camera"}
    )
    session = _SaveRecordingSession()
    with caplog.at_level(logging.WARNING):
        run_request(
            session,
            _noscan_request(
                acquisition="strict",
                trigger_profile="Strict",
                save_sets=["UC_Test", "UC_AsyncCams"],
                native_image_save=False,
            ),
            legacy_resolver,
        )
    assert preflight_calls == []
    md = session.scan_kwargs["md"]
    assert "capture_devices" not in md
    assert md["native_image_save"] is False
    assert session.control_only_flags == {
        "U_Cam": False,
        "U_Cam2": False,
        "U_Slow": False,
        "U_AsyncCam": False,
    }
    assert len(_capture_drop_warnings(caplog, "U_AsyncCam")) == 1
    assert any(
        "no capture-eligible devices resolved" in r.getMessage() for r in caplog.records
    )


def test_native_image_save_on_leaves_saving_and_still_publishes_list(
    legacy_resolver, monkeypatch
) -> None:
    """Dual-write default: saving untouched, capture list still published."""
    import geecs_bluesky.scan_request_runner as runner_mod

    monkeypatch.setattr(  # daemon heartbeat absent in tests — bypass refusal
        runner_mod, "preflight_capture_liveness", lambda *a, **k: None
    )
    monkeypatch.setattr(runner_mod, "select_capture_devices", _select_u_cam)
    session = _SaveRecordingSession()
    run_request(
        session,
        _noscan_request(acquisition="strict", trigger_profile="Strict"),
        legacy_resolver,
    )
    md = session.scan_kwargs["md"]
    assert md["capture_devices"] == ["U_Cam"]
    assert md["native_image_save"] is True
    # The save set's own save flag is preserved (whatever it was).
    assert "U_Cam" in session.save_flags
