"""Tests for scan_request_runner: resolver, adapters, and run_scan_request.

Covers the configs-repo resolver (new-schema YAML loads directly, legacy YAML
converts — the whole existing corpus is usable immediately), the SaveSet →
devices_config derivation rules, the TriggerProfile → ShotControlWrites
adapter (ordered, multi-device), action slot assembly + compilation +
wiring, multi-axis grid execution, and the request execution mapping onto a
fake GeecsSession.  The remaining documented v1 gaps (pseudo variables,
``all_scalars``, optimize without injected callables) refuse loudly; optimize
*with* actions runs but skips the actions (logged + recorded in metadata),
since optimize has no action hooks yet.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from bluesky.utils import Msg

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlWrites
from geecs_bluesky.scan_request_runner import (
    ConfigResolver,
    ConfigsRepoResolver,
    apply_experiment_defaults,
    assemble_action_slots,
    build_action_registry,
    collect_save_set_rituals,
    merge_save_sets,
    resolve_save_sets_and_rituals,
    run_scan_request,
    save_set_to_devices_config,
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


class _FakeSession:
    def __init__(self) -> None:
        self.devices: list[tuple[str, str]] = []  # (device, factory)
        self.shot_control_calls: list = []
        self.scan_kwargs: dict | None = None
        self.optimize_kwargs: dict | None = None
        self.disconnected: list = []
        self.action_factories: list[_FakeActionFactory] = []
        self.confirm_settable_calls: list = []

    def _make(self, device: str, kind: str) -> _FakeDevice:
        self.devices.append((device, kind))
        return _FakeDevice(device, kind)

    def detector(self, device, variables, *, save_images=False, name=None):
        return self._make(device, "detector")

    def contributor(self, device, variables, *, save_images=False, name=None):
        return self._make(device, "contributor")

    def snapshot(self, device, variables, *, name=None):
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

    def action_signal_factory(self):
        factory = _FakeActionFactory()
        self.action_factories.append(factory)
        return factory

    def shot_control(self, config):
        self.shot_control_calls.append(config)

    def scan(self, **kwargs):
        self.scan_kwargs = kwargs
        return "uid-scan"

    def optimize(self, **kwargs):
        self.optimize_kwargs = kwargs
        return "uid-opt", []

    def disconnect(self, *devices):
        self.disconnected.extend(devices)


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
schema_version: 1
name: NewProfile
states:
  SCAN:
    - {device: U_DG645_ShotControl, variable: Trigger.Source, value: External rising edges}
variants:
  laser_off:
    states:
      SCAN:
        - {device: U_DG645_ShotControl, variable: Trigger.Source, value: Internal}
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
    (legacy / "shot_control_configurations").mkdir()
    (legacy / "shot_control_configurations" / "HTU-Normal.yaml").write_text(
        LEGACY_SHOT_CONTROL
    )
    (legacy / "shot_control_configurations" / "Empty.yaml").write_text("")
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


def test_new_schema_trigger_profile_with_variant(modern_resolver) -> None:
    profile = modern_resolver.resolve_trigger_profile("NewProfile")
    writes = profile.writes_for("SCAN", variant="laser_off")
    assert [(w.device, w.variable, w.value) for w in writes] == [
        ("U_DG645_ShotControl", "Trigger.Source", "Internal")
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


def test_trigger_adapter_applies_variant(modern_resolver) -> None:
    profile = modern_resolver.resolve_trigger_profile("NewProfile")
    writes = trigger_writes_from_profile(profile, "laser_off")
    assert writes.writes_for_state("SCAN") == [
        ("U_DG645_ShotControl", "Trigger.Source", "Internal")
    ]


def test_trigger_adapter_unknown_variant_raises(modern_resolver) -> None:
    profile = modern_resolver.resolve_trigger_profile("NewProfile")
    with pytest.raises(GeecsConfigurationError, match="laser_off"):
        trigger_writes_from_profile(profile, "nope")


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


# ---------------------------------------------------------------------------
# run_scan_request
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
    uid = run_scan_request(session, _noscan_request(), legacy_resolver)

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
    assert kwargs["mode"] == "free_run"
    assert kwargs["description"] == "stats"
    assert kwargs["scan_info"]["scan_mode"] == "noscan"
    assert kwargs["scan_info"]["background"] is False
    # no trigger profile named → shot control detached
    assert session.shot_control_calls == [None]
    # the run disconnects what it created
    assert len(session.disconnected) == 3


def test_strict_request_builds_all_sync_as_detectors(legacy_resolver) -> None:
    session = _FakeSession()
    run_scan_request(session, _noscan_request(acquisition="strict"), legacy_resolver)
    assert session.devices == [
        ("U_Cam", "detector"),
        ("U_Cam2", "detector"),
        ("U_Slow", "snapshot"),
    ]
    assert session.scan_kwargs["mode"] == "strict"


def test_trigger_profile_is_attached_via_the_adapter(legacy_resolver) -> None:
    session = _FakeSession()
    run_scan_request(
        session, _noscan_request(trigger_profile="HTU-Normal"), legacy_resolver
    )
    (writes,) = session.shot_control_calls
    assert isinstance(writes, ShotControlWrites)
    assert writes.devices == ["U_DG645_ShotControl"]


def test_step_request_setpoint_variable_uses_settable(legacy_resolver) -> None:
    session = _FakeSession()
    request = _noscan_request(
        mode="step",
        axes=[{"variable": "jet_z", "positions": {"start": 0, "end": 1, "step": 0.5}}],
    )
    run_scan_request(session, request, legacy_resolver)
    kwargs = session.scan_kwargs
    assert kwargs["positions"] == [0.0, 0.5, 1.0]
    assert kwargs["motor"].kind == "settable"  # legacy entries default setpoint
    assert kwargs["scan_info"]["scan_parameter"] == "U_ESP_JetXYZ:Position.Axis 3"
    assert kwargs["md"]["scan_variable"] == "jet_z"


def test_step_request_motor_kind_and_position_list(modern_resolver) -> None:
    session = _FakeSession()
    request = _noscan_request(
        mode="step",
        save_sets=["NewSet"],
        axes=[{"variable": "jet_z", "positions": {"values": [4.0, 4.5, 6.0]}}],
    )
    run_scan_request(session, request, modern_resolver)
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
    run_scan_request(session, request, modern_resolver)
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
    run_scan_request(session, request, legacy_resolver)

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
    info = kwargs["scan_info"]
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
    # Both movables are disconnected with the scan's devices.
    assert len(session.disconnected) == 5  # 3 detectors + 2 movables


def test_single_axis_request_shape_is_unchanged_by_grid_support(
    legacy_resolver,
) -> None:
    """Regression: one axis still passes a bare motor + flat float positions."""
    session = _FakeSession()
    request = _noscan_request(
        mode="step",
        axes=[{"variable": "jet_z", "positions": {"start": 0, "end": 1, "step": 0.5}}],
    )
    run_scan_request(session, request, legacy_resolver)
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
    run_scan_request(session, request, legacy_resolver)

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
    # and the factory rides the scan's device cleanup.
    (factory,) = session.action_factories
    assert set(factory.settables) == {
        ("U_PLC", "DO.Ch2"),
        ("U_PLC", "DO.Ch3"),
        ("U_PLC", "DO.Ch4"),
    }
    assert factory in session.disconnected


def test_request_without_actions_passes_no_hooks(legacy_resolver) -> None:
    session = _FakeSession()
    run_scan_request(session, _noscan_request(), legacy_resolver)
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
        run_scan_request(session, request, legacy_resolver)
    assert session.devices == []  # failed before any hardware was touched


def test_pseudo_variable_raises_not_implemented(modern_resolver) -> None:
    request = _noscan_request(
        mode="step",
        save_sets=["NewSet"],
        axes=[{"variable": "combo", "positions": {"start": 0, "end": 1, "step": 1}}],
    )
    with pytest.raises(NotImplementedError, match="pseudo"):
        run_scan_request(_FakeSession(), request, modern_resolver)


def test_noscan_without_save_set_is_rejected(legacy_resolver) -> None:
    with pytest.raises(GeecsConfigurationError, match="save set"):
        run_scan_request(_FakeSession(), _noscan_request(save_sets=[]), legacy_resolver)


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


def test_optimize_without_injected_callables_is_a_documented_gap(
    legacy_resolver,
) -> None:
    with pytest.raises(NotImplementedError, match="objective"):
        run_scan_request(_FakeSession(), _optimize_request(), legacy_resolver)


def test_optimize_binder_claims_binds_and_supplies_callables(
    legacy_resolver, monkeypatch
) -> None:
    """The optimization_binder path: the runner claims the scan first (the
    binder's stack needs the real ScanTag), calls the binder with the
    connected movables + detectors, and threads the returned callables and
    the pre-claimed number/folder into session.optimize."""
    import geecs_bluesky.scan_request_runner as runner_module

    tag = SimpleNamespace(number=7)
    claims: list = []
    monkeypatch.setattr(
        runner_module,
        "claim_scan",
        lambda experiment: claims.append(experiment)
        or (tag, "/nonexistent/scans/Scan007"),
    )
    session = _FakeSession()
    objective, suggester = object(), object()
    bind_kwargs: dict = {}

    def binder(*, devices, scan_tag, scan_folder=None):
        bind_kwargs.update(
            devices=list(devices), scan_tag=scan_tag, scan_folder=scan_folder
        )
        return objective, suggester

    uid = run_scan_request(
        session, _optimize_request(), legacy_resolver, optimization_binder=binder
    )
    assert uid == "uid-opt"
    assert claims == [""]  # fake session exposes no experiment name
    assert bind_kwargs["scan_tag"] is tag
    assert bind_kwargs["scan_folder"] == "/nonexistent/scans/Scan007"
    # Movables (2 VOCS variables) + the save set's detectors (3 devices).
    assert len(bind_kwargs["devices"]) == 5
    kwargs = session.optimize_kwargs
    assert kwargs["objective"] is objective
    assert kwargs["suggester"] is suggester
    assert kwargs["scan_number"] == 7
    assert kwargs["scan_folder"] == "/nonexistent/scans/Scan007"


def test_optimize_binder_ignored_when_callables_given(
    legacy_resolver, monkeypatch
) -> None:
    """Ready-made objective/suggester win: no claim, binder never called."""
    import geecs_bluesky.scan_request_runner as runner_module

    def _no_claim(experiment):
        raise AssertionError("claim_scan must not be called")

    monkeypatch.setattr(runner_module, "claim_scan", _no_claim)
    session = _FakeSession()
    objective, suggester = object(), object()

    def binder(**_kwargs):
        raise AssertionError("binder must not be called")

    run_scan_request(
        session,
        _optimize_request(),
        legacy_resolver,
        objective=objective,
        suggester=suggester,
        optimization_binder=binder,
    )
    kwargs = session.optimize_kwargs
    assert kwargs["objective"] is objective
    assert kwargs["scan_number"] is None


def test_optimize_maps_onto_session_optimize(legacy_resolver) -> None:
    session = _FakeSession()

    def objective(bin_data) -> float:
        return 1.0

    suggester = object()
    uid = run_scan_request(
        session,
        _optimize_request(),
        legacy_resolver,
        objective=objective,
        suggester=suggester,
    )
    assert uid == "uid-opt"
    kwargs = session.optimize_kwargs
    # catalog name resolved to its target; Device:Variable passes through
    assert set(kwargs["variables"]) == {"jet_z", "U_S1H:Current"}
    assert kwargs["variables"]["jet_z"].kind == "settable"
    assert kwargs["max_iterations"] == 7
    assert kwargs["shots_per_iteration"] == 5
    assert kwargs["on_finish"] == "best"
    assert kwargs["mode"] == "free_run"
    assert kwargs["objective"] is objective
    assert kwargs["suggester"] is suggester
    # save-set detectors were built and passed along
    assert [d._geecs_device_name for d in kwargs["detectors"]] == [
        "U_Cam",
        "U_Cam2",
        "U_Slow",
    ]


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


def test_multi_device_span_via_variant_adapts() -> None:
    """A variant dragging in a second device lands in the writes."""
    profile = TriggerProfile(
        name="variant-spans",
        states={
            "SCAN": [
                {"device": "U_DG645", "variable": "Trigger.Source", "value": "Ext"},
            ],
        },
        variants={
            "jet_on": {
                "states": {
                    "SCAN": [
                        {"device": "U_Jet", "variable": "Pressure", "value": "5"},
                    ],
                }
            }
        },
    )
    assert trigger_writes_from_profile(profile).devices == ["U_DG645"]
    overlaid = trigger_writes_from_profile(profile, "jet_on")
    assert overlaid.writes_for_state("SCAN") == [
        ("U_DG645", "Trigger.Source", "Ext"),
        ("U_Jet", "Pressure", "5"),
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

    run_scan_request(session, _noscan_request(trigger_profile="spans"), _Resolver())
    (writes,) = session.shot_control_calls
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
    assert updated.trigger_profile == "HTU-Normal"
    assert updated.actions.setup == ["prep"]
    assert updated.actions.closeout == ["shutdown"]
    assert applied == {
        "trigger_profile": "HTU-Normal",
        "actions.setup": ["prep"],
        "actions.closeout": ["shutdown"],
    }
    # The original request is never mutated.
    assert request.trigger_profile is None
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
    run_scan_request(session, _noscan_request(), legacy_resolver)

    (writes,) = session.shot_control_calls
    assert isinstance(writes, ShotControlWrites)
    assert writes.devices == ["U_DG645_ShotControl"]
    assert session.scan_kwargs["md"]["applied_defaults"] == {
        "trigger_profile": "HTU-Normal"
    }


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
    run_scan_request(session, request, legacy_resolver)

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
    run_scan_request(session, request, legacy_resolver)

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
    run_scan_request(session, request, legacy_resolver)

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

    uid = run_scan_request(session, request, legacy_resolver)

    assert uid == "uid-scan"
    # Multi-device trigger attached as ordered writes.
    (writes,) = session.shot_control_calls
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
    assert kwargs["md"]["applied_defaults"] == {
        "actions.setup": ["default_prep"],
        "actions.closeout": ["default_cleanup"],
    }
    # Cleanup: detectors + 2 movables + the action signal factory.
    (factory,) = session.action_factories
    assert factory in session.disconnected


def test_optimize_skips_actions_and_records_them(legacy_resolver, caplog) -> None:
    """Optimize runs; its action plans are skipped, logged, and recorded.

    Optimize mode has no action hooks yet, but refusing would block every
    optimization the moment an experiment defines default bracket actions.
    So the run proceeds with the actions skipped — never silently: a WARNING
    is logged and the skip lands in run metadata.
    """
    session = _FakeSession()
    request = _optimize_request(
        actions={"setup": ["scan_prep"], "closeout": ["cam_park"]}
    )

    def objective(bin_data) -> float:
        return 1.0

    with caplog.at_level(logging.WARNING):
        uid = run_scan_request(
            session, request, legacy_resolver, objective=objective, suggester=object()
        )
    assert uid == "uid-opt"
    skipped = session.optimize_kwargs["md"]["skipped_action_plans"]
    assert skipped["setup"] == ["scan_prep"]
    assert skipped["closeout"] == ["cam_park"]
    assert "scan_prep" in caplog.text


def test_optimize_skips_entry_rituals_and_records_them(legacy_resolver) -> None:
    """Save-set entry rituals are skipped and recorded, not refused."""
    session = _FakeSession()
    request = _optimize_request(save_sets=["RitualSet"])

    def objective(bin_data) -> float:
        return 1.0

    uid = run_scan_request(
        session, request, legacy_resolver, objective=objective, suggester=object()
    )
    assert uid == "uid-opt"
    assert (
        "cam_ritual"
        in session.optimize_kwargs["md"]["skipped_action_plans"]["save_set_rituals"]
    )


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
    """Fake session exposing experiment + soft telemetry factory."""

    experiment = "TestExp"

    def __init__(self) -> None:
        super().__init__()
        self.telemetry_calls: list = []
        self.dead_devices: set = set()

    def telemetry(self, device, variables, *, name=None):
        self.telemetry_calls.append((device, list(variables)))
        if device in self.dead_devices:
            return None  # unreachable at scan start → dropped
        return self._make(f"telemetry:{device}", "telemetry")


def _install_policy(monkeypatch, policy) -> None:
    """Force run_scan_request to use *policy* instead of a real GeecsDb."""
    import geecs_bluesky.scan_request_runner as runner

    monkeypatch.setattr(runner, "make_scalar_policy", lambda session: policy)


def _db_noscan_request(**overrides):
    base = dict(
        mode="noscan",
        shots_per_step=2,
        acquisition="strict",
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
        run_scan_request(session, _db_noscan_request(save_sets=["X"]), legacy_resolver)

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


def test_telemetry_selects_non_saveset_devices(monkeypatch, legacy_resolver) -> None:
    policy = _M3cPolicy(
        subscribed={
            "U_Cam": ["MaxCounts"],  # in save set (UC_Test) → excluded
            "U_Press": ["Pressure"],  # not in save set → telemetry
        }
    )
    _install_policy(monkeypatch, policy)
    session = _M3cSession()
    run_scan_request(session, _db_noscan_request(), legacy_resolver)
    assert session.telemetry_calls == [("U_Press", ["Pressure"])]
    # Telemetry device appended to the read set, never the reference.
    assert ("telemetry:U_Press", "telemetry") in session.devices
    assert session.scan_kwargs["md"]["background_telemetry"] == {
        "U_Press": ["Pressure"]
    }


def test_telemetry_dead_device_dropped_not_raised(
    monkeypatch, legacy_resolver, caplog
) -> None:
    policy = _M3cPolicy(subscribed={"U_Press": ["Pressure"]})
    _install_policy(monkeypatch, policy)
    session = _M3cSession()
    session.dead_devices = {"U_Press"}
    # Must not raise even though the telemetry device is unreachable.
    run_scan_request(session, _db_noscan_request(), legacy_resolver)
    # Attempted, returned None → not in the read set.
    assert session.telemetry_calls == [("U_Press", ["Pressure"])]
    assert not any(d[0].startswith("telemetry:") for d in session.devices)


def test_background_telemetry_off_skips_telemetry(monkeypatch, legacy_resolver) -> None:
    policy = _M3cPolicy(subscribed={"U_Press": ["Pressure"]})
    _install_policy(monkeypatch, policy)
    session = _M3cSession()
    run_scan_request(
        session,
        _db_noscan_request(background_telemetry=False),
        legacy_resolver,
    )
    assert session.telemetry_calls == []
    assert "background_telemetry" not in session.scan_kwargs["md"]


def test_request_telemetry_flag_overrides_experiment_default(
    monkeypatch, legacy_resolver
) -> None:
    policy = _M3cPolicy(subscribed={"U_Press": ["Pressure"]})
    _install_policy(monkeypatch, policy)
    # Experiment default off, request explicitly on → telemetry runs.
    monkeypatch.setattr(
        legacy_resolver,
        "resolve_experiment_defaults",
        lambda: ExperimentDefaults.model_validate(
            {"schema_version": 1, "background_telemetry": False}
        ),
    )
    session = _M3cSession()
    run_scan_request(
        session,
        _db_noscan_request(background_telemetry=True),
        legacy_resolver,
    )
    assert session.telemetry_calls == [("U_Press", ["Pressure"])]


def test_no_provider_leaves_m3b_behavior_unchanged(legacy_resolver) -> None:
    # A session with no experiment attribute → no policy → no DB writes, no
    # telemetry, explicit-only scalars (the M3b path, still green).
    session = _FakeSession()
    run_scan_request(session, _db_noscan_request(), legacy_resolver)
    assert "db_scan_writes" not in session.scan_kwargs["md"]
    assert "background_telemetry" not in session.scan_kwargs["md"]


def test_telemetry_metadata_excludes_dropped_devices(
    monkeypatch, legacy_resolver
) -> None:
    """md background_telemetry records only devices that connected (review P2).

    A device dropped as unreachable at scan start contributes no columns, so
    the start-doc must not advertise it (EVENT_SCHEMA.md contract).
    """
    policy = _M3cPolicy(
        subscribed={
            "U_Press": ["Pressure"],  # live → recorded
            "U_Dead": ["X"],  # unreachable → dropped, must be absent from md
        }
    )
    _install_policy(monkeypatch, policy)
    session = _M3cSession()
    session.dead_devices = {"U_Dead"}
    run_scan_request(session, _db_noscan_request(), legacy_resolver)
    recorded = session.scan_kwargs["md"]["background_telemetry"]
    assert recorded == {"U_Press": ["Pressure"]}
    assert "U_Dead" not in recorded


def test_optimize_warns_on_reserved_boundary_fields(
    monkeypatch, legacy_resolver, caplog
) -> None:
    """Optimize mode also warns once on reserved at_scan_start/at_scan_end (P3).

    The reserved-field warning must fire in every mode that resolves a save set,
    not only scan/noscan — optimize ignores the set-side too.
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
            )
        ],
    )
    monkeypatch.setattr(legacy_resolver, "resolve_save_set", lambda name: save_set)

    def objective(bin_data) -> float:
        return 1.0

    with caplog.at_level(logging.WARNING):
        run_scan_request(
            session,
            _optimize_request(save_sets=["X"]),
            legacy_resolver,
            objective=objective,
            suggester=object(),
        )
    reserved = [
        r
        for r in caplog.records
        if "reserved DB scan start/end fields" in r.getMessage()
    ]
    assert len(reserved) == 1
    assert "U_DG645_ShotControl" in reserved[0].getMessage()


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
    run_scan_request(session, request, legacy_resolver)
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


def test_telemetry_excludes_devices_from_all_named_sets(
    monkeypatch, legacy_resolver
) -> None:
    # A device in ANY named set must be excluded from Tier-2 telemetry: pass
    # the merged save set (all devices across all sets) to the selector.
    policy = _M3cPolicy(
        subscribed={
            "U_Cam": ["MaxCounts"],  # in UC_Test → excluded
            "U_Aux": ["Aux1"],  # in UC_Aux → excluded
            "U_Press": ["Pressure"],  # in neither → telemetry
        }
    )
    _install_policy(monkeypatch, policy)
    session = _M3cSession()
    run_scan_request(
        session,
        _db_noscan_request(save_sets=["UC_Test", "UC_Aux"]),
        legacy_resolver,
    )
    assert session.telemetry_calls == [("U_Press", ["Pressure"])]
    assert session.scan_kwargs["md"]["background_telemetry"] == {
        "U_Press": ["Pressure"]
    }


# ---------------------------------------------------------------------------
# Runner hooks (the GUI-bridge seams): preflight + on_scan_start
# ---------------------------------------------------------------------------


def test_preflight_receives_assembled_detectors_and_strict(legacy_resolver) -> None:
    """The hook sees the fully assembled detector list and the strict flag,
    and its (possibly reduced) return is what session.scan runs with."""
    session = _FakeSession()
    seen: dict = {}

    def preflight(detectors: list, strict: bool):
        seen["devices"] = [d._geecs_device_name for d in detectors]
        seen["strict"] = strict
        return detectors[:2]  # drop the snapshot device

    uid = run_scan_request(
        session,
        _noscan_request(acquisition="strict"),
        legacy_resolver,
        preflight=preflight,
    )
    assert uid == "uid-scan"
    assert seen["strict"] is True
    assert seen["devices"] == ["U_Cam", "U_Cam2", "U_Slow"]
    assert [d._geecs_device_name for d in session.scan_kwargs["detectors"]] == [
        "U_Cam",
        "U_Cam2",
    ]


def test_preflight_none_aborts_without_scanning(legacy_resolver) -> None:
    """A None return aborts pre-claim; created devices are still disconnected."""
    session = _FakeSession()
    result = run_scan_request(
        session, _noscan_request(), legacy_resolver, preflight=lambda d, s: None
    )
    assert result is None
    assert session.scan_kwargs is None
    assert len(session.disconnected) == 3  # the runner's finally cleaned up


def test_preflight_strict_flag_false_for_free_run(legacy_resolver) -> None:
    session = _FakeSession()
    flags: list[bool] = []
    run_scan_request(
        session,
        _noscan_request(acquisition="free_run"),
        legacy_resolver,
        preflight=lambda detectors, strict: flags.append(strict) or detectors,
    )
    assert flags == [False]


def test_on_scan_start_totals_for_noscan(legacy_resolver) -> None:
    session = _FakeSession()
    calls: list[tuple[int, int]] = []
    run_scan_request(
        session,
        _noscan_request(),  # shots_per_step=3
        legacy_resolver,
        on_scan_start=lambda steps, shots: calls.append((steps, shots)),
    )
    assert calls == [(1, 3)]


def test_on_scan_start_totals_for_grid(legacy_resolver) -> None:
    """A 2×3 grid is 6 steps; totals reflect the flat outer-product length."""
    session = _FakeSession()
    calls: list[tuple[int, int]] = []
    request = _noscan_request(
        mode="step",
        axes=[
            {"variable": "jet_z", "positions": {"start": 0, "end": 1, "step": 1}},
            {"variable": "jet_x", "positions": {"values": [4.0, 5.0, 6.0]}},
        ],
    )
    run_scan_request(
        session,
        request,
        legacy_resolver,
        on_scan_start=lambda steps, shots: calls.append((steps, shots)),
    )
    assert calls == [(6, 18)]  # shots_per_step=3


def test_optimize_binder_operator_abort_notes_folder_calmly(
    legacy_resolver, monkeypatch, caplog
) -> None:
    """A quiet aborted optimize return draws the calm WARNING, never the ERROR."""
    import geecs_bluesky.scan_request_runner as runner_module

    tag = SimpleNamespace(number=7)
    monkeypatch.setattr(
        runner_module,
        "claim_scan",
        lambda experiment: (tag, "/nonexistent/scans/Scan007"),
    )
    session = _FakeSession()

    def optimize_aborted_by_operator(**kwargs):
        session.optimize_kwargs = kwargs
        session.last_run_aborted = True  # session.optimize's quiet abort return
        return "uid-opt", []

    session.optimize = optimize_aborted_by_operator

    with caplog.at_level(logging.INFO):
        uid = run_scan_request(
            session,
            _optimize_request(),
            legacy_resolver,
            optimization_binder=lambda **_kw: (object(), object()),
        )

    assert uid == "uid-opt"
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == [], (
        "an operator-requested abort must not log ERROR records"
    )
    notes = [
        r
        for r in caplog.records
        if "aborted by operator" in r.getMessage()
        and "Optimization scan" in r.getMessage()
    ]
    assert [r.levelno for r in notes] == [logging.WARNING]
