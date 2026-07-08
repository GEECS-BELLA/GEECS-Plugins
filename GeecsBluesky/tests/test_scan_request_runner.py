"""Tests for scan_request_runner: resolver, adapters, and run_scan_request.

Covers the configs-repo resolver (new-schema YAML loads directly, legacy YAML
converts — the whole existing corpus is usable immediately), the SaveSet →
devices_config derivation rules, the TriggerProfile → ShotControlConfig
adapter, and the request execution mapping onto a fake GeecsSession —
including every documented v1 gap (multi-axis, actions, pseudo variables,
optimize without injected callables) refusing loudly.
"""

from __future__ import annotations

import pytest

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlConfig
from geecs_bluesky.scan_request_runner import (
    MULTI_AXIS_MESSAGE,
    ConfigResolver,
    ConfigsRepoResolver,
    apply_experiment_defaults,
    collect_save_set_action_names,
    resolve_save_set_checked,
    run_scan_request,
    save_set_to_devices_config,
    shot_control_config_from_trigger_profile,
)
from geecs_schemas import (
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


class _FakeSession:
    def __init__(self) -> None:
        self.devices: list[tuple[str, str]] = []  # (device, factory)
        self.shot_control_calls: list = []
        self.scan_kwargs: dict | None = None
        self.optimize_kwargs: dict | None = None
        self.disconnected: list = []

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
"""

NEW_SCAN_VARIABLES = """\
schema_version: 1
variables:
  jet_z: {target: "U_ESP_JetXYZ:Position.Axis 3", kind: motor}
  hexapod_y: {target: "U_Hexapod:ypos"}
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
# TriggerProfile → ShotControlConfig adapter
# ---------------------------------------------------------------------------


def test_trigger_adapter_preserves_state_semantics(legacy_resolver) -> None:
    """values_for_state / defines_state agree between the two models."""
    profile = legacy_resolver.resolve_trigger_profile("HTU-Normal")
    config = shot_control_config_from_trigger_profile(profile)
    assert isinstance(config, ShotControlConfig)
    assert config.device == profile.devices[0]
    for state in ("OFF", "SCAN", "STANDBY", "SINGLESHOT", "ARMED"):
        expected = {w.variable: w.value for w in profile.writes_for(state)}
        assert config.values_for_state(state) == expected, state
        assert config.defines_state(state) == profile.defines_state(state), state


def test_trigger_adapter_applies_variant(modern_resolver) -> None:
    profile = modern_resolver.resolve_trigger_profile("NewProfile")
    config = shot_control_config_from_trigger_profile(profile, "laser_off")
    assert config.values_for_state("SCAN") == {"Trigger.Source": "Internal"}


def test_trigger_adapter_unknown_variant_raises(modern_resolver) -> None:
    profile = modern_resolver.resolve_trigger_profile("NewProfile")
    with pytest.raises(GeecsConfigurationError, match="laser_off"):
        shot_control_config_from_trigger_profile(profile, "nope")


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
        save_set="UC_Test",
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
    (config,) = session.shot_control_calls
    assert isinstance(config, ShotControlConfig)
    assert config.device == "U_DG645_ShotControl"


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
        save_set="NewSet",
        axes=[{"variable": "jet_z", "positions": {"values": [4.0, 4.5, 6.0]}}],
    )
    run_scan_request(session, request, modern_resolver)
    kwargs = session.scan_kwargs
    assert kwargs["motor"].kind == "motor"
    assert kwargs["positions"] == [4.0, 4.5, 6.0]


def test_multi_axis_raises_not_implemented(legacy_resolver) -> None:
    request = _noscan_request(
        mode="step",
        axes=[
            {"variable": "jet_z", "positions": {"start": 0, "end": 1, "step": 0.5}},
            {"variable": "jet_x", "positions": {"start": 0, "end": 1, "step": 0.5}},
        ],
    )
    with pytest.raises(NotImplementedError, match=MULTI_AXIS_MESSAGE):
        run_scan_request(_FakeSession(), request, legacy_resolver)


def test_valid_actions_are_validated_then_refused(legacy_resolver) -> None:
    session = _FakeSession()
    request = _noscan_request(actions={"setup": ["close_shutters"]})
    with pytest.raises(NotImplementedError, match="close_shutters"):
        run_scan_request(session, request, legacy_resolver)
    assert session.devices == []  # refused before any hardware was touched


def test_unknown_action_name_fails_validation_first(legacy_resolver) -> None:
    request = _noscan_request(actions={"closeout": ["not_a_plan"]})
    with pytest.raises(GeecsConfigurationError, match="not_a_plan"):
        run_scan_request(_FakeSession(), request, legacy_resolver)


def test_pseudo_variable_raises_not_implemented(modern_resolver) -> None:
    request = _noscan_request(
        mode="step",
        save_set="NewSet",
        axes=[{"variable": "combo", "positions": {"start": 0, "end": 1, "step": 1}}],
    )
    with pytest.raises(NotImplementedError, match="pseudo"):
        run_scan_request(_FakeSession(), request, modern_resolver)


def test_noscan_without_save_set_is_rejected(legacy_resolver) -> None:
    with pytest.raises(GeecsConfigurationError, match="save_set"):
        run_scan_request(
            _FakeSession(), _noscan_request(save_set=None), legacy_resolver
        )


def _optimize_request(**overrides) -> ScanRequest:
    base = dict(
        mode="optimize",
        shots_per_step=5,
        acquisition="free_run",
        save_set="UC_Test",
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
# Multi-device trigger profiles: single-device fast path / engine-pending raise
# ---------------------------------------------------------------------------


def test_multi_device_shape_single_device_fast_path() -> None:
    """Write lists that all name one device adapt exactly as before."""
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
    config = shot_control_config_from_trigger_profile(profile)
    assert config.device == "U_DG645"
    assert config.values_for_state("SCAN") == {
        "Trigger.Source": "Ext",
        "Amplitude.Ch AB": "4.0",
    }
    assert config.values_for_state("STANDBY") == {"Amplitude.Ch AB": "0.5"}
    assert not config.defines_state("OFF")


def test_multi_device_writes_are_engine_pending() -> None:
    """Writes spanning devices are schema-legal but refused by the engine."""
    profile = TriggerProfile(
        name="spans-devices",
        states={
            "SCAN": [
                {"device": "U_DG645", "variable": "Trigger.Source", "value": "Ext"},
                {"device": "U_PLC", "variable": "DO.Ch9", "value": "on"},
            ],
        },
    )
    assert profile.devices == ["U_DG645", "U_PLC"]  # schema accepts it
    with pytest.raises(
        NotImplementedError,
        match="multi-device trigger profiles land with a later milestone",
    ):
        shot_control_config_from_trigger_profile(profile)


def test_multi_device_span_via_variant_is_also_refused() -> None:
    """A variant that drags in a second device trips the same refusal."""
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
    # The base profile alone is single-device and adapts fine.
    assert shot_control_config_from_trigger_profile(profile).device == "U_DG645"
    with pytest.raises(NotImplementedError, match="multi-device"):
        shot_control_config_from_trigger_profile(profile, "jet_on")


def test_profile_without_any_device_is_rejected() -> None:
    profile = TriggerProfile(name="empty", states={})
    with pytest.raises(GeecsConfigurationError, match="names no trigger device"):
        shot_control_config_from_trigger_profile(profile)


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

    (config,) = session.shot_control_calls
    assert isinstance(config, ShotControlConfig)
    assert config.device == "U_DG645_ShotControl"
    assert session.scan_kwargs["md"]["applied_defaults"] == {
        "trigger_profile": "HTU-Normal"
    }


def test_default_actions_get_the_same_refusal_treatment(
    configs_root, legacy_resolver
) -> None:
    """Defaults-supplied actions are validated, then refused like explicit ones."""
    (configs_root / "LegacyExp" / "experiment_defaults.yaml").write_text(
        "actions:\n  setup: [close_shutters]\n"
    )
    session = _FakeSession()
    with pytest.raises(NotImplementedError, match="close_shutters"):
        run_scan_request(session, _noscan_request(), legacy_resolver)
    assert session.devices == []  # refused before any hardware


# ---------------------------------------------------------------------------
# SaveSet entry-level setup/closeout references (newer SaveSet schema)
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
        return object()


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
    assert collect_save_set_action_names(save_set) == ["prep_cam", "park"]
    # Entries without references contribute nothing.
    plain = SaveSet(name="s", entries=[SaveSetEntry(device="U_A", scalars=["x"])])
    assert collect_save_set_action_names(plain) == []


def _entry_action_save_set(**entry_overrides) -> SaveSet:
    entry = dict(device="U_A", scalars=["x"])
    entry.update(entry_overrides)
    return SaveSet.model_validate({"name": "s", "entries": [entry]})


def test_entry_level_actions_validated_then_refused() -> None:
    resolver = _SaveSetResolver(_entry_action_save_set(setup=["prep_cam"]))
    with pytest.raises(NotImplementedError, match="prep_cam"):
        resolve_save_set_checked(resolver, "s")


def test_entry_level_unknown_action_fails_validation_first() -> None:
    resolver = _SaveSetResolver(_entry_action_save_set(setup=["nope"]))
    with pytest.raises(GeecsConfigurationError, match="nope"):
        resolve_save_set_checked(resolver, "s")


def test_converted_element_actions_resolve_and_are_refused(legacy_resolver) -> None:
    """A legacy element with setup_action converts to entry refs that
    validate against the extracted plans and then get the M3b refusal."""
    with pytest.raises(NotImplementedError, match="UC_WithActions_setup"):
        resolve_save_set_checked(legacy_resolver, "UC_WithActions")


def test_run_refuses_save_set_with_converted_element_actions(
    legacy_resolver,
) -> None:
    session = _FakeSession()
    request = _noscan_request(save_set="UC_WithActions")
    with pytest.raises(NotImplementedError, match="UC_WithActions_setup"):
        run_scan_request(session, request, legacy_resolver)
    assert session.devices == []  # refused before any hardware
