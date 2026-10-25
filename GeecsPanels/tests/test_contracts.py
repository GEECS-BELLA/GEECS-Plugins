"""The bridge's wire contracts, and the pin holding the family files to the panels' tables."""

from __future__ import annotations

import json

import pytest
from geecs_schemas import FamilyFile, PanelDevice, load_families
from pydantic import ValidationError

from geecs_panels import camera, motor, power, scope
from geecs_panels.contracts import (
    CONTRACTS,
    DeviceFrame,
    History,
    PutRequest,
    PutResult,
    RoleReading,
    SubscriptionRequest,
    docstring_examples,
    json_schemas,
    main,
)

# ---- the contracts -------------------------------------------------------------


def test_every_contract_has_a_json_schema_carrying_its_examples() -> None:
    schemas = json_schemas()
    assert set(schemas) == set(CONTRACTS)
    for name, model in CONTRACTS.items():
        assert schemas[name]["title"] == model.__name__
    assert len(schemas["device_frame"]["examples"]) == 4


@pytest.mark.parametrize("name", list(CONTRACTS))
def test_the_docstring_examples_validate(name: str) -> None:
    model = CONTRACTS[name]
    examples = docstring_examples(model)
    assert examples, f"{model.__name__} has no Examples section"
    for example in examples:
        parsed = model.model_validate(example)
        assert parsed.model_dump(exclude_unset=True) == example  # nothing rewritten
        # The wire is the default serialization: what the bridge sends.
        assert model.model_validate_json(parsed.model_dump_json()) == parsed


def test_the_module_prints_every_schema(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(json.dumps(json_schemas()))


def test_nan_travels_as_null_everywhere() -> None:
    reading = RoleReading.model_validate(
        {"value": float("nan"), "freshness": "stale", "fields": {"DRVH": float("inf")}}
    )
    assert reading.value is None and reading.fields == {"DRVH": None}
    history = History.model_validate({"keys": ["cx"], "points": [[1.0, float("nan")]]})
    assert history.points == [[1.0, None]]
    frame = DeviceFrame.model_validate(
        {"device": "d", "family": "f", "seq": 0, "server_time": 0.0, "full": True,
         "connected": True, "roles": {"x": reading.model_dump()}}
    )  # fmt: skip
    json.loads(frame.model_dump_json())  # strict JSON: no NaN token


def test_a_role_travels_whole_in_every_frame() -> None:
    """A delta's role is complete on the wire: nothing in it is 'unchanged' to merge."""
    frame = DeviceFrame.model_validate(
        {"device": "d", "family": "f", "seq": 4, "server_time": 10.0, "full": False,
         "connected": True, "roles": {"x": {"freshness": "stale", "value": 1.5}}}
    )  # fmt: skip
    wire = json.loads(frame.model_dump_json())
    role = wire["roles"]["x"]
    assert set(role) == {"value", "severity", "advanced_at", "freshness", "fields"}
    assert role["value"] == 1.5  # a freshness change never reads as a disconnect


def test_a_delta_frame_carries_only_what_changed() -> None:
    frame = DeviceFrame.model_validate(
        {"device": "d", "family": "f", "seq": 3, "server_time": 0.0, "full": False,
         "connected": True}
    )  # fmt: skip
    assert frame.roles == {} and frame.state is None and frame.history is None


def test_a_put_names_its_operator_and_nothing_else() -> None:
    base = {"device": "d", "role": "r", "value": 1, "operator": "A. Person"}
    assert PutRequest.model_validate(base).confirm is False
    with pytest.raises(ValidationError):
        PutRequest.model_validate({**base, "operator": ""})
    with pytest.raises(ValidationError):
        PutRequest.model_validate({**base, "suffix": "cam1:Acquire"})
    with pytest.raises(ValidationError):
        PutRequest.model_validate({**base, "value": None})


def test_a_put_result_is_journaled() -> None:
    with pytest.raises(ValidationError, match="journal_id"):
        PutResult.model_validate({"outcome": "accepted"})
    with pytest.raises(ValidationError):
        PutResult.model_validate({"outcome": "ok", "journal_id": 1})


def test_a_subscription_names_each_pv_once() -> None:
    with pytest.raises(ValidationError, match="named twice"):
        SubscriptionRequest.model_validate({"pvs": ["a:b", "a:b"]})
    with pytest.raises(ValidationError, match="not PV names"):
        SubscriptionRequest.model_validate({"pvs": ["a b"]})


# ---- the pin: family files == the panels' tables --------------------------------
# GEECS-Schemas cannot import this package, so the pin lives here. When a
# table here changes, its family file changes with it (or the table goes).


def _monitored(family: FamilyFile) -> set[str]:
    return {r.suffix for r in family.roles if r.read_mode == "monitor"}


def _strings(family: FamilyFile) -> set[str]:
    return {
        r.suffix
        for r in family.roles
        if r.read_mode == "monitor" and (r.type == "enum" or r.text)
    }


def _writes(family: FamilyFile) -> set[str]:
    return {s for r in family.roles if (s := r.write_suffix) is not None}


def _settled(family: FamilyFile) -> dict[str, str]:
    return {
        str(r.write_suffix): family.role(r.settled_by).suffix
        for r in family.roles
        if r.settled_by
    }


def test_the_motor_family_is_the_motor_panels_table() -> None:
    family = load_families()["motor_record"]
    assert family.per_axis
    assert _monitored(family) == set(motor.AXIS_FIELDS.values())
    strings = {motor.AXIS_FIELDS[k] for k in motor.STRING_KEYS}
    assert _strings(family) == strings
    assert _writes(family) == motor.WRITABLE
    assert set(_settled(family)) == motor.NO_WAIT
    actions = {
        name: frozenset(family.role(r).suffix for r in action.roles)
        for name, action in family.actions.items()
    }
    assert actions == motor.ACTIONS
    motion = {a for a, act in family.actions.items() if "stop" in act.requires}
    assert motion == motor.MOTION_ACTIONS


def test_the_camera_family_is_the_camera_panels_tables() -> None:
    family = load_families()["ad_camera_chain"]
    assert _monitored(family) == set(camera.monitored_suffixes())
    assert _strings(family) == camera.string_suffixes()
    assert _writes(family) == set(camera.WRITABLE)
    assert _settled(family) == camera.SETTLED_BY_READBACK
    arrays = {r.suffix for r in family.roles if r.type == "array"}
    assert arrays == set(camera.ARRAYS.values())
    assert {r.suffix for r in family.roles if r.type == "image"} == {
        camera.JPG_URL_SUFFIX
    }
    locked = {s for r in family.roles if r.locked and (s := r.write_suffix)}
    assert locked == {
        w for w in camera.WRITABLE if w.startswith(camera.LOCKED_PREFIXES)
    }
    readings = {r.suffix: r for r in family.roles}
    for reading in camera.READINGS:
        role = readings[reading.suffix]
        fresh = family.role(role.fresh_by).suffix if role.fresh_by else None
        flag = family.role(role.computed_by).suffix if role.computed_by else None
        plugin = camera.counter_suffix(reading.plugin) if reading.plugin else None
        assert (fresh, flag) == (plugin, reading.computed_by), reading.key


def test_the_scope_family_is_the_scope_panels_tables() -> None:
    family = load_families()["scope_contract"]
    assert _monitored(family) == set(scope.monitored())
    strings = {s for s, (_key, as_string) in scope.monitored().items() if as_string}
    assert _strings(family) == strings
    char_arrays = {r.suffix for r in family.roles if r.text == "char_array"}
    assert char_arrays == set(scope.CHAR_STRING_SUFFIXES)
    assert _writes(family) == set(scope.WRITABLE)
    assert _settled(family) == scope.SETTLED_BY_READBACK
    on_request = {r.suffix for r in family.roles if r.read_mode == "on_request"}
    captured = {scope.ARRAY_DATA, scope.TIME_AXIS, *scope.ARRAY_FIELDS.values()}
    assert on_request == captured - {scope.ARRAY_UID}


def _supply_table(
    family: FamilyFile, device: PanelDevice
) -> dict[str, tuple[str, bool]]:
    """The family's monitored table for one device, keyed the way ``power.monitored`` keys it."""
    table: dict[str, tuple[str, bool]] = {}
    for role in device.resolve(family):
        if role.read_mode != "monitor":
            continue
        key = f"bit.{role.name}" if role.fault else role.name
        table[key] = (role.suffix, role.type == "enum" or bool(role.text))
        for field in role.fields:
            table[f"{key}.{field.lower()}"] = (
                role.suffix + "." + field,
                field in _STRING_FIELDS,
            )
    return table


#: ``power.STRING_FIELDS`` by record field.
_STRING_FIELDS = {f.upper() for f in power.STRING_FIELDS}


@pytest.mark.parametrize("template", ["fastps", "easydriver"])
def test_a_supply_family_is_the_supply_panels_table(template: str) -> None:
    family = load_families()[template]
    device = PanelDevice(family=template)
    roles = {r.name: r.suffix for r in family.roles if not r.fault}
    bits = {r.name: r.suffix for r in family.roles if r.fault}
    assert _supply_table(family, device) == power.monitored(roles, bits)
    power.check_roles(
        roles, list(family.actions)
    )  # every required role, only known ones
    actions = {a: act.roles for a, act in family.actions.items()}
    assert actions == {a: [r] for a, r in power.ACTIONS.items() if r in roles}


def test_the_fpstest1_bench_entry_is_the_fastps_family_as_it_stands() -> None:
    """The README's bench supply needs no overrides: the family carries its PVs."""
    family = load_families()["fastps"]
    bench_roles = {
        "current_sp": "current_ramp:SP", "current": "current", "voltage": "voltage",
        "output_sp": "enable_output:SP", "output": "enable_output", "reset": "reset:SP",
        "loop_mode": "loop_mode",
    }  # fmt: skip
    bench_bits = {
        "fault": "fault",
        "interlock_1": "interlock_1",
        "interlock_2": "interlock_2",
    }
    assert {r.name: r.suffix for r in family.roles if not r.fault} == bench_roles
    assert {r.name: r.suffix for r in family.roles if r.fault} == bench_bits
