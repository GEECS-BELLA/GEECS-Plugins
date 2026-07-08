"""Tests for building a DeviceSpec from GEECS DB metadata (no network).

Exercises the pure ``DeviceSpec.from_db_metadata`` core using rows shaped like
``GeecsDb.get_device_variables`` returns for the real U_S1H device.
"""

from __future__ import annotations

import pytest

from geecs_ca_gateway.alarms import AlarmLimits
from geecs_ca_gateway.config import DeviceSpec, GatewayConfig
from geecs_ca_gateway.db.geecs_db import GeecsDb
from geecs_ca_gateway.gateway import GeecsCaGateway

# Mirrors GeecsDb.get_device_variables("U_S1H") observed against real hardware.
U_S1H_META = [
    {"name": "Current", "units": "A", "min": -5.0, "max": 5.0, "settable": True},
    {"name": "Voltage", "units": "V", "min": -10.0, "max": 10.0, "settable": False},
    {"name": "SlewRate", "units": "", "min": 0.0, "max": 12.0, "settable": False},
    {"name": "IPAddress", "units": "", "min": None, "max": None, "settable": False},
]


def test_from_db_metadata_maps_units_limits_settable() -> None:
    """Units, min/max, and the settable flag map onto VariableSpec fields."""
    spec = DeviceSpec.from_db_metadata("U_S1H", "192.168.7.150", 65050, U_S1H_META)
    by_name = {v.geecs_var: v for v in spec.variables}

    assert by_name["Current"].settable is True
    assert by_name["Current"].egu == "A"
    assert (by_name["Current"].lo, by_name["Current"].hi) == (-5.0, 5.0)

    assert by_name["Voltage"].settable is False
    assert by_name["Voltage"].egu == "V"

    # units default to empty; missing limits stay None
    assert by_name["IPAddress"].lo is None and by_name["IPAddress"].hi is None


def test_alarm_limits_validate_order_and_presence() -> None:
    """Curated alarm limits must be explicit and monotonic."""
    limits = AlarmLimits(low=1.0, high=5.0)
    assert limits.low == 1.0
    with pytest.raises(ValueError, match="at least one"):
        AlarmLimits()
    with pytest.raises(ValueError, match="increase"):
        AlarmLimits(low=5.0, high=1.0)


def test_from_geecs_experiment_attaches_numeric_alarm_limits(monkeypatch) -> None:
    """DB alarm rows attach only to served numeric readback variables."""
    monkeypatch.setattr(
        GeecsDb,
        "get_experiment_devices",
        classmethod(lambda cls, experiment, enabled_only=True: {"U_A": ("h", 1)}),
    )
    monkeypatch.setattr(
        GeecsDb,
        "get_experiment_device_variables",
        classmethod(
            lambda cls, experiment, enabled_only=True: {
                "U_A": [
                    {
                        "name": "Current",
                        "units": "A",
                        "min": -5.0,
                        "max": 5.0,
                        "settable": True,
                        "variabletype": "numeric",
                    }
                ]
            }
        ),
    )
    monkeypatch.setattr(
        GeecsDb,
        "get_subscribed_variables",
        classmethod(lambda cls, experiment, enabled_only=True: {"U_A": ["Current"]}),
    )
    monkeypatch.setattr(
        GeecsDb,
        "get_ca_alarm_limits",
        classmethod(
            lambda cls, experiment: {("U_A", "Current"): AlarmLimits(high=4.0)}
        ),
    )

    cfg = GatewayConfig.from_geecs_experiment("Undulator")
    assert cfg.devices[0].variables[0].alarm_limits == AlarmLimits(high=4.0)


def test_include_filters_variables() -> None:
    """``include`` restricts which variables are exposed."""
    spec = DeviceSpec.from_db_metadata(
        "U_S1H", "h", 1, U_S1H_META, include=["Current", "Voltage"]
    )
    assert {v.geecs_var for v in spec.variables} == {"Current", "Voltage"}


def test_include_settable_exposes_control_surface() -> None:
    """Settable variables survive the include filter with include_settable.

    The include list is the monitoring subset (get='yes'); settable variables
    are the control surface (camera save/localsavingpath, magnet setpoints)
    and CA clients need their :SP PVs regardless of the get-list.
    """
    # Voltage-only get-list; Current is settable but not monitored.
    spec = DeviceSpec.from_db_metadata(
        "U_S1H", "h", 1, U_S1H_META, include=["Voltage"], include_settable=True
    )
    assert {v.geecs_var for v in spec.variables} == {"Voltage", "Current"}
    # Without the flag, the include list is strict (existing behavior).
    spec = DeviceSpec.from_db_metadata("U_S1H", "h", 1, U_S1H_META, include=["Voltage"])
    assert {v.geecs_var for v in spec.variables} == {"Voltage"}


def test_dtypes_override() -> None:
    """Per-variable dtype overrides win over the float default."""
    spec = DeviceSpec.from_db_metadata(
        "U_S1H", "h", 1, U_S1H_META, dtypes={"IPAddress": "string"}
    )
    by_name = {v.geecs_var: v for v in spec.variables}
    assert by_name["IPAddress"].dtype == "string"
    assert by_name["Current"].dtype == "float"


def test_pvdb_built_from_db_spec_has_limits() -> None:
    """A gateway built from the DB spec carries CA control limits on channels."""
    spec = DeviceSpec.from_db_metadata(
        "U_S1H", "h", 1, U_S1H_META, include=["Current", "Voltage"]
    )
    # __init__ only builds the pvdb — no network until connect() is called.
    gw = GeecsCaGateway(GatewayConfig(devices=[spec]))

    current = gw.pvdb["U_S1H:Current"]
    # display (informational) limits, not enforced control limits
    assert current.upper_disp_limit == 5.0
    assert current.lower_disp_limit == -5.0
    assert current.units == "A"
    assert "U_S1H:Current:SP" in gw.pvdb  # settable -> setpoint exists


def test_from_db_metadata_dedupes_duplicate_variables() -> None:
    """The GEECS DB can list a variable twice (real case: U_GhostFilters)."""
    meta = [
        {
            "name": "Transmission.Channel11.Pos1",
            "units": "",
            "min": None,
            "max": None,
            "settable": True,
        },
        {
            "name": "Transmission.Channel11.Pos1",
            "units": "",
            "min": None,
            "max": None,
            "settable": True,
        },
    ]
    spec = DeviceSpec.from_db_metadata(
        "U_GhostFilters", "h", 1, meta, experiment="Undulator"
    )
    assert len(spec.variables) == 1  # deduped
    # and building the gateway does not raise a spurious collision
    gw = GeecsCaGateway(GatewayConfig(devices=[spec]))
    assert "Undulator:U_GhostFilters:Transmission_Channel11_Pos1" in gw.pvdb


def test_from_db_metadata_maps_variable_types() -> None:
    """variabletype maps to dtype; image/1darray are skipped; enum gets choices."""
    meta = [
        {
            "name": "Current",
            "units": "A",
            "min": -5.0,
            "max": 5.0,
            "settable": True,
            "variabletype": "numeric",
            "choices": "numeric",
        },
        {
            "name": "Enable_Output",
            "units": "",
            "min": None,
            "max": None,
            "settable": True,
            "variabletype": "choice",
            "choices": "on,off",
        },
        {
            "name": "IPAddress",
            "units": "",
            "min": None,
            "max": None,
            "settable": False,
            "variabletype": "string",
            "choices": "string",
        },
        {
            "name": "SaveDir",
            "units": "",
            "min": None,
            "max": None,
            "settable": False,
            "variabletype": "path",
            "choices": "path",
        },
        {
            "name": "Cam",
            "units": "",
            "min": None,
            "max": None,
            "settable": False,
            "variabletype": "image",
            "choices": "image",
        },
        {
            "name": "Trace",
            "units": "",
            "min": None,
            "max": None,
            "settable": False,
            "variabletype": "1darray",
            "choices": "1darray",
        },
    ]
    spec = DeviceSpec.from_db_metadata("D", "h", 1, meta)
    by = {v.geecs_var: v for v in spec.variables}

    assert "Cam" not in by and "Trace" not in by  # image / 1darray skipped
    assert by["Current"].dtype == "float"
    assert by["Enable_Output"].dtype == "enum"
    assert by["Enable_Output"].choices == ["on", "off"]
    assert by["IPAddress"].dtype == "string"
    assert by["SaveDir"].dtype == "path"  # long-string char-array PV


def test_choice_pointing_at_type_descriptor_is_skipped() -> None:
    """variabletype='choice' but choices='image'/'1darray' is a non-scalar, skip it."""
    meta = [
        {"name": "bakground image", "variabletype": "choice", "choices": "image"},
        {"name": "scopeTrace.Channel0", "variabletype": "choice", "choices": "1darray"},
        {
            "name": "Enable",
            "variabletype": "choice",
            "choices": "on,off",
            "settable": True,
        },
    ]
    spec = DeviceSpec.from_db_metadata("D", "h", 1, meta)
    by = {v.geecs_var: v for v in spec.variables}
    assert "bakground image" not in by  # image skipped, not a 1-option enum
    assert "scopeTrace.Channel0" not in by  # 1darray skipped
    assert by["Enable"].dtype == "enum"  # a real option list is still an enum


def test_timestamp_ladder_default_prefers_acq_then_sys() -> None:
    """Every device subscribes to acq_timestamp (preferred) then systimestamp."""
    dev = DeviceSpec(name="D", host="h", port=1)
    assert dev.timestamp_vars == ["acq_timestamp", "systimestamp"]


def test_blank_variabletype_inferred_from_choices() -> None:
    """NULL variabletype but a real option list => enum (real U_VisaPlungers case)."""
    meta = [
        {
            "name": "DigitalOutput.Channel 3",
            "variabletype": None,
            "choices": "on,off",
            "settable": True,
        },
        {
            "name": "Reading",
            "variabletype": None,
            "choices": "numeric",
            "settable": False,
        },
        {"name": "Where", "variabletype": None, "choices": "path", "settable": False},
    ]
    spec = DeviceSpec.from_db_metadata("U_VisaPlungers", "h", 1, meta)
    by = {v.geecs_var: v for v in spec.variables}
    assert by["DigitalOutput.Channel 3"].dtype == "enum"  # not float!
    assert by["DigitalOutput.Channel 3"].choices == ["on", "off"]
    assert by["Reading"].dtype == "float"  # 'numeric' descriptor
    assert by["Where"].dtype == "path"  # 'path' descriptor -> long string


def test_choice_without_options_falls_back_to_string() -> None:
    """A choice variable with no options resolves to a plain string PV."""
    meta = [
        {
            "name": "X",
            "units": "",
            "min": None,
            "max": None,
            "settable": False,
            "variabletype": "choice",
            "choices": None,
        },
    ]
    spec = DeviceSpec.from_db_metadata("D", "h", 1, meta)
    assert spec.variables[0].dtype == "string"


def test_choice_exceeding_ca_enum_limits_falls_back_to_string() -> None:
    """A choice with a >26-char label or >16 options can't be a CA enum."""
    long_label = [
        {
            "name": "L",
            "variabletype": "choice",
            "choices": "MZ switch and encoder Index: 27,short",
        },
    ]
    assert (
        DeviceSpec.from_db_metadata("D", "h", 1, long_label).variables[0].dtype
        == "string"
    )

    many = ",".join(f"opt{i}" for i in range(20))  # 20 > 16 states
    too_many = [{"name": "M", "variabletype": "choice", "choices": many}]
    spec = DeviceSpec.from_db_metadata("D", "h", 1, too_many)
    assert spec.variables[0].dtype == "string"
    assert spec.variables[0].choices == []


def _patch_experiment_db(
    monkeypatch,
    *,
    endpoints: dict,
    var_map: dict,
    sub_map: dict,
) -> None:
    """Stub the three batched GeecsDb queries from_geecs_experiment issues."""
    from geecs_ca_gateway.db.geecs_db import GeecsDb

    monkeypatch.setattr(
        GeecsDb,
        "get_experiment_devices",
        classmethod(lambda cls, experiment, *, enabled_only=True: endpoints),
    )
    monkeypatch.setattr(
        GeecsDb,
        "get_experiment_device_variables",
        classmethod(lambda cls, experiment, *, enabled_only=True: var_map),
    )
    monkeypatch.setattr(
        GeecsDb,
        "get_subscribed_variables",
        classmethod(lambda cls, experiment, *, enabled_only=True: sub_map),
    )


def _meta(name: str, *, settable: bool = False) -> dict:
    return {"name": name, "units": "", "min": None, "max": None, "settable": settable}


def test_from_geecs_experiment_subscribed_only(monkeypatch) -> None:
    """Default: down-select to get='yes' vars; a broken device skips, not aborts."""
    _patch_experiment_db(
        monkeypatch,
        endpoints={"U_A": ("h", 1), "U_B": ("h", 2), "U_BAD": ("h", 3)},
        var_map={
            "U_A": [_meta("Current", settable=True), _meta("Voltage")],
            "U_B": [_meta("Current", settable=True), _meta("Voltage")],
            "U_BAD": [{"units": "malformed row without a name"}],
        },
        sub_map={"U_A": ["Current"], "U_B": ["Current", "Voltage"], "U_BAD": ["X"]},
    )

    cfg = GatewayConfig.from_geecs_experiment("Undulator")
    assert {d.name for d in cfg.devices} == {"U_A", "U_B"}  # U_BAD skipped, not fatal
    by_name = {d.name: d for d in cfg.devices}
    # U_A monitors only Current; Voltage is neither subscribed nor settable
    assert {v.geecs_var for v in by_name["U_A"].variables} == {"Current"}
    assert {v.geecs_var for v in by_name["U_B"].variables} == {"Current", "Voltage"}
    gw = GeecsCaGateway(cfg)
    assert "Undulator:U_B:Voltage" in gw.pvdb


def test_from_geecs_experiment_keeps_settable_only_devices(monkeypatch) -> None:
    """A device with zero get='yes' variables keeps its :SP control surface.

    Losing every PV of a device just because nothing is monitored per shot was
    a real deployment gap (DEPLOYMENT.md documented it as known): the device's
    settable variables are its control surface and must survive subscribed mode.
    """
    _patch_experiment_db(
        monkeypatch,
        endpoints={"U_MON": ("h", 1), "U_CTRL": ("h", 2), "U_IDLE": ("h", 3)},
        var_map={
            "U_MON": [_meta("Voltage")],
            "U_CTRL": [_meta("save", settable=True), _meta("Voltage")],
            "U_IDLE": [_meta("Voltage")],  # nothing subscribed, nothing settable
        },
        sub_map={"U_MON": ["Voltage"]},  # U_CTRL and U_IDLE absent from get-map
    )

    cfg = GatewayConfig.from_geecs_experiment("Undulator")
    by_name = {d.name: d for d in cfg.devices}
    # control surface survives: only the settable variable, not the readback set
    assert {v.geecs_var for v in by_name["U_CTRL"].variables} == {"save"}
    gw = GeecsCaGateway(cfg)
    assert "Undulator:U_CTRL:save:SP" in gw.pvdb
    # a device exposing nothing at all is dropped (no pointless connections)
    assert "U_IDLE" not in by_name
    # --no-settable restores the strict get-list behavior
    cfg = GatewayConfig.from_geecs_experiment("Undulator", include_settable=False)
    assert {d.name for d in cfg.devices} == {"U_MON"}


def test_from_geecs_experiment_warns_on_endpointless_device(
    monkeypatch, caplog
) -> None:
    """A get-map device missing from the device table is warned about, not lost."""
    _patch_experiment_db(
        monkeypatch,
        endpoints={"U_A": ("h", 1)},
        var_map={"U_A": [_meta("Current", settable=True)]},
        sub_map={"U_A": ["Current"], "U_GHOST": ["Current"]},
    )

    with caplog.at_level("WARNING"):
        cfg = GatewayConfig.from_geecs_experiment("Undulator")
    assert {d.name for d in cfg.devices} == {"U_A"}
    assert any("U_GHOST" in r.message for r in caplog.records)


def test_from_geecs_experiment_all_variables(monkeypatch) -> None:
    """subscribed_only=False exposes every variable of every enabled device."""
    _patch_experiment_db(
        monkeypatch,
        endpoints={"U_A": ("h", 1), "U_B": ("h", 2)},
        var_map={
            "U_A": [_meta("Current", settable=True), _meta("Voltage")],
            "U_B": [_meta("Voltage")],
        },
        sub_map={},  # not consulted when subscribed_only=False
    )

    cfg = GatewayConfig.from_geecs_experiment("Undulator", subscribed_only=False)
    by_name = {d.name: d for d in cfg.devices}
    assert {v.geecs_var for v in by_name["U_A"].variables} == {"Current", "Voltage"}
    assert {v.geecs_var for v in by_name["U_B"].variables} == {"Voltage"}
