"""Tests for building a DeviceSpec from GEECS DB metadata (no network).

Exercises the pure ``DeviceSpec.from_db_metadata`` core using rows shaped like
``GeecsDb.get_device_variables`` returns for the real U_S1H device.
"""

from __future__ import annotations

from geecs_ca_gateway.config import DeviceSpec, GatewayConfig, VariableSpec
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
    assert by["SaveDir"].dtype == "string"  # path -> string


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
    assert by["Where"].dtype == "string"  # 'path' descriptor


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


def test_from_geecs_experiment_subscribed_only(monkeypatch) -> None:
    """Default: down-select to get='yes' vars (passed as include); skip failures."""
    from geecs_bluesky.db.geecs_db import GeecsDb

    monkeypatch.setattr(
        GeecsDb,
        "get_subscribed_variables",
        classmethod(
            lambda cls, experiment, *, enabled_only=True: {
                "U_A": ["Current"],
                "U_BAD": ["Current"],
                "U_B": ["Current", "Voltage"],
            }
        ),
    )
    seen_include: dict[str, object] = {}

    def fake_from_db(cls, name, *, experiment=None, include=None, **kwargs):
        seen_include[name] = include
        if name == "U_BAD":
            raise RuntimeError("device not resolvable")
        return DeviceSpec(
            name=name,
            host="h",
            port=1,
            experiment=experiment,
            variables=[
                VariableSpec(geecs_var=v, settable=True) for v in (include or [])
            ],
        )

    monkeypatch.setattr(DeviceSpec, "from_geecs_db", classmethod(fake_from_db))

    cfg = GatewayConfig.from_geecs_experiment("Undulator")
    assert {d.name for d in cfg.devices} == {"U_A", "U_B"}  # U_BAD skipped, not fatal
    assert seen_include["U_B"] == ["Current", "Voltage"]  # get-vars → include filter
    gw = GeecsCaGateway(cfg)
    assert "Undulator:U_B:Voltage" in gw.pvdb


def test_from_geecs_experiment_all_variables(monkeypatch) -> None:
    """subscribed_only=False enumerates all enabled devices, no include filter."""
    from geecs_bluesky.db.geecs_db import GeecsDb

    monkeypatch.setattr(
        GeecsDb,
        "list_devices",
        classmethod(lambda cls, experiment, *, enabled_only=True: ["U_A", "U_B"]),
    )

    def fake_from_db(cls, name, *, experiment=None, include=None, **kwargs):
        assert include is None  # no down-select
        return DeviceSpec(
            name=name,
            host="h",
            port=1,
            experiment=experiment,
            variables=[VariableSpec(geecs_var="Current")],
        )

    monkeypatch.setattr(DeviceSpec, "from_geecs_db", classmethod(fake_from_db))

    cfg = GatewayConfig.from_geecs_experiment("Undulator", subscribed_only=False)
    assert {d.name for d in cfg.devices} == {"U_A", "U_B"}
