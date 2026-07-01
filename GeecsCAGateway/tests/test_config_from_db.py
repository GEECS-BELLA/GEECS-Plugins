"""Tests for building a DeviceSpec from GEECS DB metadata (no network).

Exercises the pure ``DeviceSpec.from_db_metadata`` core using rows shaped like
``GeecsDb.get_device_variables`` returns for the real U_S1H device.
"""

from __future__ import annotations

from geecs_ca_gateway.config import DeviceSpec, GatewayConfig
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
    assert current.upper_ctrl_limit == 5.0
    assert current.lower_ctrl_limit == -5.0
    assert current.units == "A"
    assert "U_S1H:Current:SP" in gw.pvdb  # settable -> setpoint exists
