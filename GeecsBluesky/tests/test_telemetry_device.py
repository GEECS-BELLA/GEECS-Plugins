"""Hermetic tests for CaTelemetryReadable — the soft Tier-2 telemetry device.

Mock backends, no gateway.  Verifies the softness contract: telemetry columns
carry the ``telemetry_`` name prefix, a read returns values normally, and a
failing signal read degrades to a dtype-appropriate null cell (NaN for numeric,
``""`` for string) rather than raising into the plan (so telemetry can never
abort a scan).  Also pins the M3c type-tolerance fix: signal dtype is inferred
per-variable (``datatype=None`` default), so a device mixing a numeric and an
enum/string ``get='yes'`` variable logs both — the string var is captured as a
label and the device is not dropped for its type.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("aioca")  # CA backend needs the `ca` extra

from ophyd_async.core import set_mock_value  # noqa: E402

import geecs_bluesky.devices.ca.telemetry as telemetry_mod  # noqa: E402
from geecs_bluesky.devices.ca.telemetry import (  # noqa: E402
    TELEMETRY_NAME_PREFIX,
    CaTelemetryReadable,
)


def _build_typed_telemetry(
    monkeypatch, device, variable_list, dtype_by_var, **kwargs
) -> CaTelemetryReadable:
    """Build a telemetry device whose per-variable signal dtypes are pinned.

    Mock backends cannot infer a PV's native CA type (there is no live PV to
    connect to), so ``datatype=None`` inference always resolves to float under
    mock.  This helper stands in for that inference by pinning each variable's
    signal dtype explicitly — the same per-column outcome real ``datatype=None``
    inference produces on live PVs (numeric → float column, enum/string → string
    column), letting a hermetic test exercise a MIX of dtypes in one device.
    """
    real_signal_r = telemetry_mod.epics_signal_r

    def _typed_signal_r(datatype, read_pv, *args, **inner):
        for var, dtype in dtype_by_var.items():
            if read_pv.endswith(var):
                return real_signal_r(dtype, read_pv, *args, **inner)
        return real_signal_r(datatype, read_pv, *args, **inner)

    monkeypatch.setattr(telemetry_mod, "epics_signal_r", _typed_signal_r)
    return CaTelemetryReadable(device, variable_list, **kwargs)


async def test_telemetry_columns_carry_the_prefix() -> None:
    dev = CaTelemetryReadable("U_Press", ["Pressure"], experiment="Undulator")
    await dev.connect(mock=True)
    assert dev.name.startswith(TELEMETRY_NAME_PREFIX)
    desc = await dev.describe()
    assert all(key.startswith(TELEMETRY_NAME_PREFIX) for key in desc)
    # Not folded into the s-file header map (Tier 2, not save-set scalars).
    assert dev._column_headers == {}


async def test_telemetry_reads_values_normally() -> None:
    # safe_name lowercases variable names into attrs and event keys.
    dev = CaTelemetryReadable("U_Press", ["Pressure", "Temp"], name="telemetry_press")
    await dev.connect(mock=True)
    set_mock_value(dev.pressure, 1.5)
    set_mock_value(dev.temp, 300.0)
    reading = await dev.read()
    assert reading["telemetry_press-pressure"]["value"] == 1.5
    assert reading["telemetry_press-temp"]["value"] == 300.0


async def test_failing_signal_read_degrades_to_nan_not_raise(monkeypatch) -> None:
    dev = CaTelemetryReadable("U_Press", ["Pressure", "Temp"], name="telemetry_press")
    await dev.connect(mock=True)
    set_mock_value(dev.temp, 300.0)

    # Simulate a dead device: pressure's read raises (a CA fault mid-scan).
    async def _boom() -> dict:
        raise TimeoutError("device gone")

    monkeypatch.setattr(dev.pressure, "read", _boom)

    reading = await dev.read()  # must NOT raise
    # The failed signal is a NaN cell; the healthy one still reports its value.
    assert math.isnan(reading["telemetry_press-pressure"]["value"])
    assert reading["telemetry_press-pressure"]["alarm_severity"] == 3  # INVALID
    assert reading["telemetry_press-temp"]["value"] == 300.0


async def test_default_signal_datatype_is_inferred_not_forced_float() -> None:
    """The default is ``datatype=None`` (inference), not the old float-only.

    Forcing ``float`` was the bug: an enum/string ``get='yes'`` PV failed to
    connect and the caller dropped the whole device.  Pinning the default to
    ``None`` lets ophyd-async infer each PV's native type on live PVs (numeric
    stays float, enum/string becomes a label string).
    """
    import inspect

    default = (
        inspect.signature(CaTelemetryReadable.__init__).parameters["datatype"].default
    )
    assert default is None


async def test_mixed_numeric_and_string_vars_both_logged(monkeypatch) -> None:
    """A device mixing a numeric var and a string/enum var: both are captured.

    Regression pin for the M3c telemetry type-drop bug.  ``U_VisaPlungers``-
    style enum PVs (``DigitalOutput.Channel N``) must be logged as their label
    string while the numeric var stays numeric, and the device must NOT be
    dropped for the string var's type.  On the old float-only construction the
    string signal would fail to connect and the whole device (including its
    numeric column) would be dropped by the caller.
    """
    dev = _build_typed_telemetry(
        monkeypatch,
        "U_VisaPlungers",
        ["Pressure", "DigitalOutput.Channel 3"],
        {"pressure": float, "channel_3": str},
        name="telemetry_visaplungers",
    )
    await dev.connect(mock=True)  # must NOT raise a type-coercion error

    set_mock_value(dev.pressure, 2.5)
    set_mock_value(getattr(dev, "digitaloutput_channel_3"), "DigitalOutput.Channel 3")

    desc = await dev.describe()
    # Numeric var stays numeric; string/enum var is a string column.
    assert desc["telemetry_visaplungers-pressure"]["dtype"] == "number"
    assert desc["telemetry_visaplungers-digitaloutput_channel_3"]["dtype"] == "string"

    reading = await dev.read()
    # Both columns present — neither the string var nor the device is dropped.
    assert reading["telemetry_visaplungers-pressure"]["value"] == 2.5
    assert (
        reading["telemetry_visaplungers-digitaloutput_channel_3"]["value"]
        == "DigitalOutput.Channel 3"
    )


async def test_failed_read_of_string_signal_degrades_to_empty_string(
    monkeypatch,
) -> None:
    """A string signal whose read fails yields ``""`` — not a NaN float.

    The soft-tier null substitution is dtype-aware: a failed numeric read is
    NaN, a failed string read is ``""``, so a dead read never forces a string
    telemetry column to hold a float.
    """
    dev = _build_typed_telemetry(
        monkeypatch,
        "U_VisaPlungers",
        ["Pressure", "DigitalOutput.Channel 3"],
        {"pressure": float, "channel_3": str},
        name="telemetry_visaplungers",
    )
    await dev.connect(mock=True)
    set_mock_value(dev.pressure, 2.5)

    channel = getattr(dev, "digitaloutput_channel_3")

    async def _boom() -> dict:
        raise TimeoutError("device gone")

    monkeypatch.setattr(channel, "read", _boom)

    reading = await dev.read()  # must NOT raise
    assert reading["telemetry_visaplungers-pressure"]["value"] == 2.5
    assert reading["telemetry_visaplungers-digitaloutput_channel_3"]["value"] == ""
    assert (
        reading["telemetry_visaplungers-digitaloutput_channel_3"]["alarm_severity"] == 3
    )


async def test_telemetry_is_read_only_no_setpoint_signals() -> None:
    dev = CaTelemetryReadable("U_Press", ["Pressure"])
    await dev.connect(mock=True)
    assert not hasattr(dev, "save")
    assert not hasattr(dev, "localsavingpath")
    assert not hasattr(dev, "_setpoint")
