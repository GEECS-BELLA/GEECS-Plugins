"""Hermetic tests for CaTelemetryReadable — the soft Tier-2 telemetry device.

Mock backends, no gateway.  Verifies the softness contract: telemetry columns
carry the ``telemetry_`` name prefix, a read returns values normally, and a
failing signal read degrades to a NaN cell rather than raising into the plan
(so telemetry can never abort a scan).
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("aioca")  # CA backend needs the `ca` extra

from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca.telemetry import (  # noqa: E402
    TELEMETRY_NAME_PREFIX,
    CaTelemetryReadable,
)


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


async def test_telemetry_is_read_only_no_setpoint_signals() -> None:
    dev = CaTelemetryReadable("U_Press", ["Pressure"])
    await dev.connect(mock=True)
    assert not hasattr(dev, "save")
    assert not hasattr(dev, "localsavingpath")
    assert not hasattr(dev, "_setpoint")
