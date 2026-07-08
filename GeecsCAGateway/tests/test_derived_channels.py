"""Tests for gateway-local derived numeric channels."""

from __future__ import annotations

import asyncio
import socket

import pytest
from caproto import AlarmSeverity, AlarmStatus
from pydantic import ValidationError

from geecs_ca_gateway.config import DeviceSpec, GatewayConfig, VariableSpec
from geecs_ca_gateway.derived import (
    DerivedChannelSpec,
    DerivedExpressionError,
    DerivedInputSpec,
    ExpressionEvaluator,
    default_derived_channels_path,
    load_derived_channels,
    scanner_configs_base,
)
from geecs_ca_gateway.gateway import GeecsCaGateway
from geecs_ca_gateway.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer

pytestmark = pytest.mark.fake_server


def _free_port() -> int:
    """Return an available localhost TCP port."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return port


async def _wait_until(predicate, timeout: float = 6.0, interval: float = 0.05) -> bool:
    """Poll ``predicate`` until true or ``timeout`` elapses."""
    waited = 0.0
    while waited < timeout:
        if predicate():
            return True
        await asyncio.sleep(interval)
        waited += interval
    return predicate()


async def _start_on_port(device: FakeGeecsDevice, port: int) -> FakeGeecsServer:
    """Start a fake server bound to a specific localhost port."""
    last_error: OSError | None = None
    for _ in range(10):
        srv = FakeGeecsServer(device, host="127.0.0.1", port=port)
        try:
            await srv.start()
            return srv
        except OSError as exc:
            last_error = exc
            await asyncio.sleep(0.05)
    assert last_error is not None
    raise last_error


def test_expression_evaluator_supports_convectron_formula() -> None:
    """The restricted evaluator supports numeric Convectron-style formulas."""
    evaluator = ExpressionEvaluator("10**(v - 5)", {"v"})
    assert evaluator.evaluate({"v": 5.0}) == pytest.approx(1.0)
    assert evaluator.evaluate({"v": 6.0}) == pytest.approx(10.0)


def test_expression_evaluator_rejects_non_numeric_python() -> None:
    """Expressions are arithmetic, not arbitrary Python."""
    with pytest.raises(DerivedExpressionError):
        ExpressionEvaluator("__import__('os').system('true')", {"v"})
    with pytest.raises(DerivedExpressionError):
        ExpressionEvaluator("v.real", {"v"})


def test_derived_channel_schema_is_single_source_device_v1() -> None:
    """v1 derived channels are same-source-device only."""
    with pytest.raises(ValidationError, match="one source device"):
        DerivedChannelSpec(
            device="U_ChamberVac",
            variable="Pressure",
            expression="a + b",
            inputs=[
                DerivedInputSpec(symbol="a", device="U_DaqPad1", variable="AI 1"),
                DerivedInputSpec(symbol="b", device="U_DaqPad2", variable="AI 1"),
            ],
        )


def test_load_derived_channels_yaml(tmp_path) -> None:
    """The production overlay file path loads through geecs-schemas."""
    path = tmp_path / "derived_channels.yaml"
    path.write_text(
        """
schema_version: 1
derived_channels:
  - device: TargetChamberPressure
    variable: Pressure
    expression: "10**(v - 6)"
    inputs:
      - symbol: v
        device: U_VacuumGauge
        variable: "AI_mean.Channel 0"
    egu: Torr
    precision: 6
""",
        encoding="utf-8",
    )

    [spec] = load_derived_channels(path)
    assert spec.device == "TargetChamberPressure"
    assert spec.inputs[0].variable == "AI_mean.Channel 0"
    assert spec.egu == "Torr"
    assert spec.precision == 6


def test_load_derived_channels_error_names_path(tmp_path) -> None:
    """Bad overlay files fail with the offending path in the exception."""
    path = tmp_path / "derived_channels.yaml"
    path.write_text("derived_channels: [", encoding="utf-8")

    with pytest.raises(ValueError, match=str(path)):
        load_derived_channels(path)


def test_default_derived_channels_path_from_configs_repo_env(
    monkeypatch, tmp_path
) -> None:
    """The production configs repo convention is discoverable by experiment."""
    repo = tmp_path / "GEECS-Plugins-Configs"
    path = (
        repo
        / "scanner_configs"
        / "experiments"
        / "Undulator"
        / "gateway"
        / "derived_channels.yaml"
    )
    path.parent.mkdir(parents=True)
    path.write_text("schema_version: 1\nderived_channels: []\n", encoding="utf-8")

    monkeypatch.delenv("GEECS_SCANNER_CONFIG_DIR", raising=False)
    monkeypatch.setenv("GEECS_PLUGINS_CONFIGS", str(repo))

    assert scanner_configs_base() == repo / "scanner_configs" / "experiments"
    assert default_derived_channels_path("Undulator") == path
    assert default_derived_channels_path("Missing") is None


def test_derived_pvdb_has_numeric_readback_and_manifest() -> None:
    """A derived spec creates one read-only float PV."""
    cfg = GatewayConfig(
        devices=[
            DeviceSpec(
                name="U_DaqPad1",
                host="h",
                port=1,
                experiment="Undulator",
                variables=[VariableSpec(geecs_var="Analog Input 10", dtype="float")],
            )
        ],
        derived_channels=[
            DerivedChannelSpec(
                device="U_ChamberVac",
                variable="Pressure",
                expression="10**(v - 5)",
                inputs=[
                    DerivedInputSpec(
                        symbol="v",
                        device="U_DaqPad1",
                        variable="Analog Input 10",
                    )
                ],
                egu="Torr",
                precision=4,
            )
        ],
    )
    gw = GeecsCaGateway(cfg)

    pv = "Undulator:U_ChamberVac:Pressure"
    assert pv in gw.pvdb
    assert gw.pvdb[pv].units == "Torr"
    assert gw.pvdb[pv].precision == 4
    assert int(gw.pvdb[pv].severity) == int(AlarmSeverity.INVALID_ALARM)
    assert int(gw.pvdb[pv].status) == int(AlarmStatus.UDF)
    assert gw.manifest[pv] == ("U_DaqPad1", "Pressure", "derived")


async def test_derived_channel_updates_from_same_source_frame() -> None:
    """A source frame writes raw inputs first, then the derived output."""
    labview = 2_082_844_800 + 1_782_949_690.5
    device = FakeGeecsDevice(
        "U_DaqPad1",
        variables={"Analog Input 10": 6.0, "systimestamp": labview},
    )
    async with FakeGeecsServer(device) as srv:
        cfg = GatewayConfig(
            devices=[
                DeviceSpec(
                    name="U_DaqPad1",
                    host=srv.host,
                    port=srv.port,
                    experiment="Undulator",
                    variables=[
                        VariableSpec(geecs_var="Analog Input 10", dtype="float")
                    ],
                )
            ],
            derived_channels=[
                DerivedChannelSpec(
                    device="U_ChamberVac",
                    variable="Pressure",
                    expression="10**(v - 5)",
                    inputs=[
                        DerivedInputSpec(
                            symbol="v",
                            device="U_DaqPad1",
                            variable="Analog Input 10",
                        )
                    ],
                )
            ],
        )
        gw = GeecsCaGateway(cfg)
        await gw.connect()
        await gw.subscribe()
        raw = gw.pvdb["Undulator:U_DaqPad1:Analog_Input_10"]
        pressure = gw.pvdb["Undulator:U_ChamberVac:Pressure"]
        try:
            assert await _wait_until(lambda: pressure.value == pytest.approx(10.0))
            assert raw.value == pytest.approx(6.0)
            assert pressure.timestamp == pytest.approx(1_782_949_690.5, abs=1e-3)
        finally:
            await gw.close()


async def test_derived_only_input_is_subscribed_without_raw_pv() -> None:
    """A derived input need not be exposed as its own readback PV."""
    device = FakeGeecsDevice("U_DaqPad1", variables={"Analog Input 10": 5.0})
    async with FakeGeecsServer(device) as srv:
        cfg = GatewayConfig(
            devices=[
                DeviceSpec(
                    name="U_DaqPad1",
                    host=srv.host,
                    port=srv.port,
                    experiment="Undulator",
                    variables=[],
                )
            ],
            derived_channels=[
                DerivedChannelSpec(
                    device="U_ChamberVac",
                    variable="Pressure",
                    expression="10**(v - 5)",
                    inputs=[
                        DerivedInputSpec(
                            symbol="v",
                            device="U_DaqPad1",
                            variable="Analog Input 10",
                        )
                    ],
                )
            ],
        )
        gw = GeecsCaGateway(cfg)
        assert "Undulator:U_DaqPad1:Analog_Input_10" not in gw.pvdb
        await gw.connect()
        await gw.subscribe()
        pressure = gw.pvdb["Undulator:U_ChamberVac:Pressure"]
        try:
            assert await _wait_until(lambda: pressure.value == pytest.approx(1.0))
        finally:
            await gw.close()


async def test_derived_expression_failure_marks_invalid_calc() -> None:
    """Persistent runtime failures do not leave the last good value looking live."""
    cfg = GatewayConfig(
        devices=[
            DeviceSpec(
                name="U_DaqPad1",
                host="127.0.0.1",
                port=1,
                experiment="Undulator",
                variables=[VariableSpec(geecs_var="Analog Input 10", dtype="float")],
            )
        ],
        derived_channels=[
            DerivedChannelSpec(
                device="U_ChamberVac",
                variable="Pressure",
                expression="1 / v",
                inputs=[
                    DerivedInputSpec(
                        symbol="v",
                        device="U_DaqPad1",
                        variable="Analog Input 10",
                    )
                ],
            )
        ],
    )
    gw = GeecsCaGateway(cfg)
    pressure = gw.pvdb["Undulator:U_ChamberVac:Pressure"]
    callback = gw._make_callback(cfg.devices[0])

    await callback({"Analog Input 10": 2.0})
    assert pressure.value == pytest.approx(0.5)
    assert int(pressure.severity) == int(AlarmSeverity.NO_ALARM)

    await callback({"Analog Input 10": 0.0})
    assert pressure.value == pytest.approx(0.5)
    assert int(pressure.severity) == int(AlarmSeverity.INVALID_ALARM)
    assert int(pressure.status) == int(AlarmStatus.CALC)

    await callback({"Analog Input 10": 4.0})
    assert pressure.value == pytest.approx(0.25)
    assert int(pressure.severity) == int(AlarmSeverity.NO_ALARM)
    assert int(pressure.status) == int(AlarmStatus.NO_ALARM)


async def test_missing_derived_input_marks_invalid_udf() -> None:
    """A mistyped or absent source variable does not serve a valid-looking zero."""
    cfg = GatewayConfig(
        devices=[
            DeviceSpec(
                name="U_DaqPad1",
                host="127.0.0.1",
                port=1,
                experiment="Undulator",
                variables=[],
            )
        ],
        derived_channels=[
            DerivedChannelSpec(
                device="U_ChamberVac",
                variable="Pressure",
                expression="10**(v - 5)",
                inputs=[
                    DerivedInputSpec(
                        symbol="v",
                        device="U_DaqPad1",
                        variable="Analog Input 10",
                    )
                ],
            )
        ],
    )
    gw = GeecsCaGateway(cfg)
    pressure = gw.pvdb["Undulator:U_ChamberVac:Pressure"]
    callback = gw._make_callback(cfg.devices[0])

    await callback({"Other Input": 5.0})

    assert pressure.value == pytest.approx(0.0)
    assert int(pressure.severity) == int(AlarmSeverity.INVALID_ALARM)
    assert int(pressure.status) == int(AlarmStatus.UDF)


async def test_derived_reconnect_clears_invalid_even_when_value_unchanged() -> None:
    """Derived deadband cache is cleared so reconnect recovery always posts."""
    port = _free_port()
    device = FakeGeecsDevice("U_DaqPad1", variables={"Analog Input 10": 5.0})
    srv = await _start_on_port(device, port)
    cfg = GatewayConfig(
        devices=[
            DeviceSpec(
                name="U_DaqPad1",
                host="127.0.0.1",
                port=port,
                experiment="Undulator",
                variables=[],
            )
        ],
        derived_channels=[
            DerivedChannelSpec(
                device="U_ChamberVac",
                variable="Pressure",
                expression="10**(v - 5)",
                inputs=[
                    DerivedInputSpec(
                        symbol="v",
                        device="U_DaqPad1",
                        variable="Analog Input 10",
                    )
                ],
            )
        ],
    )
    gw = GeecsCaGateway(cfg, reconnect_min_s=0.2, reconnect_max_s=0.4)
    await gw.connect()
    await gw.subscribe()
    pressure = gw.pvdb["Undulator:U_ChamberVac:Pressure"]
    try:
        assert await _wait_until(lambda: pressure.value == pytest.approx(1.0))
        assert int(pressure.severity) == int(AlarmSeverity.NO_ALARM)

        await srv.stop()
        assert await _wait_until(
            lambda: int(pressure.severity) == int(AlarmSeverity.INVALID_ALARM)
        )

        srv2 = await _start_on_port(
            FakeGeecsDevice("U_DaqPad1", variables={"Analog Input 10": 5.0}),
            port,
        )
        try:
            assert await _wait_until(
                lambda: int(pressure.severity) == int(AlarmSeverity.NO_ALARM)
                and pressure.value == pytest.approx(1.0)
            )
        finally:
            await srv2.stop()
    finally:
        await gw.close()
