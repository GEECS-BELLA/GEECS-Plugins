"""Tests for gateway-local derived numeric channels."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from geecs_ca_gateway.config import DeviceSpec, GatewayConfig, VariableSpec
from geecs_ca_gateway.derived import (
    DerivedChannelSpec,
    DerivedExpressionError,
    DerivedInputSpec,
    ExpressionEvaluator,
    load_derived_channels,
)
from geecs_ca_gateway.gateway import GeecsCaGateway
from geecs_ca_gateway.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer

pytestmark = pytest.mark.fake_server


async def _wait_until(predicate, timeout: float = 6.0, interval: float = 0.05) -> bool:
    """Poll ``predicate`` until true or ``timeout`` elapses."""
    waited = 0.0
    while waited < timeout:
        if predicate():
            return True
        await asyncio.sleep(interval)
        waited += interval
    return predicate()


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
  - device: U_ChamberVac
    variable: Pressure
    expression: "10**(v - 5)"
    inputs:
      - symbol: v
        device: U_DaqPad1
        variable: "Analog Input 10"
    egu: Torr
""",
        encoding="utf-8",
    )

    [spec] = load_derived_channels(path)
    assert spec.device == "U_ChamberVac"
    assert spec.inputs[0].variable == "Analog Input 10"
    assert spec.egu == "Torr"


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
