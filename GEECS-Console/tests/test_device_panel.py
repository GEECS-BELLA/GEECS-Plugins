"""Device-panel seam tests: pure helpers + GatewayDevicePanel with a fake aioca.

No live CA anywhere: the real backend's lazy ``import aioca`` (inside the
coroutines it submits to its persistent loop) is intercepted by planting a
fake module in ``sys.modules``, so subscribe/unsubscribe/set exercise the
real threading + event-loop machinery against recorded fakes.
"""

import sys
import time
import types

import pytest

from geecs_console.services.device_panel import (
    DevicePanelBackend,
    GatewayDevicePanel,
    StubDevicePanel,
    format_readback,
    parse_device_variable,
    parse_set_value,
    readback_pv,
    setpoint_pv,
)


def wait_for(predicate, timeout=3.0):
    """Poll *predicate* until true or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


class TestParseDeviceVariable:
    def test_simple_pair(self):
        assert parse_device_variable("U_Hexapod:ypos") == ("U_Hexapod", "ypos")

    def test_splits_on_first_colon_only(self):
        # GEECS variable names may contain dots/spaces but never lead with
        # another device name; the split is on the first colon.
        assert parse_device_variable("Dev:Var:extra") == ("Dev", "Var:extra")

    def test_strips_whitespace(self):
        assert parse_device_variable(" Dev : Position.Axis 1 ") == (
            "Dev",
            "Position.Axis 1",
        )

    @pytest.mark.parametrize("text", ["", "nodots", ":var", "dev:", " : "])
    def test_invalid_forms_return_none(self, text):
        assert parse_device_variable(text) is None


class TestParseSetValue:
    def test_numeric_text_becomes_float(self):
        assert parse_set_value("2.5") == 2.5
        assert isinstance(parse_set_value("3"), float)

    def test_non_numeric_text_stays_string(self):
        assert parse_set_value("on") == "on"
        assert parse_set_value(" Single shot ") == "Single shot"


class TestFormatReadback:
    def test_float_fixed_decimals_width_stable(self):
        """Fixed decimals: readback noise in trailing digits must not change
        the string length (window-width jitter, owner report 0.19.1)."""
        assert format_readback(3.141592653589793) == "3.1416"
        assert format_readback(0.5) == "0.5000"
        # The jitter pair that motivated this: same width either way.
        assert len(format_readback(0.05)) == len(format_readback(0.0498))

    def test_float_extreme_magnitudes_go_scientific(self):
        assert format_readback(1.23456e-5) == "1.235e-05"
        assert format_readback(4.2e7) == "4.200e+07"
        assert format_readback(0.0) == "0.0000"

    def test_int_and_string_pass_through(self):
        assert format_readback(42) == "42"
        assert format_readback("Connected") == "Connected"

    def test_bool_renders_as_bool(self):
        assert format_readback(True) == "True"


class TestPvNames:
    def test_readback_pv_is_bare_and_normalized(self):
        # ca_pv applies the gateway naming contract (dots/spaces -> _),
        # bare_pv strips the ca:// scheme raw aioca must never see.
        assert (
            readback_pv("HTU", "U_Hexapod", "Position.Axis 1")
            == "htu:u_hexapod:position_axis_1"
        )

    def test_empty_experiment_drops_out(self):
        assert readback_pv("", "Dev", "Var") == "dev:var"

    def test_setpoint_pv_appends_sp(self):
        assert setpoint_pv("HTU", "Dev", "Var") == "htu:dev:var:SP"


# ----------------------------------------------------------------------
# Protocol / stub
# ----------------------------------------------------------------------


class TestProtocolAndStub:
    def test_stub_satisfies_protocol(self):
        assert isinstance(StubDevicePanel(), DevicePanelBackend)

    def test_gateway_backend_satisfies_protocol(self):
        assert isinstance(GatewayDevicePanel(), DevicePanelBackend)

    def test_stub_subscribe_and_unsubscribe_are_noops(self):
        stub = StubDevicePanel()
        values = []
        stub.subscribe("Exp", "Dev", "Var", values.append)
        stub.unsubscribe()
        assert values == []

    def test_stub_set_raises(self):
        with pytest.raises(RuntimeError, match="not wired"):
            StubDevicePanel().set("Exp", "Dev", "Var", 1.0)


# ----------------------------------------------------------------------
# GatewayDevicePanel against a fake aioca (no live CA)
# ----------------------------------------------------------------------


class FakeSubscription:
    def __init__(self, pv, callback):
        self.pv = pv
        self.callback = callback
        self.closed = False

    def close(self):
        self.closed = True


class FakeAioca:
    """Records camonitor/caput calls; stands in for the aioca module."""

    def __init__(self):
        self.subscriptions = []
        self.puts = []

    def make_module(self):
        module = types.ModuleType("aioca")

        def camonitor(pv, callback, **kwargs):
            subscription = FakeSubscription(pv, callback)
            self.subscriptions.append(subscription)
            return subscription

        async def caput(pv, value, wait=False, timeout=None):
            self.puts.append((pv, value, wait, timeout))

        module.camonitor = camonitor
        module.caput = caput
        return module


@pytest.fixture
def fake_aioca(monkeypatch):
    fake = FakeAioca()
    monkeypatch.setitem(sys.modules, "aioca", fake.make_module())
    return fake


class TestGatewayDevicePanel:
    def test_subscribe_opens_monitor_on_bare_readback_pv(self, fake_aioca):
        panel = GatewayDevicePanel()
        panel.subscribe("HTU", "U_Hexapod", "ypos", lambda value: None)
        assert wait_for(lambda: len(fake_aioca.subscriptions) == 1)
        assert fake_aioca.subscriptions[0].pv == "htu:u_hexapod:ypos"
        panel.unsubscribe()

    def test_monitor_values_reach_on_value(self, fake_aioca):
        panel = GatewayDevicePanel()
        values = []
        panel.subscribe("HTU", "Dev", "Var", values.append)
        assert wait_for(lambda: fake_aioca.subscriptions)
        fake_aioca.subscriptions[0].callback(3.25)
        assert values == [3.25]
        panel.unsubscribe()

    def test_resubscribe_closes_previous_monitor(self, fake_aioca):
        panel = GatewayDevicePanel()
        panel.subscribe("HTU", "Dev", "Var", lambda value: None)
        assert wait_for(lambda: len(fake_aioca.subscriptions) == 1)
        panel.subscribe("HTU", "Dev2", "Var2", lambda value: None)
        assert wait_for(lambda: len(fake_aioca.subscriptions) == 2)
        assert wait_for(lambda: fake_aioca.subscriptions[0].closed)
        assert fake_aioca.subscriptions[1].pv == "htu:dev2:var2"
        panel.unsubscribe()
        assert wait_for(lambda: fake_aioca.subscriptions[1].closed)

    def test_straggler_callback_after_unsubscribe_is_dropped(self, fake_aioca):
        panel = GatewayDevicePanel()
        values = []
        panel.subscribe("HTU", "Dev", "Var", values.append)
        assert wait_for(lambda: fake_aioca.subscriptions)
        subscription = fake_aioca.subscriptions[0]
        panel.unsubscribe()
        # A CA callback already in flight when the monitor was retired must
        # not repaint the panel with a stale device's value.
        subscription.callback(9.9)
        assert values == []

    def test_subscribe_many_opens_one_monitor_per_target(self, fake_aioca):
        panel = GatewayDevicePanel()
        received = []
        panel.subscribe_many(
            "HTU",
            [("U_S3H", "Current"), ("U_S4H", "Current")],
            lambda index, value: received.append((index, value)),
        )
        assert wait_for(lambda: len(fake_aioca.subscriptions) == 2)
        assert [s.pv for s in fake_aioca.subscriptions] == [
            "htu:u_s3h:current",
            "htu:u_s4h:current",
        ]
        fake_aioca.subscriptions[1].callback(-0.1)
        fake_aioca.subscriptions[0].callback(0.05)
        assert received == [(1, -0.1), (0, 0.05)]
        panel.unsubscribe()
        assert wait_for(lambda: all(s.closed for s in fake_aioca.subscriptions))

    def test_subscribe_many_stragglers_dropped_after_unsubscribe(self, fake_aioca):
        panel = GatewayDevicePanel()
        received = []
        panel.subscribe_many(
            "HTU",
            [("A", "x"), ("B", "y")],
            lambda index, value: received.append((index, value)),
        )
        assert wait_for(lambda: len(fake_aioca.subscriptions) == 2)
        retired = list(fake_aioca.subscriptions)
        panel.unsubscribe()
        for subscription in retired:
            subscription.callback(9.9)
        assert received == []

    def test_unsubscribe_without_subscribe_is_noop(self):
        GatewayDevicePanel().unsubscribe()  # must not raise, must not spin a loop

    def test_unsubscribe_returns_immediately(self, fake_aioca):
        panel = GatewayDevicePanel()
        panel.subscribe("HTU", "Dev", "Var", lambda value: None)
        assert wait_for(lambda: fake_aioca.subscriptions)
        started = time.monotonic()
        panel.unsubscribe()
        assert time.monotonic() - started < 0.2  # scheduled, never joined

    def test_set_puts_wire_value_to_bare_sp_pv(self, fake_aioca):
        panel = GatewayDevicePanel()
        panel.set("HTU", "Dev", "Position.Axis 1", 2.5)
        assert fake_aioca.puts == [("htu:dev:position_axis_1:SP", 2.5, True, 10.0)]

    def test_set_string_value_goes_as_wire_string(self, fake_aioca):
        panel = GatewayDevicePanel()
        panel.set("", "Dev", "Trigger.Source", "Single shot")
        (pv, value, wait, _timeout) = fake_aioca.puts[0]
        assert pv == "dev:trigger_source:SP"
        assert value == "Single shot"
        assert wait is True

    def test_set_failure_propagates(self, monkeypatch):
        fake = FakeAioca()
        module = fake.make_module()

        async def failing_caput(pv, value, wait=False, timeout=None):
            raise TimeoutError("GEECS did not accept the set")

        module.caput = failing_caput
        monkeypatch.setitem(sys.modules, "aioca", module)
        panel = GatewayDevicePanel()
        with pytest.raises(TimeoutError, match="did not accept"):
            panel.set("HTU", "Dev", "Var", 1.0)
