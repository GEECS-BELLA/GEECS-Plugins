"""GatewaySetpointPut — the one blessed gateway ``:SP`` put primitive.

Pins the centralized addressing rule (``ca://`` stripped for raw CA, other
schemes rejected — issue #490), each consumer's wire-value convention (the
ShotController always-string pin must stay byte-identical — hardware-proven),
the timeout policy, ``AsyncStatus`` wrapping, and mock behavior.
"""

from __future__ import annotations

import pytest

pytest.importorskip("aioca")  # the raw transport needs the `ca` extra

import aioca  # noqa: E402

from geecs_bluesky.devices.ca.gateway_put import (  # noqa: E402
    GatewaySetpointPut,
    bare_pv,
    wire_value,
)
from geecs_bluesky.shot_controller import CaPutSetter  # noqa: E402


class _CaputRecorder:
    """Stands in for ``aioca.caput``: records every call, resolves at once."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def __call__(self, pv, value, **kwargs) -> None:
        self.calls.append((pv, value, kwargs))


@pytest.fixture()
def caput(monkeypatch) -> _CaputRecorder:
    recorder = _CaputRecorder()
    monkeypatch.setattr(aioca, "caput", recorder)
    return recorder


class _SignalRecorder:
    """Stands in for a typed ophyd ``:SP`` signal: records ``set`` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def set(self, value, **kwargs) -> None:
        self.calls.append((value, kwargs))


# ---------------------------------------------------------------------------
# The addressing rule — the one place the ophyd-vs-aioca dialect is decided
# ---------------------------------------------------------------------------


def test_bare_pv_strips_the_ca_scheme() -> None:
    assert bare_pv("ca://Undulator:U_Dev:Var:SP") == "Undulator:U_Dev:Var:SP"


def test_bare_pv_passes_bare_names_through() -> None:
    assert bare_pv("Undulator:U_Dev:Var:SP") == "Undulator:U_Dev:Var:SP"


@pytest.mark.parametrize("pv", ["pva://Undulator:U_Dev:Var:SP", "http://x/y"])
def test_bare_pv_rejects_non_ca_schemes(pv: str) -> None:
    with pytest.raises(ValueError):
        bare_pv(pv)


def test_primitive_normalizes_schemed_names_at_construction() -> None:
    """The issue-#490 invariant: a raw put never sees a ``ca://`` name."""
    put = GatewaySetpointPut("ca://Undulator:U_Dev:Var:SP", mock=True)
    assert put._pv == "Undulator:U_Dev:Var:SP"


# ---------------------------------------------------------------------------
# Constructor policy
# ---------------------------------------------------------------------------


def test_exactly_one_transport_is_required() -> None:
    with pytest.raises(ValueError):
        GatewaySetpointPut()
    with pytest.raises(ValueError):
        GatewaySetpointPut("X:SP", signal=_SignalRecorder())


def test_mock_is_raw_transport_only() -> None:
    with pytest.raises(ValueError):
        GatewaySetpointPut(signal=_SignalRecorder(), mock=True)


def test_raw_transport_requires_a_timeout() -> None:
    with pytest.raises(ValueError):
        GatewaySetpointPut("X:SP", timeout=None)


# ---------------------------------------------------------------------------
# Raw-CA transport: wire conventions and timeout policy
# ---------------------------------------------------------------------------


async def test_shot_control_convention_is_byte_identical(caput) -> None:
    """CaPutSetter: every value goes as its wire string, 10 s default budget
    (the hardware-proven ShotController behavior — do not drift this)."""
    setter = CaPutSetter("Undulator:U_DG645_ShotControl:Amplitude_Ch_AB:SP")
    await setter.set(4.0)
    assert caput.calls == [
        (
            "Undulator:U_DG645_ShotControl:Amplitude_Ch_AB:SP",
            "4.0",
            {"wait": True, "timeout": 10.0},
        )
    ]


async def test_wire_value_convention_sends_numerics_natively(caput) -> None:
    """The action convention: native numbers, strings as wire strings."""
    put = GatewaySetpointPut("X:SP", coerce=wire_value, timeout=30.0)
    await put.put(5.0)
    await put.put("External rising edges")
    assert caput.calls == [
        ("X:SP", 5.0, {"wait": True, "timeout": 30.0}),
        ("X:SP", "External rising edges", {"wait": True, "timeout": 30.0}),
    ]


async def test_per_put_timeout_overrides_the_default(caput) -> None:
    put = GatewaySetpointPut("X:SP", timeout=10.0)
    await put.put(1.0, timeout=45.0)
    assert caput.calls[0][2]["timeout"] == 45.0


async def test_mock_records_instead_of_touching_ca(caput) -> None:
    put = GatewaySetpointPut("X:SP", coerce=wire_value, mock=True)
    await put.set(4.0)
    assert put.last_mock_put == "4.0"
    assert caput.calls == []


# ---------------------------------------------------------------------------
# Signal transport: CaSettable/CaMotor Layer-1 semantics
# ---------------------------------------------------------------------------


async def test_signal_transport_defers_to_signal_default_timeout() -> None:
    """timeout=None → the signal's own default (the CaSettable behavior)."""
    signal = _SignalRecorder()
    put = GatewaySetpointPut(signal=signal, timeout=None)
    await put.put(3.5)
    assert signal.calls == [(3.5, {})]


async def test_signal_transport_passes_the_move_budget_through() -> None:
    """A per-put timeout reaches signal.set (CaMotor's move_timeout)."""
    signal = _SignalRecorder()
    put = GatewaySetpointPut(signal=signal, timeout=None)
    await put.put(3.5, timeout=30.0)
    assert signal.calls == [(3.5, {"timeout": 30.0})]
