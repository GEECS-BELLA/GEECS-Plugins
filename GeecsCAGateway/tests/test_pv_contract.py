"""Thin example-tests pinning claims made in ``PV_CONTRACT.md``.

Each test here backs a specific statement in the contract document that no
existing test already covers; the document's "Pinned-by test map" is the index.
Broader behavioral coverage lives in the layer test files (``test_naming``,
``test_channels``, ``test_gateway``, ``test_transport``, …) — do not duplicate
it here, reference it from the contract instead.

No sockets, no CA server: these tests exercise the naming/config/channel
helpers and the gateway's fan-out callback directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from geecs_ca_gateway.channels import make_readback_channel, make_setpoint_channel
from geecs_ca_gateway.config import DeviceSpec, GatewayConfig, VariableSpec
from geecs_ca_gateway.gateway import GeecsCaGateway
from geecs_ca_gateway.pv_naming import pv_name

DEVICE = "U_ESP_JetXYZ"


# ---------------------------------------------------------------------------
# PV_CONTRACT.md §1 — naming
# ---------------------------------------------------------------------------


def test_pv_name_drops_falsy_parts_and_normalizes() -> None:
    """Falsy namespace parts drop out; every component is normalized.

    Contract §1: the experiment prefix is optional — an absent (``""``/``None``)
    part simply drops out of the ``:``-join — and ``:`` is applied only between
    components, never inside one.
    """
    assert pv_name("", "U_S1H", "Current") == "u_s1h:current"
    assert pv_name(None, "U_S1H", "Current") == "u_s1h:current"
    assert (
        pv_name("Undulator", "U DG645", "Trigger.Source")
        == "undulator:u_dg645:trigger_source"
    )


# ---------------------------------------------------------------------------
# PV_CONTRACT.md §6 — the reserved status namespace is collision-guarded
# ---------------------------------------------------------------------------


def test_status_pv_namespace_is_collision_guarded() -> None:
    """A device/variable shadowing a status PV is a startup error, not a clobber.

    Contract §6: per-device ``…:CONNECTED`` and ``[Experiment:]CAGateway:*``
    go through the same collision guard as data PVs.
    """
    # A GEECS variable literally named CONNECTED collides with the status PV.
    shadow_var = GatewayConfig(
        devices=[
            DeviceSpec(
                name="U_Dev",
                host="h",
                port=1,
                variables=[VariableSpec(geecs_var="CONNECTED")],
            )
        ]
    )
    with pytest.raises(ValueError, match="collision"):
        GeecsCaGateway(shadow_var)

    # A GEECS device named CAGateway collides with the gateway diagnostics.
    shadow_dev = GatewayConfig(
        devices=[
            DeviceSpec(
                name="CAGateway",
                host="h",
                port=1,
                experiment="Test",
                variables=[VariableSpec(geecs_var="UPTIME")],
            )
        ]
    )
    with pytest.raises(ValueError, match="collision"):
        GeecsCaGateway(shadow_dev)


# ---------------------------------------------------------------------------
# PV_CONTRACT.md §2 — setpoint put failure leaves the PV unchanged
# ---------------------------------------------------------------------------


async def test_setpoint_put_failure_leaves_value_unstored() -> None:
    """A failing GEECS set fails the CA put and does NOT store the value.

    Contract §2: the setter runs *before* the value is stored, so a rejected /
    failed / timed-out GEECS set leaves the ``:SP`` PV at its previous value.
    """

    class _Boom(RuntimeError):
        pass

    async def failing_setter(value: Any) -> Any:
        raise _Boom("GEECS set failed")

    channel = make_setpoint_channel(
        VariableSpec(geecs_var="Current", dtype="float", settable=True),
        failing_setter,
    )
    before = channel.value
    with pytest.raises(_Boom):
        await channel.write(4.2)
    assert channel.value == before  # not stored — put fails atomically


# ---------------------------------------------------------------------------
# PV_CONTRACT.md §3 — the 0.0 pre-acquisition placeholder
# ---------------------------------------------------------------------------


def test_float_readback_initializes_to_zero_placeholder() -> None:
    """Float readbacks (incl. acq_timestamp) hold 0.0 before any update.

    Contract §3: clients must treat a non-positive ``acq_timestamp`` as
    "no acquisition yet" — 0.0 is the channel's pre-acquisition placeholder,
    never a shot at the epoch.
    """
    channel = make_readback_channel(
        VariableSpec(geecs_var="acq_timestamp", dtype="float")
    )
    assert float(channel.value) == 0.0


# ---------------------------------------------------------------------------
# PV_CONTRACT.md §4 — deadband policy
# ---------------------------------------------------------------------------


def test_db_tolerance_is_not_used_as_monitor_deadband() -> None:
    """DB ``tolerance`` is a set-convergence criterion — deadband stays 0.0.

    Contract §4: through 0.5.0 the DB tolerance was wired as the monitor
    deadband and hid real sub-tolerance motion from readbacks (and therefore
    from recorded scan rows and s-files). Regression guard for the 0.5.1 fix.
    """
    meta = [
        {
            "name": "Current",
            "units": "A",
            "min": -5.0,
            "max": 5.0,
            "settable": True,
            "variabletype": "numeric",
            "choices": None,
            "tolerance": 0.05,  # coarse magnet-PSU convergence criterion
        }
    ]
    spec = DeviceSpec.from_db_metadata("U_S1H", "h", 1, meta)
    assert spec.variables[0].deadband == 0.0


class _RecordingChannel:
    """Fake readback channel recording every posted value."""

    def __init__(self, posts: list[float]) -> None:
        self._posts = posts

    async def write(self, value: Any, **kwargs: Any) -> None:
        """Record the value instead of writing a caproto channel."""
        self._posts.append(value)


async def test_zero_deadband_posts_changes_suppresses_exact_repeats() -> None:
    """At deadband 0.0 every changed frame posts; exact repeats are suppressed.

    Contract §4: a sub-tolerance change (e.g. 5.0 → 5.0001) MUST post — only
    a frame whose value is exactly unchanged is skipped, so static devices
    stay silent without hiding real motion.
    """
    cfg = GatewayConfig(
        devices=[
            DeviceSpec(
                name=DEVICE,
                host="127.0.0.1",
                port=1,
                variables=[VariableSpec(geecs_var="Pos", dtype="float")],
            )
        ]
    )
    gw = GeecsCaGateway(cfg)
    posts: list[float] = []
    _channel, spec = gw._readbacks[DEVICE]["Pos"]
    assert spec.deadband == 0.0  # the contract default
    gw._readbacks[DEVICE]["Pos"] = (_RecordingChannel(posts), spec)
    callback = gw._make_callback(cfg.devices[0])

    await callback({"Pos": 5.0})  # first frame posts
    await callback({"Pos": 5.0})  # exact repeat suppressed
    await callback({"Pos": 5.0001})  # sub-tolerance change still posts
    await callback({"Pos": 5.0001})  # exact repeat suppressed again

    assert posts == [pytest.approx(5.0), pytest.approx(5.0001)]


# ---------------------------------------------------------------------------
# PV_CONTRACT.md §6 — CAGateway:RESTART requests a clean shutdown
# ---------------------------------------------------------------------------


async def test_restart_pv_requests_clean_shutdown() -> None:
    """Writing ``Restart`` sets the shutdown request; ``Idle`` is a no-op.

    Contract §6: ``CAGateway:RESTART`` is the one client-writable status PV
    (devIocStats ``SYSRESET`` pattern); label, index, and numeric-array puts
    all count, and writing ``Idle``/0 must not trigger anything.
    """
    cfg = GatewayConfig(
        devices=[
            DeviceSpec(
                name=DEVICE,
                host="127.0.0.1",
                port=1,
                experiment="Undulator",
                variables=[VariableSpec(geecs_var="Pos", dtype="float")],
            )
        ]
    )
    gw = GeecsCaGateway(cfg)
    restart = gw.pvdb["undulator:cagateway:restart"]

    await restart.write("Idle")
    await restart.write(0)
    assert not gw._restart_requested.is_set()

    await restart.write("Restart")
    assert gw._restart_requested.is_set()

    # index form works too (fresh gateway — the event latches)
    gw2 = GeecsCaGateway(cfg)
    await gw2.pvdb["undulator:cagateway:restart"].write(1)
    assert gw2._restart_requested.is_set()
