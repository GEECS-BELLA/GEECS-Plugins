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

import asyncio
import math
import struct

import pytest
from caproto import (
    AlarmSeverity,
    AlarmStatus,
    ChannelData,
    ChannelType,
    SubscriptionType,
)
from caproto._data import CannotExceedLimits
from caproto._utils import ChannelFilter
from caproto.server.common import SubscriptionSpec

from geecs_ca_gateway.channels import make_readback_channel, make_setpoint_channel
from geecs_ca_gateway.config import DeviceSpec, GatewayConfig, VariableSpec
from geecs_ca_gateway.gateway import GeecsCaGateway
from geecs_core.pv_naming import pv_name

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


async def test_setpoint_put_failure_stamps_write_invalid_alarm() -> None:
    """A failed GEECS set alarms the ``:SP`` WRITE/INVALID; success clears it.

    Contract §7 (set-failure observability): put-completion is invisible to
    no-callback writers, so the failure must also surface as channel state —
    the ``:SP`` alarm goes ``WRITE``/``INVALID`` on a failed set and returns
    to ``NO_ALARM`` on the next successful one.
    """
    fail = True

    async def setter(value: Any) -> Any:
        if fail:
            raise RuntimeError("GEECS set failed")

    channel = make_setpoint_channel(
        VariableSpec(geecs_var="Current", dtype="float", settable=True),
        setter,
    )
    assert channel.alarm.severity == AlarmSeverity.NO_ALARM
    with pytest.raises(RuntimeError):
        await channel.write(4.2)
    assert channel.alarm.severity == AlarmSeverity.INVALID_ALARM
    assert channel.alarm.status == AlarmStatus.WRITE

    fail = False
    await channel.write(1.0)
    assert channel.alarm.severity == AlarmSeverity.NO_ALARM
    assert channel.alarm.status == AlarmStatus.NO_ALARM
    assert channel.value == pytest.approx(1.0)


async def test_failed_set_alarm_is_published_on_transition_only() -> None:
    """The failure alarm is *published*; identical repeats and clears are not.

    Contract §7: monitor clients are the audience for the failure alarm, so
    the failed-set ``WRITE``/``INVALID`` write must publish (a state change a
    subscriber never receives would make the whole feature a no-op — this
    test fails if the failure path passes ``publish=False``). A repeated
    identical failure must not re-publish, and the success-path clear rides
    the value publish that follows (``publish=False`` here).
    """
    fail = True

    async def setter(value: Any) -> Any:
        if fail:
            raise RuntimeError("GEECS set failed")

    channel = make_setpoint_channel(
        VariableSpec(geecs_var="Current", dtype="float", settable=True),
        setter,
    )
    calls: list[dict[str, Any]] = []
    original_write = channel.alarm.write

    async def counting_write(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return await original_write(**kwargs)

    channel.alarm.write = counting_write  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await channel.write(4.2)
    assert len(calls) == 1
    assert calls[0].get("publish", True) is True  # the failure IS published

    with pytest.raises(RuntimeError):
        await channel.write(4.3)
    assert len(calls) == 1  # repeated identical failure: no re-publish

    fail = False
    await channel.write(1.0)
    assert len(calls) == 2  # the clear...
    assert calls[1]["publish"] is False  # ...rides the value publish instead


async def test_client_value_error_does_not_alarm_setpoint() -> None:
    """A value-shape error the gateway rejects pre-forward does not alarm.

    Contract §7: the ``WRITE``/``INVALID`` alarm mirrors GEECS *forward*
    outcomes only — a put that fails before any UDP exchange (here an
    uncastable scalar) fails the caput but leaves the alarm untouched.
    (Unknown enum labels are different: they forward verbatim so GEECS
    itself rejects them — §4 — and that rejection *does* alarm.)
    """
    forwarded: list[Any] = []

    async def setter(value: Any) -> Any:
        forwarded.append(value)

    channel = make_setpoint_channel(
        VariableSpec(geecs_var="Current", dtype="float", settable=True),
        setter,
    )
    with pytest.raises(ValueError):
        await channel.write("not-a-number")
    assert forwarded == []  # never reached GEECS
    assert channel.alarm.severity == AlarmSeverity.NO_ALARM


async def test_out_of_ctrl_limit_put_rejected_before_forward() -> None:
    """An out-of-range put fails at the CA layer, BEFORE any UDP forward.

    Contract §2: ``:SP`` channels enforce the DB span as control limits
    pre-forward. The ordering is the load-bearing part — with DB limits
    tighter than the device's, a post-forward check would move the hardware
    and then fail the put. And per §7 it is a *client* error: no ``:SP``
    alarm, nothing on ``LAST_SET_ERROR`` (the setter — where the gateway's
    error recording hangs — is never invoked).
    """
    forwarded: list[Any] = []

    async def setter(value: Any) -> Any:
        forwarded.append(value)

    channel = make_setpoint_channel(
        VariableSpec(geecs_var="Current", dtype="float", settable=True, lo=-5, hi=5),
        setter,
    )
    with pytest.raises(CannotExceedLimits, match="Limits are set to"):
        await channel.write(100.0)
    assert forwarded == []  # the device never saw the put
    assert channel.alarm.severity == AlarmSeverity.NO_ALARM
    assert float(channel.value) != 100.0  # not stored

    await channel.write(2.5)  # in-range put forwards and stores as before
    assert forwarded == [2.5]
    assert float(channel.value) == 2.5


async def test_missing_or_degenerate_db_span_stays_unlimited() -> None:
    """No/one-sided/degenerate DB spans serve an unenforced ``:SP``.

    Contract §2: control limits are served only for well-formed spans (both
    bounds, ``lo < hi``). A variable with no DB limits, one bound, or a
    degenerate ``lo == hi`` row keeps an unlimited setpoint — a bad DB row
    must never brick an axis at the CA layer.
    """
    for spec_kwargs in (
        {},  # no bounds
        {"lo": -5.0},  # one-sided
        {"lo": 0.0, "hi": 0.0},  # degenerate (real rows: U_TRAServer01)
    ):
        forwarded: list[Any] = []

        async def setter(value: Any) -> Any:
            forwarded.append(value)  # noqa: B023 — closure runs this iteration

        channel = make_setpoint_channel(
            VariableSpec(
                geecs_var="Current", dtype="float", settable=True, **spec_kwargs
            ),
            setter,
        )
        await channel.write(1e6)  # any value forwards
        assert forwarded == [1e6], f"blocked with {spec_kwargs!r}"


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
# PV_CONTRACT.md §4 — archive deadband (DBE_LOG)
# ---------------------------------------------------------------------------

#: The Archiver Appliance's CA monitor mask (its MonitorMask.ARCHIVE).
ARCHIVER_MASK = SubscriptionType.DBE_LOG | SubscriptionType.DBE_ALARM
#: A display/ophyd client's mask (caproto/pyepics/Phoebus default).
LIVE_MASK = SubscriptionType.DBE_VALUE | SubscriptionType.DBE_ALARM


class _Monitors:
    """Two real caproto subscriptions on one channel: an archiver and a display."""

    def __init__(self, channel: ChannelData) -> None:
        self.channel = channel
        self.queue: asyncio.Queue = asyncio.Queue()
        no_filter = ChannelFilter(ts=False, dbnd=None, arr=None, sync=None)
        self.specs = {
            name: SubscriptionSpec(
                db_entry=channel,
                data_type_name=ChannelType.TIME_DOUBLE.name,
                mask=mask,
                channel_filter=no_filter,
            )
            for name, mask in (
                ("archiver", ARCHIVER_MASK),
                ("live", LIVE_MASK),
                ("alarm_only", SubscriptionType.DBE_ALARM),
            )
        }

    async def start(self) -> None:
        for spec in self.specs.values():
            await self.channel.subscribe(self.queue, spec, None)
        self.received()  # drop the on-subscribe initial readings

    def received(self) -> dict[str, list[float]]:
        """Values each monitor was sent since the last call."""
        out: dict[str, list[float]] = {name: [] for name in self.specs}
        while not self.queue.empty():
            specs, _metadata, values, _flags, _sub = self.queue.get_nowait()
            for name, spec in self.specs.items():
                if spec in specs:
                    # caproto queues the wire (big-endian) encoding
                    out[name].append(struct.unpack(">d", bytes(values))[0])
        return out


def _gated_gateway(archive_deadband: float | None) -> GeecsCaGateway:
    return GeecsCaGateway(
        GatewayConfig(
            devices=[
                DeviceSpec(
                    name=DEVICE,
                    host="127.0.0.1",
                    port=1,
                    variables=[
                        VariableSpec(
                            geecs_var="Pos",
                            dtype="float",
                            archive_deadband=archive_deadband,
                        )
                    ],
                )
            ]
        )
    )


async def test_archive_deadband_gates_only_the_archivers_events() -> None:
    """Live monitors see every change; the archiver only moves beyond the deadband.

    Contract §4: the archive deadband is the EPICS ADEL — a ``DBE_LOG`` event
    when the value left the band around the last *archived* value (so slow
    drift still archives once it accumulates), plus the first frame and a
    NaN crossing.  ``DBE_VALUE`` clients (displays, ophyd, scans) are never
    throttled by it.  Uses real caproto subscriptions, so it fails if the
    channel stops filtering by mask (caproto 1.3.0 does not do it itself).
    """
    gw = _gated_gateway(archive_deadband=0.01)
    channel, _spec = gw._readbacks[DEVICE]["Pos"]
    monitors = _Monitors(channel)
    await monitors.start()
    callback = gw._make_callback(gw.config.devices[0])

    frames = [5.0, 5.004, 5.008, 5.012, 5.013, 5.0, math.nan, 5.0]
    for value in frames:
        await callback({"Pos": value})

    got = monitors.received()
    assert got["live"] == pytest.approx(frames, nan_ok=True)
    # 5.004/5.008 inside the band; 5.012 drifted out of it; 5.013 inside the
    # new band; 5.0 left it; NaN and the return from NaN always archive.
    assert got["archiver"] == pytest.approx(
        [5.0, 5.012, 5.0, math.nan, 5.0], nan_ok=True
    )


async def test_alarm_change_inside_the_deadband_is_archived(monkeypatch) -> None:
    """An alarm transition reaches the archiver — and alarm-only monitors — inside the band."""
    gw = _gated_gateway(archive_deadband=1.0)
    channel, _spec = gw._readbacks[DEVICE]["Pos"]
    monitors = _Monitors(channel)
    await monitors.start()
    callback = gw._make_callback(gw.config.devices[0])
    alarm: dict[str, Any] = {}
    monkeypatch.setattr(gw, "_alarm_write_kwargs", lambda *_args: dict(alarm))

    await callback({"Pos": 5.0})
    alarm.update(severity=AlarmSeverity.MAJOR_ALARM, status=AlarmStatus.HIHI)
    await callback({"Pos": 5.1})  # alarm raised: archived
    await callback({"Pos": 5.2})  # same alarm, inside the band: live only

    got = monitors.received()
    assert got["archiver"] == pytest.approx([5.0, 5.1])
    assert got["alarm_only"] == pytest.approx(
        [5.1]
    )  # the transition, flagged DBE_ALARM


async def test_no_archive_deadband_reaches_every_monitor() -> None:
    """``archive_deadband=None`` (DB NULL) is today's behaviour: every change archived."""
    gw = _gated_gateway(archive_deadband=None)
    channel, _spec = gw._readbacks[DEVICE]["Pos"]
    assert "ArchiveGated" not in type(channel).__name__  # a stock channel
    monitors = _Monitors(channel)
    await monitors.start()
    callback = gw._make_callback(gw.config.devices[0])

    for value in (5.0, 5.0001, 5.0002):
        await callback({"Pos": value})

    got = monitors.received()
    assert got["archiver"] == got["live"] == pytest.approx([5.0, 5.0001, 5.0002])


async def test_unflagged_writes_still_reach_the_archiver() -> None:
    """A write without explicit flags (INVALID marking, …) reaches every monitor."""
    gw = _gated_gateway(archive_deadband=0.01)
    channel, _spec = gw._readbacks[DEVICE]["Pos"]
    monitors = _Monitors(channel)
    await monitors.start()

    await channel.write(
        5.0, severity=AlarmSeverity.INVALID_ALARM, status=AlarmStatus.COMM
    )

    got = monitors.received()
    assert got["archiver"] == got["live"] == pytest.approx([5.0])


def test_archive_deadband_comes_from_its_own_db_field() -> None:
    """``archive_deadband`` maps through; ``tolerance`` still sets no deadband of any kind."""
    meta = [
        {
            "name": "Voltage",
            "units": "V",
            "min": -10.0,
            "max": 10.0,
            "settable": False,
            "variabletype": "numeric",
            "choices": None,
            "tolerance": 0.01,
            "archive_deadband": 0.002,
        },
        {
            "name": "Current",
            "units": "A",
            "min": -5.0,
            "max": 5.0,
            "settable": True,
            "variabletype": "numeric",
            "choices": None,
            "tolerance": 0.05,
        },
    ]
    voltage, current = DeviceSpec.from_db_metadata("U_S1H", "h", 1, meta).variables
    assert voltage.archive_deadband == pytest.approx(0.002)
    assert voltage.deadband == 0.0  # the monitor deadband is untouched
    assert current.archive_deadband is None


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
