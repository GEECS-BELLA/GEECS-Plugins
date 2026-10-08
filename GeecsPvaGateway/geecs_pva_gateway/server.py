"""The PVA gateway: GEECS camera frames and device arrays in, NTNDArray PVs out.

One process serves every device in its config — each stream variable (an
image, or a ``1darray`` lineout / trace) as one PV. Gating is per variable: a
variable's GEECS TCP subscription starts with its first PVA client and stops
with its last, so an unwatched variable costs the LabVIEW device nothing —
each watched variable holds its own subscription connection. Decode runs off
the event loop; delivery is latest-wins (a stalled consumer drops stale
frames, never backlogs). Each variable's ``:connected`` PV shows the state
of that subscription (Idle / Disconnected / Connected), and a watched device
that stays unreachable is re-resolved from the DB at the backoff ceiling
(#854). The served set itself follows the DB while the process lives: a
roster resolver is re-run off-loop every ``ROSTER_INTERVAL_S`` and the
running instance reconciled to it — a device enabled in the DB gains its
PVs, a disabled one loses them, with no restart (#943). Instance identity
PVs (`version`, `heartbeat`, `devices`) support fleet monitoring.
"""

from __future__ import annotations

import asyncio
from geecs_core.db.variable_types import LABVIEW_EPOCH_OFFSET as _LABVIEW_EPOCH_OFFSET
import contextlib
import logging
import socket
import threading
import time
from collections.abc import Callable

import numpy as np
from p4p.nt import NTEnum, NTNDArray, NTScalar
from p4p.server import Server, StaticProvider
from p4p.server.thread import SharedPV

from geecs_pva_gateway.config import instance_pv_prefix
from geecs_core.db.variable_types import TIMESTAMP_LADDER
from geecs_core.pv_naming import CONNECTED_SUFFIX
from geecs_core.transport.tcp_subscriber import GeecsTcpSubscriber
from geecs_pva_gateway.streams import decode_array, decode_image

from geecs_pva_gateway import __version__, file_plugin
from geecs_pva_gateway.config import DeviceSpec, PvaGatewayConfig
from geecs_pva_gateway.file_plugin import HdfFilePlugin

logger = logging.getLogger(__name__)

# LabVIEW epoch (1904) -> Unix epoch (1970), same ladder as the CA gateway.

_TIMESTAMP_VARS = TIMESTAMP_LADDER  # the one ladder (geecs_core.db.variable_types)
_RECONNECT_MIN_S = 0.5
_RECONNECT_MAX_S = 30.0
_HEARTBEAT_PERIOD_S = 5.0
#: Budget for one off-loop DB endpoint lookup, and the ceiling cycles between
#: two of them for a device that stays down — the CA gateway's numbers
#: (``GeecsCaGateway._ENDPOINT_RESOLVE_*``): a blip never pays a DB query, a
#: device left off overnight never holds a MySQL churn open.
_ENDPOINT_RESOLVE_TIMEOUT_S = 10.0
_ENDPOINT_RESOLVE_HOLDOFF_CYCLES = 10
#: The served-set re-read (#943): how often the roster resolver is re-run
#: and the running instance reconciled to its answer.  One minute: a device
#: enabled or disabled in the DB lands within the time an operator takes to
#: switch windows, for four batched queries per instance per minute — nothing
#: a DB notices.  The ``--roster-interval`` default; ``0`` reads once at start.
ROSTER_INTERVAL_S = 60.0
#: Budget for one off-loop roster read.  GeecsDb bounds each connect to
#: ``CONNECT_TIMEOUT_S`` (10 s), so a dead route fails well inside it; a
#: read that outlives the budget is left to finish (never a second one in
#: flight) and its answer, if any, is taken at the next tick.
_ROSTER_RESOLVE_TIMEOUT_S = 30.0
#: Ticks a read may outlive its budget before it is abandoned and a fresh
#: one started: GeecsDb bounds the connect, not the query, so a DB host
#: reset mid-query leaves the thread blocked in ``recv`` for good — abandoning
#: it (one thread left behind, logged) is what keeps the re-read alive.
_ROSTER_ABANDON_TICKS = 10

#: The ``:connected`` states, in enum index order.  ``Idle``: the subscription
#: is gated off (no watcher; nothing is known).  ``Disconnected``: a watcher
#: holds it and the device is unreachable or dropped (MAJOR alarm, as the CA
#: gateway's ``:connected``).  ``Connected``: the subscription is live.
CONNECTED_STATES = ("Idle", "Disconnected", "Connected")
CONNECTED_IDLE, CONNECTED_DOWN, CONNECTED_UP = CONNECTED_STATES

# Process exit code meaning "restart requested via the :restart PV" — the
# service manager relaunches, which re-resolves DB config and (with the
# pull-on-restart launcher) reinstalls from the source clone. Mirrors the
# CA gateway's CAGateway:RESTART -> exit 86 -> systemd relaunch pattern.
RESTART_EXIT_CODE = 86


def _frame_timestamp(update: dict) -> float:
    """Frame time from the GEECS timestamp ladder, else receive time.

    Plausibility is checked on the *converted* (post-epoch-offset) value,
    matching the CA gateway's contract (PV_CONTRACT.md "Timestamp ladder"):
    a LabVIEW value in ``(0, offset]`` — a device counting from boot, or a
    zeroed channel — must fall through to receive time, never become a
    negative Unix timestamp (which would poison downstream consumers that
    key frames on the PVA timestamp, e.g. the file plugin's dedupe and
    the analysis-side ``acq_timestamp`` join).
    """
    for var in _TIMESTAMP_VARS:
        value = update.get(var)
        if isinstance(value, (int, float)):
            converted = float(value) - _LABVIEW_EPOCH_OFFSET
            if converted > 0:
                return converted
    return time.time()


def _connected_value(nt: NTEnum, state: str):
    """The value of one ``:connected`` state (alarm MAJOR when down).

    Wrapped by the PV's own *nt*: p4p posts only the exact type the PV was
    opened with, and every ``NTEnum()`` instance mints its own.
    """
    value = nt.wrap(
        {"index": CONNECTED_STATES.index(state), "choices": list(CONNECTED_STATES)},
        timestamp=time.time(),
    )
    # Set (mark) the alarm fields on every post: p4p posts marked fields only,
    # so a MAJOR left from Disconnected would otherwise outlive the state.
    down = state == CONNECTED_DOWN
    value["alarm.severity"] = 2 if down else 0  # MAJOR while unreachable
    value["alarm.message"] = "device unreachable" if down else ""
    return value


class _Gate:
    """p4p handler that refcounts one variable's client connections."""

    def __init__(self, worker: "_DeviceWorker", var: str) -> None:
        self._worker = worker
        self._var = var

    def onFirstConnect(self, pv: SharedPV) -> None:
        self._worker.retain(self._var)

    def onLastDisconnect(self, pv: SharedPV) -> None:
        self._worker.release(self._var)


class _RestartHandler:
    """p4p handler for the writable :restart PV — any put requests restart."""

    def __init__(self, on_restart) -> None:
        self._on_restart = on_restart

    def put(self, pv: SharedPV, op) -> None:
        logger.warning("restart requested via :restart PV")
        op.done()
        self._on_restart()


class _DeviceWorker:
    """One served device: per-stream-variable PVs, gated + supervised subscriptions.

    A stream variable is an image (decoded as IMAQ) or an array (one of the
    three array wire shapes, at native length) —
    :mod:`geecs_pva_gateway.streams`; everything below the decode is shared.

    Parameters
    ----------
    spec :
        The device, its stream variables and its endpoint from the DB.
    loop :
        The gateway's event loop.
    endpoint_resolver :
        ``device -> (host, port)``, re-queried off-loop once a watched
        variable's reconnect backoff sits at its ceiling, so a device app
        that came up on another port after this gateway started is found
        without a restart (#854; the CA gateway's ``endpoint_resolver``).
        A resolved endpoint on another host is *not* adopted — the served
        set is host-scoped — only logged.  ``None`` keeps the startup
        endpoint forever.
    """

    def __init__(
        self,
        spec: DeviceSpec,
        loop: asyncio.AbstractEventLoop,
        endpoint_resolver: Callable[[str], tuple[str, int]] | None = None,
    ) -> None:
        self._spec = spec
        self._loop = loop
        self._endpoint_resolver = endpoint_resolver
        # The last decoded frame per variable: the file plugin arms on it
        # (#894).  Written by _publish on the loop, read on the writer thread.
        self._last_frame: dict[str, np.ndarray] = {}
        self._pvs: dict[str, SharedPV] = {
            var: SharedPV(
                handler=_Gate(self, var),
                nt=NTNDArray(),
                # An array PV starts as one NaN-free float; an image as one pixel.
                initial=(
                    np.zeros((1,), dtype=np.float64)
                    if spec.is_array(var)
                    else np.zeros((1, 1), dtype=np.uint16)
                ),
            )
            for var in spec.stream_variables
        }
        self._connected_nt: dict[str, NTEnum] = {
            var: NTEnum() for var in spec.stream_variables
        }
        self._connected: dict[str, SharedPV] = {
            var: SharedPV(nt=nt, initial=_connected_value(nt, CONNECTED_IDLE))
            for var, nt in self._connected_nt.items()
        }
        self._connected_state: dict[str, str] = dict.fromkeys(
            spec.stream_variables, CONNECTED_IDLE
        )
        self._clients: dict[str, int] = dict.fromkeys(spec.stream_variables, 0)
        # The file plugin (#806): one per stream variable, a second consumer
        # of the push frame that holds the subscription like a client does.
        # Served only where its writer library is installed (file_plugin.available).
        self._plugins: dict[str, HdfFilePlugin] = {}
        if file_plugin.available():
            for var in spec.stream_variables:
                try:
                    self._plugins[var] = HdfFilePlugin(
                        device=spec.device,
                        variable=var,
                        experiment=spec.experiment,
                        retain=self.retain,
                        release=self.release,
                        scalar_variables=spec.scalar_variables,
                        last_frame=lambda v=var: self._last_frame.get(v),
                        decoder=lambda blob, v=var: self.decode(v, blob),
                        is_array=spec.is_array(var),
                    )
                except Exception:
                    # A plugin that refuses its rows: the ones built before
                    # it have writer threads running — stop them, so a
                    # refused device (the roster re-read's retry) leaks none.
                    for plugin in self._plugins.values():
                        plugin.stop()
                    raise
        self._supervisors: dict[str, asyncio.Task] = {}
        self._latest: dict[str, tuple[str, float]] = {}
        self._publishing: set[str] = set()
        self._stopping = False

    @property
    def device(self) -> str:
        """The GEECS device name this worker serves."""
        return self._spec.device

    @property
    def spec(self) -> DeviceSpec:
        """The device as the DB described it when this worker was built."""
        return self._spec

    @property
    def capturing_variables(self) -> list[str]:
        """The stream variables with a file-plugin capture session open.

        Read on the event loop from writer-thread state: a verdict for the
        roster reconcile (a device mid-capture is never torn down), not a
        lock — a ``Capture=1`` landing in the same instant is the one race,
        and the stack it opens is cut short.
        """
        return [var for var, plugin in self._plugins.items() if plugin.capturing]

    def provider_entries(self) -> list[tuple[str, str, SharedPV]]:
        """``[(pv_name, variable, SharedPV), ...]`` — one row per variable.

        A list, not a dict: two variables of this camera that normalize to the
        same PV name must both surface so the collision guard can see them
        rather than one silently shadowing the other.
        """
        entries = [
            (self._spec.pv_name_for(var), var, pv) for var, pv in self._pvs.items()
        ]
        entries.extend(
            (self._spec.connected_pv_for(var), f"{var}{CONNECTED_SUFFIX}", pv)
            for var, pv in self._connected.items()
        )
        for plugin in self._plugins.values():
            entries.extend(plugin.provider_entries())
        return entries

    @property
    def plugins(self) -> dict[str, HdfFilePlugin]:
        """The file plugins by stream variable (empty where h5py is not installed)."""
        return self._plugins

    def decode(self, var: str, blob: str) -> tuple[np.ndarray, dict]:
        """One pushed value of *var* → ``(array, NTNDArray attributes)``.

        Images decode as IMAQ; arrays as their wire shape at native length
        (:mod:`geecs_pva_gateway.streams`).  Raises on a
        payload it cannot account for — the caller counts, never guesses.
        Thread-agnostic: the publisher runs it off-loop, the plugin on its
        writer thread.
        """
        if self._spec.is_array(var):
            return decode_array(blob)
        return decode_image(blob)

    async def stop(self) -> None:
        """Cancel all supervisors (gateway shutdown)."""
        # Latch first: a client connecting mid-shutdown must not spawn a
        # supervisor this method has already passed.
        self._stopping = True
        # Snapshot: client disconnects mutate the dict concurrently.
        tasks = list(self._supervisors.values())
        self._supervisors.clear()
        for task in tasks:
            task.cancel()
        # The supervisors' own CancelledErrors come back as results; a
        # cancellation of the *caller* (the roster task at shutdown, #943)
        # still propagates — a per-task try/except swallowed it.
        await asyncio.gather(*tasks, return_exceptions=True)
        for plugin in self._plugins.values():
            await self._loop.run_in_executor(None, plugin.stop)

    # -- gating; retain/release arrive on p4p worker threads ---------------

    def retain(self, var: str) -> None:
        self._loop.call_soon_threadsafe(self._retain, var)

    def release(self, var: str) -> None:
        self._loop.call_soon_threadsafe(self._release, var)

    def _retain(self, var: str) -> None:
        if self._stopping:
            return
        self._clients[var] += 1
        existing = self._supervisors.get(var)
        if self._clients[var] == 1 and (existing is None or existing.done()):
            logger.info("first client for %s %s: subscribing", self._spec.device, var)
            self._supervisors[var] = self._loop.create_task(
                self._run(var), name=f"camera[{self._spec.device}:{var}]"
            )

    def _release(self, var: str) -> None:
        self._clients[var] = max(0, self._clients[var] - 1)
        if self._clients[var] == 0 and var in self._supervisors:
            logger.info("last client for %s %s: unsubscribing", self._spec.device, var)
            self._supervisors.pop(var).cancel()

    # -- subscription supervisor (one per watched variable) ----------------

    def subscription_variables(self, var: str) -> list[str]:
        """The one TCP subscription for *var*.

        The frame, the timestamp ladder, and — where the file plugin serves
        the variable — the device's subscribed scalars, so the per-frame
        attributes come from the same push as the frame
        (still one subscription, no second stream).
        """
        names = [var, *_TIMESTAMP_VARS]
        if var in self._plugins:
            names.extend(s for s in self._spec.scalar_variables if s not in names)
        return names

    def _set_connected(self, var: str, state: str) -> None:
        """Post the variable's ``:connected`` state when it changes (event loop)."""
        if self._connected_state[var] == state:
            return
        self._connected_state[var] = state
        self._connected[var].post(_connected_value(self._connected_nt[var], state))

    async def _resolve_endpoint(self) -> tuple[str, int] | None:
        """Re-query the device's endpoint off-loop; ``None`` keeps the old one."""
        assert self._endpoint_resolver is not None
        try:
            host, port = await asyncio.wait_for(
                asyncio.to_thread(self._endpoint_resolver, self._spec.device),
                timeout=_ENDPOINT_RESOLVE_TIMEOUT_S,
            )
            return str(host).strip(), int(port)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - DB down: keep the last endpoint
            logger.warning(
                "%s: endpoint re-resolve failed (%s); keeping the last endpoint",
                self._spec.device,
                exc,
            )
            return None

    async def _run(self, var: str) -> None:
        """Keep one variable's subscription alive; reconnect on socket drops.

        Silence is not a drop (a box ARMED through a long move pushes
        nothing, #894; a dead peer is the socket's keepalive to detect).
        Once the reconnect backoff sits at its ceiling the outage is not a
        blip: the endpoint is re-asked of the DB every
        ``_ENDPOINT_RESOLVE_HOLDOFF_CYCLES``-th cycle, so a device app that
        came up on another port after this gateway started is redialed
        there instead of found only by a gateway restart (#854).
        """
        backoff = _RECONNECT_MIN_S
        host, port = self._spec.host, self._spec.port
        resolve_holdoff = 0
        try:
            while True:
                subscriber = GeecsTcpSubscriber(host, port)
                try:
                    await subscriber.connect()
                    await subscriber.subscribe(
                        self.subscription_variables(var),
                        lambda update: self._on_frame(var, update),
                        text_variables={var},
                    )
                    self._set_connected(var, CONNECTED_UP)
                    backoff = _RECONNECT_MIN_S
                    resolve_holdoff = 0
                    await subscriber.wait_disconnected()
                    logger.warning(
                        "subscription to %s %s (%s:%s) dropped; reconnecting",
                        self._spec.device,
                        var,
                        host,
                        port,
                    )
                except asyncio.CancelledError:
                    await subscriber.close()
                    raise
                except Exception:  # noqa: BLE001 — retry loop must survive
                    logger.warning(
                        "connect/subscribe to %s %s (%s:%s) failed; retry in %.1fs",
                        self._spec.device,
                        var,
                        host,
                        port,
                        backoff,
                        exc_info=True,
                    )
                await subscriber.close()
                self._set_connected(var, CONNECTED_DOWN)
                if self._endpoint_resolver is not None and backoff >= _RECONNECT_MAX_S:
                    if resolve_holdoff > 0:
                        resolve_holdoff -= 1
                    else:
                        resolve_holdoff = _ENDPOINT_RESOLVE_HOLDOFF_CYCLES
                        resolved = await self._resolve_endpoint()
                        if resolved is not None and resolved != (host, port):
                            if resolved[0] != self._spec.host:
                                logger.warning(
                                    "%s: endpoint moved off this host to %s:%s (DB "
                                    "re-resolve); keeping %s:%s — the served set "
                                    "follows the DB at the next roster re-read",
                                    self._spec.device,
                                    *resolved,
                                    host,
                                    port,
                                )
                            else:
                                logger.warning(
                                    "%s: endpoint moved %s:%s -> %s:%s (DB "
                                    "re-resolve); redialing there",
                                    self._spec.device,
                                    host,
                                    port,
                                    *resolved,
                                )
                                host, port = resolved
                                backoff = _RECONNECT_MIN_S  # dial it promptly
                                resolve_holdoff = 0
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX_S)
        finally:
            # Gated off (or the gateway stopping): nothing is known any more.
            self._set_connected(var, CONNECTED_IDLE)

    # -- frame pipeline ----------------------------------------------------

    def _on_frame(self, var: str, update: dict) -> None:
        """Push-frame callback (event loop): stash latest, schedule publish."""
        blob = update.get(var)
        if not blob or not isinstance(blob, str):
            return
        stamp = _frame_timestamp(update)
        # The file plugin's lossless intake branches off first (#806): its
        # delivery contract is the opposite of the stream's below.
        plugin = self._plugins.get(var)
        if plugin is not None:
            plugin.offer(
                blob,
                stamp,
                time.time(),
                {name: update.get(name) for name in plugin.scalar_variables},
            )
        # Latest-wins slot: an unconsumed frame is replaced, never queued.
        self._latest[var] = (blob, stamp)
        if var not in self._publishing:
            self._publishing.add(var)
            self._loop.create_task(self._publish(var))

    async def _publish(self, var: str) -> None:
        try:
            while (item := self._latest.pop(var, None)) is not None:
                blob, ts = item
                try:
                    image, attrib = await self._loop.run_in_executor(
                        None, self.decode, var, blob
                    )
                    if self._stopping:
                        return  # the PVs are closing (roster removal / shutdown)
                    self._last_frame[var] = image
                    if attrib:
                        self._pvs[var].post(image, timestamp=ts, attrib=attrib)
                    else:
                        self._pvs[var].post(image, timestamp=ts)
                except Exception:  # noqa: BLE001 — retry loop must survive
                    logger.warning(
                        "decode/post failed for %s %s (%d bytes)",
                        self._spec.device,
                        var,
                        len(blob),
                        exc_info=True,
                    )
        finally:
            self._publishing.discard(var)


class GeecsPvaGateway:
    """Serve a :class:`PvaGatewayConfig`'s devices' stream variables as NTNDArray PVs.

    Parameters
    ----------
    config :
        The served set at start, and the host the instance is named after.
    endpoint_resolver :
        ``device -> (host, port)`` for the per-variable supervisors (#854).
    roster_resolver :
        ``() -> [DeviceSpec, ...]``: the served set as the DB sees it *now*,
        built with the same scoping rule as *config* (the CLI passes the
        same ``from_geecs_experiment`` call).  Re-run off-loop every
        *roster_interval_s* and the running instance reconciled to its
        answer: a device that entered the set is served (its PVs and gated
        subscriptions exactly as at start), one that left is dropped —
        unless a file-plugin capture session is open on it, in which case
        the removal waits for the next tick.  A raise or a read that
        outlives its budget keeps the last good set: the set **never
        shrinks on a failure**, only on an answer.  ``None`` (and an
        interval of 0) keeps the startup set for the life of the process.
    roster_interval_s :
        Seconds between two roster reads (:data:`ROSTER_INTERVAL_S`).
    """

    def __init__(
        self,
        config: PvaGatewayConfig,
        *,
        endpoint_resolver: Callable[[str], tuple[str, int]] | None = None,
        roster_resolver: Callable[[], list[DeviceSpec]] | None = None,
        roster_interval_s: float = ROSTER_INTERVAL_S,
    ) -> None:
        self._config = config
        self._endpoint_resolver = endpoint_resolver
        self._roster_resolver = roster_resolver
        self._roster_interval_s = roster_interval_s
        self._workers: list[_DeviceWorker] = []
        self._server: Server | None = None
        self._restart_event: asyncio.Event | None = None
        # The live provider (PVs come and go with the roster) and the
        # collision map over everything it serves: PV name -> (device, var).
        self._provider: StaticProvider | None = None
        self._owners: dict[str, tuple[str, str]] = {}
        self._devices_pv: SharedPV | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Roster bookkeeping: the failure streak (logged once), the devices
        # refused for a PV-name collision and those whose stream variables
        # drifted from the served shape (each logged once while it holds).
        self._roster_failures = 0
        self._refused: dict[str, DeviceSpec] = {}  # device -> the rows refused
        self._drifted: set[str] = set()

    def conf(self) -> dict:
        """Client configuration for the running server (test isolation)."""
        assert self._server is not None
        return self._server.conf()

    @property
    def restart_requested(self) -> bool:
        """True once a client has written the :restart PV."""
        return self._restart_event is not None and self._restart_event.is_set()

    @property
    def pv_names(self) -> list[str]:
        """Every image PV name this config serves (no server needed)."""
        return [
            spec.pv_name_for(var)
            for spec in self._config.devices
            for var in spec.stream_variables
        ]

    @property
    def served_devices(self) -> list[str]:
        """The devices served right now, sorted — the ``:devices`` instance PV's value."""
        return sorted(worker.device for worker in self._workers)

    def _instance_host(self) -> str:
        """Identity for the instance PVs: the served host's address.

        The config's ``host`` (the ``--host`` argument, else the lab-facing
        local address the roster was scoped to) so an instance with no
        device to serve is still ``{exp}:pvagateway:<ip>:*`` — the name the
        fleet probe and the Phoebus screen ask for, and the ``:restart``
        that picks up a newly enabled device.  Falls back to the first
        device's endpoint, then to the machine name (tests without either).
        """
        if self._config.host:
            return self._config.host
        if self._config.devices:
            return self._config.devices[0].host
        return socket.gethostname()

    def _claim(self, worker: _DeviceWorker) -> list[tuple[str, SharedPV]]:
        """Claim the worker's PV names in the collision map; ``[(name, pv), ...]``.

        PV naming is lossy (normalization), so two variables landing on one
        name — within a device or across devices, against what is served
        already — would shadow silently.  Iterates per-variable entries (a
        per-device dict would collapse the within-device case) and raises
        ``ValueError`` claiming nothing, so a refused device leaves no trace.
        """
        claimed: dict[str, tuple[str, str]] = {}
        entries = worker.provider_entries()
        for name, var, _pv in entries:
            source = (worker.device, var)
            owner = self._owners.get(name) or claimed.get(name)
            if owner is not None:
                raise ValueError(
                    f"PV name collision after normalization: {name!r} from "
                    f"{source} and {owner}"
                )
            claimed[name] = source
        self._owners.update(claimed)
        return [(name, pv) for name, _var, pv in entries]

    async def run(self, *, isolate: bool = False) -> None:
        """Serve until cancelled. ``isolate`` sandboxes ports for tests."""
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._owners = {}
        self._workers = []
        providers: dict[str, SharedPV] = {}
        for spec in self._config.devices:
            worker = _DeviceWorker(spec, loop, self._endpoint_resolver)
            self._workers.append(worker)
            providers.update(self._claim(worker))  # a collision refuses to start

        self._restart_event = asyncio.Event()
        prefix = instance_pv_prefix(self._config.experiment, self._instance_host())
        heartbeat_pv = SharedPV(nt=NTScalar("I"), initial=0)
        providers[f"{prefix}:version"] = SharedPV(nt=NTScalar("s"), initial=__version__)
        providers[f"{prefix}:heartbeat"] = heartbeat_pv
        providers[f"{prefix}:restart"] = SharedPV(
            handler=_RestartHandler(
                lambda: loop.call_soon_threadsafe(self._restart_event.set)
            ),
            nt=NTScalar("i"),
            initial=0,
        )
        # The served set as the instance holds it (the roster re-read posts
        # every change), so the fleet probe can diff it against the DB.
        self._devices_pv = SharedPV(nt=NTScalar("as"), initial=self.served_devices)
        providers[f"{prefix}:devices"] = self._devices_pv

        # One provider the server keeps for its lifetime: devices join and
        # leave it as the roster moves (#943); the identity PVs never move.
        self._provider = StaticProvider()
        for name in sorted(providers):
            logger.info("serving %s", name)
            self._provider.add(name, providers[name])

        self._server = Server(providers=[self._provider], isolate=isolate)
        roster_task: asyncio.Task | None = None
        if self._roster_resolver is not None and self._roster_interval_s > 0:
            roster_task = loop.create_task(self._roster_loop(), name="roster")
        try:
            beats = 0
            while not self._restart_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._restart_event.wait(), _HEARTBEAT_PERIOD_S
                    )
                except TimeoutError:
                    beats += 1
                    heartbeat_pv.post(beats)
            logger.warning("shutting down for restart (exit %d)", RESTART_EXIT_CODE)
        finally:
            if roster_task is not None:
                roster_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await roster_task
            for worker in self._workers:
                await worker.stop()
            self._server.stop()
            self._server = None
            self._provider = None

    # -- the served-set re-read (#943) ------------------------------------

    async def _roster_loop(self) -> None:
        """Re-run the roster resolver every interval and reconcile to its answer.

        The read runs off-loop (a dead DB route blocks for seconds, never
        the server) and is never overlapped: a read still running at the
        next tick is waited for again rather than doubled, so a stalled DB
        holds one executor thread, not one per tick.
        """
        assert self._roster_resolver is not None
        pending: asyncio.Future | None = None
        overdue = 0  # ticks the pending read has outlived its budget
        while True:
            await asyncio.sleep(self._roster_interval_s)
            if pending is None:
                pending = self._read_roster()
                overdue = 0
            try:
                resolved = await asyncio.wait_for(
                    asyncio.shield(pending), _ROSTER_RESOLVE_TIMEOUT_S
                )
            except TimeoutError:
                overdue += 1
                self._roster_failed(
                    f"no answer within {_ROSTER_RESOLVE_TIMEOUT_S:.0f} s; "
                    "the read is still running"
                )
                if overdue >= _ROSTER_ABANDON_TICKS:
                    # A query that never returns (DB host reset mid-query:
                    # the connect is bounded, the recv is not).  Let it go
                    # — its thread finishes whenever the socket does — so
                    # the next tick starts afresh instead of waiting forever.
                    logger.warning(
                        "roster re-read hung for %d ticks; abandoning it (its "
                        "thread is left to finish) and starting afresh",
                        overdue,
                    )
                    pending = None
                continue  # else `pending` stays: never a second read in flight
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - DB down: keep the last set
                pending = None
                self._roster_failed(f"{type(exc).__name__}: {exc}")
                continue
            pending = None
            if self._roster_failures:
                logger.info(
                    "roster re-read recovered after %d failed ticks",
                    self._roster_failures,
                )
                self._roster_failures = 0
            try:
                await self._reconcile(list(resolved))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("roster reconcile failed; the served set stands")

    def _read_roster(self) -> asyncio.Future:
        """Run the roster resolver on its own daemon thread, bridged to a loop future.

        Not the default executor: a query the DB never answers would park
        one of its threads for good — a pool shared with the frame decode,
        and one ``asyncio.run`` joins without timeout at exit, so a
        ``:restart`` would log its exit code and never exit.  A daemon
        thread costs an abandoned read nothing and never blocks exit.
        """
        assert self._loop is not None and self._roster_resolver is not None
        loop, resolver = self._loop, self._roster_resolver
        future: asyncio.Future = loop.create_future()
        # An abandoned read's eventual outcome is nobody's business: consume
        # it, or asyncio logs "exception was never retrieved" at GC.
        future.add_done_callback(lambda f: f.cancelled() or f.exception())

        def deliver(setter: Callable, value: object) -> None:
            if not future.done():
                setter(value)

        def work() -> None:
            try:
                outcome = (future.set_result, resolver())
            except Exception as exc:  # noqa: BLE001 - delivered to the loop as-is
                outcome = (future.set_exception, exc)
            with contextlib.suppress(RuntimeError):  # the loop closed: exiting
                loop.call_soon_threadsafe(deliver, *outcome)

        threading.Thread(target=work, name="roster-read", daemon=True).start()
        return future

    def _roster_failed(self, reason: str) -> None:
        """Count a failed tick; log the first of a streak only."""
        self._roster_failures += 1
        if self._roster_failures == 1:
            logger.warning(
                "roster re-read failed (%s); keeping the last good set (%d "
                "devices) — logged once per failure streak",
                reason,
                len(self._workers),
            )

    async def _reconcile(self, resolved: list[DeviceSpec]) -> None:  # noqa: C901
        """Bring the served set to *resolved*: drop the departed, add the new.

        Devices are keyed by name.  A device in both sets keeps its worker
        (its subscriptions, watchers and held frame): an endpoint move is
        the supervisor's business (#854), and a changed stream-variable
        set is logged — once — rather than churned, since re-shaping a
        device drops its watchers.
        """
        wanted = {spec.device: spec for spec in resolved}
        serving = {worker.device for worker in self._workers}
        added: list[str] = []
        removed: list[str] = []
        for worker in list(self._workers):
            spec = wanted.get(worker.device)
            if spec is not None:
                self._note_drift(worker, spec)
                continue
            busy = worker.capturing_variables
            if busy:
                logger.warning(
                    "roster: %s left the DB set but %s has a capture session "
                    "open; removal deferred to the next tick",
                    worker.device,
                    ", ".join(busy),
                )
                continue
            try:
                await self._remove_worker(worker)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "roster: removing %s failed; retried next tick", worker.device
                )
                continue
            removed.append(worker.device)
        if removed:
            # A departed device may have freed the name a newcomer collided on.
            self._refused.clear()
        for device in sorted(wanted):
            if device in serving or self._refused.get(device) == wanted[device]:
                continue  # refused: retried when its rows change, not every tick
            try:
                admitted = await self._add_worker(wanted[device])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("roster: adding %s failed; retried next tick", device)
                continue
            if admitted:
                added.append(device)
        # A refused device that leaves the set is forgotten.
        self._refused = {d: s for d, s in self._refused.items() if d in wanted}
        self._drifted &= set(wanted)
        if added or removed:
            names = self.served_devices
            assert self._devices_pv is not None
            self._devices_pv.post(names)
            logger.info(
                "roster: %s (serving %d devices: %s)",
                ", ".join([f"+{d}" for d in added] + [f"-{d}" for d in removed]),
                len(names),
                ", ".join(names) or "none; idling on the instance PVs",
            )

    async def _add_worker(self, spec: DeviceSpec) -> bool:
        """Serve *spec* as startup would; ``False`` (logged once) when it is refused.

        Refused: its PV names collide with a served device's, or it cannot
        be built from its DB rows (a plugin attribute-name collision).  A
        refused device is skipped until it leaves the set or a removal
        frees a name — never rebuilt and torn down tick after tick.
        """
        assert self._loop is not None and self._provider is not None
        try:
            worker = _DeviceWorker(spec, self._loop, self._endpoint_resolver)
        except Exception as exc:  # noqa: BLE001 - a bad row refuses one device
            self._refuse(spec, f"{type(exc).__name__}: {exc}")
            return False
        try:
            entries = self._claim(worker)
        except ValueError as exc:
            await worker.stop()  # its plugin writer threads
            self._refuse(spec, str(exc))
            return False
        # Transactional: the worker joins the served set only once every PV
        # is on the air.  A registration that fails partway is rolled back
        # whole — the PVs already added, the claims, the worker's threads —
        # so the next tick sees the device as not served and retries it
        # (nothing of the attempt survives to make that unsafe), one ERROR
        # line per attempt.
        added: list[tuple[str, SharedPV]] = []
        try:
            for name, pv in entries:
                logger.debug("serving %s", name)  # the device-level line is INFO
                self._provider.add(name, pv)
                added.append((name, pv))
        except Exception as exc:  # noqa: BLE001 - rolled back, never half-served
            failed_at = entries[len(added)][0]
            await self._unregister(worker, added)
            for name, _pv in entries:
                self._owners.pop(name, None)
            logger.error(  # noqa: TRY400 — retried every tick
                "roster: %s: PV registration failed at %r (%s: %s); rolled back, "
                "retried next tick",
                spec.device,
                failed_at,
                type(exc).__name__,
                exc,
            )
            return False
        self._workers.append(worker)
        return True

    async def _unregister(
        self, worker: _DeviceWorker, entries: list[tuple[str, SharedPV]]
    ) -> None:
        """Stop *worker*, then take *entries* off the air and release their claims.

        Provider first (no new client can find a name), then the PV (the
        attached clients are let go).  Each step is idempotent and
        shielded, so a retried removal — or a rollback — always completes.
        """
        assert self._provider is not None
        await worker.stop()
        for name, pv in entries:
            self._owners.pop(name, None)
            with contextlib.suppress(Exception):
                self._provider.remove(name)
            with contextlib.suppress(Exception):
                pv.close(destroy=True)
            logger.debug("no longer serving %s", name)

    async def _remove_worker(self, worker: _DeviceWorker) -> None:
        """Release the worker's subscriptions, then take its PVs off the air."""
        await self._unregister(
            worker, [(name, pv) for name, _var, pv in worker.provider_entries()]
        )
        self._workers.remove(worker)

    def _refuse(self, spec: DeviceSpec, reason: str) -> None:
        """Mark *spec* refused; log once while those rows stand (edited rows are retried)."""
        if self._refused.get(spec.device) != spec:
            self._refused[spec.device] = spec
            logger.error("roster: refusing %s: %s", spec.device, reason)

    def _note_drift(self, worker: _DeviceWorker, spec: DeviceSpec) -> None:
        """Log once when a served device's shape (stream variables, scalars) changed in the DB.

        The scalars matter for a device admitted on a tick where the
        scalar-policy query degraded to empty: its stacks carry no
        per-frame scalars until a restart, and this is the one line that
        says so.
        """
        was = (set(worker.spec.stream_variables), set(worker.spec.scalar_variables))
        now = (set(spec.stream_variables), set(spec.scalar_variables))
        # An empty scalar answer is as often the scalar-policy query degrading
        # on a blip (every device at once) as a real change: ambiguous, so not
        # logged — a non-empty answer is the DB's word.
        changed = was[0] != now[0] or (bool(now[1]) and was[1] != now[1])
        if not changed:
            self._drifted.discard(worker.device)
        elif worker.device not in self._drifted:
            self._drifted.add(worker.device)
            logger.warning(
                "roster: %s's served shape changed in the DB (stream %s -> %s; "
                "scalars %s -> %s); serving the old shape until a restart "
                "re-shapes it",
                worker.device,
                sorted(was[0]),
                sorted(now[0]),
                sorted(was[1]),
                sorted(now[1]),
            )
