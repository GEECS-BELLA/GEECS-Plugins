"""The PVA gateway: GEECS camera frames in, NTNDArray PVs out.

One process serves every camera in its config. Gating is per image variable: a
variable's GEECS TCP subscription starts with its first PVA client and stops
with its last, so an unwatched variable costs the LabVIEW device nothing —
each watched variable holds its own subscription connection. Decode runs off
the event loop; delivery is latest-wins (a stalled consumer drops stale
frames, never backlogs). Instance identity PVs (`version`, `heartbeat`)
support fleet monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time

import numpy as np
from p4p.nt import NTNDArray, NTScalar
from p4p.server import Server
from p4p.server.thread import SharedPV

from geecs_core.pv_naming import normalize_component, pv_name
from geecs_core.transport.tcp_subscriber import GeecsTcpSubscriber
from geecs_data_utils.io import decode_imaq_image_string

from geecs_pva_gateway import __version__
from geecs_pva_gateway.config import CameraSpec, PvaGatewayConfig

logger = logging.getLogger(__name__)

# LabVIEW epoch (1904) -> Unix epoch (1970), same ladder as the CA gateway.
_LABVIEW_EPOCH_OFFSET = 2_082_844_800
_TIMESTAMP_VARS = ("acq_timestamp", "systimestamp")
_RECONNECT_MIN_S = 0.5
_RECONNECT_MAX_S = 30.0
_HEARTBEAT_PERIOD_S = 5.0

# Process exit code meaning "restart requested via the :restart PV" — the
# service manager relaunches, which re-resolves DB config and (with the
# pull-on-restart launcher) reinstalls from the source clone. Mirrors the
# CA gateway's CAGateway:RESTART -> exit 86 -> systemd relaunch pattern.
RESTART_EXIT_CODE = 86


def _frame_timestamp(update: dict) -> float:
    """Frame time from the GEECS timestamp ladder, else receive time."""
    for var in _TIMESTAMP_VARS:
        value = update.get(var)
        if isinstance(value, (int, float)) and value > 0:
            return float(value) - _LABVIEW_EPOCH_OFFSET
    return time.time()


class _Gate:
    """p4p handler that refcounts one variable's client connections."""

    def __init__(self, worker: "_CameraWorker", var: str) -> None:
        self._worker = worker
        self._var = var

    def onFirstConnect(self, pv: SharedPV) -> None:  # noqa: N802 (p4p API)
        self._worker.retain(self._var)

    def onLastDisconnect(self, pv: SharedPV) -> None:  # noqa: N802 (p4p API)
        self._worker.release(self._var)


class _RestartHandler:
    """p4p handler for the writable :restart PV — any put requests restart."""

    def __init__(self, on_restart) -> None:
        self._on_restart = on_restart

    def put(self, pv: SharedPV, op) -> None:
        logger.warning("restart requested via :restart PV")
        op.done()
        self._on_restart()


class _CameraWorker:
    """One camera device: per-variable PVs, gated + supervised subscriptions."""

    def __init__(self, spec: CameraSpec, loop: asyncio.AbstractEventLoop) -> None:
        self._spec = spec
        self._loop = loop
        self._pvs: dict[str, SharedPV] = {
            var: SharedPV(
                handler=_Gate(self, var),
                nt=NTNDArray(),
                initial=np.zeros((1, 1), dtype=np.uint16),
            )
            for var in spec.image_variables
        }
        self._clients: dict[str, int] = dict.fromkeys(spec.image_variables, 0)
        self._supervisors: dict[str, asyncio.Task] = {}
        self._latest: dict[str, tuple[str, float]] = {}
        self._publishing: set[str] = set()
        self._stopping = False

    @property
    def device(self) -> str:
        """The GEECS device name this worker serves."""
        return self._spec.device

    def provider_entries(self) -> list[tuple[str, str, SharedPV]]:
        """``[(pv_name, variable, SharedPV), ...]`` — one row per variable.

        A list, not a dict: two variables of this camera that normalize to the
        same PV name must both surface so the collision guard can see them
        rather than one silently shadowing the other.
        """
        return [(self._spec.pv_name_for(var), var, pv) for var, pv in self._pvs.items()]

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
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

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

    async def _run(self, var: str) -> None:
        """Keep one variable's subscription alive; reconnect on socket drops."""
        backoff = _RECONNECT_MIN_S
        while True:
            subscriber = GeecsTcpSubscriber(self._spec.host, self._spec.port)
            try:
                await subscriber.connect()
                await subscriber.subscribe(
                    [var, *_TIMESTAMP_VARS],
                    lambda update: self._on_frame(var, update),
                    text_variables={var},
                )
                backoff = _RECONNECT_MIN_S
                await subscriber.wait_disconnected()
                logger.warning(
                    "subscription to %s %s (%s:%s) dropped; reconnecting",
                    self._spec.device,
                    var,
                    self._spec.host,
                    self._spec.port,
                )
            except asyncio.CancelledError:
                await subscriber.close()
                raise
            except Exception:
                logger.warning(
                    "connect/subscribe to %s %s (%s:%s) failed; retry in %.1fs",
                    self._spec.device,
                    var,
                    self._spec.host,
                    self._spec.port,
                    backoff,
                    exc_info=True,
                )
            await subscriber.close()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_S)

    # -- frame pipeline ----------------------------------------------------

    def _on_frame(self, var: str, update: dict) -> None:
        """Push-frame callback (event loop): stash latest, schedule publish."""
        blob = update.get(var)
        if not blob or not isinstance(blob, str):
            return
        # Latest-wins slot: an unconsumed frame is replaced, never queued.
        self._latest[var] = (blob, _frame_timestamp(update))
        if var not in self._publishing:
            self._publishing.add(var)
            self._loop.create_task(self._publish(var))

    async def _publish(self, var: str) -> None:
        try:
            while (item := self._latest.pop(var, None)) is not None:
                blob, ts = item
                try:
                    image = await self._loop.run_in_executor(
                        None, decode_imaq_image_string, blob
                    )
                    self._pvs[var].post(image, timestamp=ts)
                except Exception:
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
    """Serve a :class:`PvaGatewayConfig`'s cameras as NTNDArray PVs."""

    def __init__(self, config: PvaGatewayConfig) -> None:
        self._config = config
        self._workers: list[_CameraWorker] = []
        self._server: Server | None = None
        self._restart_event: asyncio.Event | None = None

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
            for spec in self._config.cameras
            for var in spec.image_variables
        ]

    def _instance_token(self) -> str:
        """Identity component for the instance PVs: the served host."""
        if self._config.cameras:
            return normalize_component(self._config.cameras[0].host)
        return normalize_component(socket.gethostname())

    async def run(self, *, isolate: bool = False) -> None:
        """Serve until cancelled. ``isolate`` sandboxes ports for tests."""
        loop = asyncio.get_running_loop()
        self._workers = [_CameraWorker(spec, loop) for spec in self._config.cameras]

        # PV naming is lossy (normalization), so guard against two variables
        # landing on one name — within a camera or across cameras — since
        # shadowing would be silent. Iterate per-variable entries (not a
        # per-camera dict, which would collapse the within-camera case).
        providers: dict[str, SharedPV] = {}
        owners: dict[str, tuple[str, str]] = {}
        for worker in self._workers:
            for name, var, pv in worker.provider_entries():
                source = (worker.device, var)
                if name in owners:
                    raise ValueError(
                        f"PV name collision after normalization: {name!r} from "
                        f"{source} and {owners[name]}"
                    )
                owners[name] = source
                providers[name] = pv

        self._restart_event = asyncio.Event()
        prefix = pv_name(self._config.experiment, "pvagateway", self._instance_token())
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

        for name in sorted(providers):
            logger.info("serving %s", name)

        self._server = Server(providers=[providers], isolate=isolate)
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
            for worker in self._workers:
                await worker.stop()
            self._server.stop()
            self._server = None
