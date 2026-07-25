"""The PVA gateway: GEECS camera frames in, NTNDArray PVs out.

One process serves every camera in its config: per camera a gated GEECS TCP
subscription (started on the first PVA client, stopped on the last, so an
unwatched camera costs the LabVIEW device nothing), IMAQ decode off the event
loop, and latest-wins posting (a stalled consumer drops stale frames, never
backlogs). Instance identity PVs (`version`, `heartbeat`) support fleet
monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from importlib.metadata import PackageNotFoundError, version as _dist_version

import numpy as np
from p4p.nt import NTNDArray, NTScalar
from p4p.server import Server
from p4p.server.thread import SharedPV

from geecs_ca_gateway.pv_naming import normalize_component, pv_name
from geecs_ca_gateway.transport.tcp_subscriber import GeecsTcpSubscriber
from geecs_data_utils.io import decode_imaq_image_string

from geecs_pva_gateway.config import CameraSpec, PvaGatewayConfig

logger = logging.getLogger(__name__)

# LabVIEW epoch (1904) -> Unix epoch (1970), same ladder as the CA gateway.
_LABVIEW_EPOCH_OFFSET = 2_082_844_800
_TIMESTAMP_VARS = ("acq_timestamp", "systimestamp")
_RECONNECT_MIN_S = 0.5
_RECONNECT_MAX_S = 30.0
_HEARTBEAT_PERIOD_S = 5.0

try:
    __version__ = _dist_version("geecs-pva-gateway")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+source"


def _frame_timestamp(update: dict) -> float:
    """Frame time from the GEECS timestamp ladder, else receive time."""
    for var in _TIMESTAMP_VARS:
        value = update.get(var)
        if isinstance(value, (int, float)) and value > 0:
            return float(value) - _LABVIEW_EPOCH_OFFSET
    return time.time()


class _Gate:
    """p4p handler that refcounts client connections into the worker."""

    def __init__(self, worker: "_CameraWorker") -> None:
        self._worker = worker

    def onFirstConnect(self, pv: SharedPV) -> None:  # noqa: N802 (p4p API)
        self._worker.retain()

    def onLastDisconnect(self, pv: SharedPV) -> None:  # noqa: N802 (p4p API)
        self._worker.release()


class _CameraWorker:
    """One camera device: N image PVs, one gated + supervised subscription."""

    def __init__(self, spec: CameraSpec, loop: asyncio.AbstractEventLoop) -> None:
        self._spec = spec
        self._loop = loop
        self._pvs: dict[str, SharedPV] = {
            var: SharedPV(
                handler=_Gate(self),
                nt=NTNDArray(),
                initial=np.zeros((1, 1), dtype=np.uint16),
            )
            for var in spec.image_variables
        }
        self._clients = 0
        self._supervisor: asyncio.Task | None = None
        self._latest: dict[str, tuple[str, float]] = {}
        self._publishing: set[str] = set()

    def providers(self) -> dict[str, SharedPV]:
        """``{pv_name: SharedPV}`` for this camera's image variables."""
        return {self._spec.pv_name_for(var): pv for var, pv in self._pvs.items()}

    async def stop(self) -> None:
        """Cancel the supervisor (used at gateway shutdown)."""
        if self._supervisor is not None:
            self._supervisor.cancel()
            try:
                await self._supervisor
            except asyncio.CancelledError:
                pass
            self._supervisor = None

    # -- gating; retain/release arrive on p4p worker threads ---------------

    def retain(self) -> None:
        self._loop.call_soon_threadsafe(self._retain)

    def release(self) -> None:
        self._loop.call_soon_threadsafe(self._release)

    def _retain(self) -> None:
        self._clients += 1
        if self._clients == 1 and (self._supervisor is None or self._supervisor.done()):
            logger.info("first client for %s: subscribing", self._spec.device)
            self._supervisor = self._loop.create_task(
                self._run(), name=f"camera[{self._spec.device}]"
            )

    def _release(self) -> None:
        self._clients = max(0, self._clients - 1)
        if self._clients == 0 and self._supervisor is not None:
            logger.info("last client for %s: unsubscribing", self._spec.device)
            self._supervisor.cancel()
            self._supervisor = None

    # -- subscription supervisor -------------------------------------------

    async def _run(self) -> None:
        """Keep the GEECS subscription alive; reconnect with backoff on drops."""
        backoff = _RECONNECT_MIN_S
        while True:
            subscriber = GeecsTcpSubscriber(self._spec.host, self._spec.port)
            try:
                await subscriber.connect()
                await subscriber.subscribe(
                    list(self._spec.image_variables) + list(_TIMESTAMP_VARS),
                    self._on_frame,
                    text_variables=set(self._spec.image_variables),
                )
                backoff = _RECONNECT_MIN_S
                await subscriber.wait_disconnected()
                logger.warning(
                    "subscription to %s (%s:%s) dropped; reconnecting",
                    self._spec.device,
                    self._spec.host,
                    self._spec.port,
                )
            except asyncio.CancelledError:
                await subscriber.close()
                raise
            except Exception:
                logger.warning(
                    "connect/subscribe to %s (%s:%s) failed; retry in %.1fs",
                    self._spec.device,
                    self._spec.host,
                    self._spec.port,
                    backoff,
                    exc_info=True,
                )
            await subscriber.close()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_S)

    # -- frame pipeline ----------------------------------------------------

    def _on_frame(self, update: dict) -> None:
        """Push-frame callback (event loop): stash latest, schedule publish."""
        ts = _frame_timestamp(update)
        for var in self._spec.image_variables:
            blob = update.get(var)
            if not blob or not isinstance(blob, str):
                continue
            # Latest-wins slot: an unconsumed frame is replaced, never queued.
            self._latest[var] = (blob, ts)
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
                except ValueError:
                    logger.warning(
                        "decode failed for %s %s (%d bytes)",
                        self._spec.device,
                        var,
                        len(blob),
                        exc_info=True,
                    )
                    continue
                self._pvs[var].post(image, timestamp=ts)
        finally:
            self._publishing.discard(var)


class GeecsPvaGateway:
    """Serve a :class:`PvaGatewayConfig`'s cameras as NTNDArray PVs."""

    def __init__(self, config: PvaGatewayConfig) -> None:
        self._config = config
        self._workers: list[_CameraWorker] = []
        self._server: Server | None = None

    def conf(self) -> dict:
        """Client configuration for the running server (test isolation)."""
        assert self._server is not None
        return self._server.conf()

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

        providers: dict[str, SharedPV] = {}
        for worker in self._workers:
            providers.update(worker.providers())

        prefix = pv_name(self._config.experiment, "pvagateway", self._instance_token())
        heartbeat_pv = SharedPV(nt=NTScalar("I"), initial=0)
        providers[f"{prefix}:version"] = SharedPV(nt=NTScalar("s"), initial=__version__)
        providers[f"{prefix}:heartbeat"] = heartbeat_pv

        for name in sorted(providers):
            logger.info("serving %s", name)

        self._server = Server(providers=[providers], isolate=isolate)
        try:
            beats = 0
            while True:
                await asyncio.sleep(_HEARTBEAT_PERIOD_S)
                beats += 1
                heartbeat_pv.post(beats)
        finally:
            for worker in self._workers:
                await worker.stop()
            self._server.stop()
            self._server = None
