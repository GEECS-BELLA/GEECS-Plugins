"""PVA frame source: the p4p seam of the capture daemon.

``FrameSource`` is the test seam — capture logic (`daemon.py`) never touches
p4p directly, so hermetic tests drive a fake source. ``P4pFrameSource`` is
the production implementation: one deep-queue monitor per camera against the
GeecsPvaGateway fleet.

Empirical basis (Phase-0 probes, 2026-08-27): the unmodified gateway
delivers every frame at 1 Hz even to default-queue clients; the deep
``record[queueSize=N]`` request is burst margin, not a correctness
requirement at current rates.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from .discovery import CameraTarget

logger = logging.getLogger(__name__)

# device, frame, acq_timestamp (Unix s), recv_timestamp (Unix s)
FrameCallback = Callable[[str, "np.ndarray", float, float], None]
# device, connected (False on a Disconnected event)
ConnectionCallback = Callable[[str, bool], None]


class FrameSource(Protocol):
    """Something that can stream camera frames to a callback."""

    def subscribe(
        self,
        targets: list["CameraTarget"],
        on_frame: FrameCallback,
        on_connection: ConnectionCallback,
    ) -> None:
        """Open one monitor per target; callbacks may fire from any thread."""
        ...

    def close(self) -> None:
        """Tear down every monitor (idempotent)."""
        ...


class P4pFrameSource:
    """Deep-queue p4p monitors against the distributed PVA gateway fleet."""

    def __init__(self, *, queue_size: int = 100) -> None:
        self._queue_size = queue_size
        self._ctx: Any = None  # p4p.client.thread.Context; Any: p4p is lazy
        self._subs: list[Any] = []  # p4p Subscription handles
        self._lock = threading.Lock()

    def subscribe(
        self,
        targets: list["CameraTarget"],
        on_frame: FrameCallback,
        on_connection: ConnectionCallback,
    ) -> None:
        """Open monitors for *targets*, building the address list from them.

        ``EPICS_PVA_ADDR_LIST`` is composed from the targets' camera-server
        IPs and must be in place before the p4p Context exists — the fleet
        spans subnets, so broadcast search cannot find it.
        """
        with self._lock:
            if self._ctx is not None:
                raise RuntimeError("P4pFrameSource is already subscribed")
            addr_list = " ".join(sorted({t.server_ip for t in targets}))
            os.environ["EPICS_PVA_ADDR_LIST"] = addr_list
            os.environ["EPICS_PVA_AUTO_ADDR_LIST"] = "NO"
            from p4p.client.thread import Context

            self._ctx = Context("pva")
            request = f"record[queueSize={self._queue_size}]field()"
            for target in targets:
                handler = _MonitorHandler(target.device, on_frame, on_connection)
                self._subs.append(
                    self._ctx.monitor(
                        target.pv,
                        handler,
                        request=request,
                        notify_disconnect=True,
                    )
                )
            logger.info(
                "PVA source: %d monitors (queueSize=%d, addr_list=%s)",
                len(self._subs),
                self._queue_size,
                addr_list,
            )

    def close(self) -> None:
        """Close all monitors and the context (idempotent)."""
        with self._lock:
            for sub in self._subs:
                try:
                    sub.close()
                except Exception:  # noqa: BLE001 - best-effort shutdown
                    logger.warning("PVA subscription close failed", exc_info=True)
            self._subs.clear()
            if self._ctx is not None:
                self._ctx.close()
                self._ctx = None


class _MonitorHandler:
    """Per-camera p4p monitor callback: unwrap, timestamp, forward."""

    def __init__(
        self, device: str, on_frame: FrameCallback, on_connection: ConnectionCallback
    ) -> None:
        self._device = device
        self._on_frame = on_frame
        self._on_connection = on_connection

    def __call__(self, value: Any) -> None:  # p4p ntndarray or Disconnected
        """Handle one monitor delivery on a p4p worker thread."""
        import time

        if isinstance(value, Exception):
            # Includes the initial not-yet-connected Disconnected event.
            self._on_connection(self._device, False)
            return
        ts = getattr(value, "timestamp", None)
        if ts is None:
            logger.warning("%s: frame without timestamp dropped", self._device)
            return
        self._on_frame(self._device, value, float(ts), time.time())
