"""The capture daemon: scan-gated central image capture over PVA.

One ``CaptureDaemon`` consumes the worker's Bluesky document stream and, for
every run whose start document carries ``nonscalar_save_paths``, opens a
``ScanCaptureSession``: PVA subscriptions on the run's capture-eligible
cameras (scan-gated per the design decision — the gateway's gate opens on
first subscriber), an ``acq_timestamp`` dedupe + stale-window filter, and a
single writer thread trail-flushing one frame-stack file per device into the
engine-created ``scans/ScanNNN/<device>/`` directories.

Design constraints this module enforces (scope doc,
``Planning/data_capture/01_central_pva_capture_scope.md``):

- The daemon NEVER creates scan folders or device directories — a missing
  directory skips that device loudly (cross-package invariant).
- Frames dedupe on ``acq_timestamp`` (the device re-pushes its last frame
  with an unchanged timestamp when idle) and frames older than the run
  start are the gateway's cached pre-scan frame — skipped, counted.
- Writing happens on ONE thread (h5py is not thread-safe); PVA callbacks
  enqueue into a bounded queue whose overflow is counted, never silent.
- Phase-1 dual-write doctrine: the LV per-shot file save stays ON; this
  daemon runs alongside and its files are the diff surface, so a daemon
  failure can never lose data.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .discovery import CameraTarget
from .subscriber import FrameSource
from .writer import FrameStackWriter, Hdf5StackWriter

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

Document = Mapping[str, object]

# Frames stamped earlier than run start minus this margin are the gateway's
# cached pre-scan frame (or clock skew); the t0-sync doctrine bounds
# inter-machine skew well under this.
STALE_MARGIN_S = 2.0

# At 1 Hz the queue never holds more than a few frames; the bound exists so
# a wedged writer surfaces as a counted drop, not unbounded RAM growth.
WRITER_QUEUE_MAX = 256

WriterFactory = Callable[..., FrameStackWriter]


@dataclass
class _DeviceCapture:
    """Per-device state within one scan."""

    writer: FrameStackWriter
    seen_ts: set[float] = field(default_factory=set)
    received: int = 0
    written: int = 0
    duplicates_dropped: int = 0
    stale_skipped: int = 0
    shape_errors: int = 0
    disconnect_events: int = 0


class ScanCaptureSession:
    """Capture for one run: subscriptions, dedupe, one writer thread."""

    def __init__(
        self,
        *,
        run_uid: str,
        experiment: str,
        scan_number: int | None,
        start_time: float,
        targets: list[CameraTarget],
        save_paths: Mapping[str, str],
        source: FrameSource,
        writer_factory: WriterFactory = Hdf5StackWriter,
    ) -> None:
        self.run_uid = run_uid
        self._start_time = start_time
        self._devices: dict[str, _DeviceCapture] = {}
        self._queue: queue.Queue[tuple[str, "np.ndarray", float, float] | None] = (
            queue.Queue(maxsize=WRITER_QUEUE_MAX)
        )
        self._queue_drops = 0
        self._lock = threading.Lock()
        self._source = source

        captured: list[CameraTarget] = []
        for target in targets:
            save_path = save_paths.get(target.device)
            if save_path is None:
                continue
            try:
                writer = writer_factory(
                    Path(save_path),
                    device=target.device,
                    experiment=experiment,
                    scan_number=scan_number,
                    source_pv=target.pv,
                )
            except FileNotFoundError as exc:
                # Never create the dir — the engine owns it. Skip loudly.
                logger.error("capture skipping %s: %s", target.device, exc)
                continue
            self._devices[target.device] = _DeviceCapture(writer=writer)
            captured.append(target)

        self._writer_thread = threading.Thread(
            target=self._drain, name="capture-writer", daemon=True
        )
        self._writer_thread.start()
        if captured:
            self._source.subscribe(captured, self._on_frame, self._on_connection)
        logger.info(
            "capture session %s: scan %s, %d/%d cameras (%s)",
            run_uid[:8],
            scan_number,
            len(captured),
            len(targets),
            ", ".join(t.device for t in captured) or "none",
        )

    # -- callbacks (p4p threads) -------------------------------------------

    def _on_frame(
        self, device: str, frame: "np.ndarray", acq_ts: float, recv_ts: float
    ) -> None:
        dev = self._devices.get(device)
        if dev is None:
            return
        with self._lock:
            dev.received += 1
            if acq_ts < self._start_time - STALE_MARGIN_S:
                dev.stale_skipped += 1
                return
            if acq_ts in dev.seen_ts:
                dev.duplicates_dropped += 1
                return
            dev.seen_ts.add(acq_ts)
        try:
            self._queue.put_nowait((device, frame, acq_ts, recv_ts))
        except queue.Full:
            with self._lock:
                self._queue_drops += 1
                dev.seen_ts.discard(acq_ts)  # not written — keep books honest
            logger.error("capture writer queue full — dropped %s frame", device)

    def _on_connection(self, device: str, connected: bool) -> None:
        dev = self._devices.get(device)
        if dev is not None and not connected:
            with self._lock:
                dev.disconnect_events += 1

    # -- writer thread ------------------------------------------------------

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            device, frame, acq_ts, recv_ts = item
            dev = self._devices[device]
            try:
                dev.writer.append(frame, acq_ts, recv_ts)
                dev.written += 1
            except ValueError:
                dev.shape_errors += 1
                logger.warning("capture %s: frame shape mismatch dropped", device)
            except Exception:  # noqa: BLE001 - one bad frame must not kill the scan
                logger.exception("capture %s: writer append failed", device)

    # -- lifecycle ----------------------------------------------------------

    def close(self, *, finalized: bool) -> dict[str, dict[str, int]]:
        """Unsubscribe, drain, finalize (or abort) writers; return counters."""
        self._source.close()
        self._queue.put(None)
        self._writer_thread.join(timeout=30.0)
        if self._writer_thread.is_alive():
            logger.error("capture writer thread did not drain within 30 s")
        summary: dict[str, dict[str, int]] = {}
        for device, dev in self._devices.items():
            counters = {
                "frames_written": dev.written,
                "frames_received": dev.received,
                "duplicates_dropped": dev.duplicates_dropped,
                "stale_skipped": dev.stale_skipped,
                "shape_errors": dev.shape_errors,
                "writer_queue_drops": self._queue_drops,
                "disconnect_events": dev.disconnect_events,
            }
            try:
                if finalized:
                    dev.writer.finalize(counters)
                else:
                    dev.writer.abort()
            except Exception:  # noqa: BLE001 - close every writer regardless
                logger.exception("capture %s: writer close failed", device)
            summary[device] = counters
        return summary


class CaptureDaemon:
    """Document-stream consumer that opens one capture session per run.

    Best-effort like ``SFileExportCallback``: a failure is logged and never
    raised back into the dispatcher. One session runs at a time (the
    queueserver serializes runs); an unexpected second start closes the
    first session un-finalized rather than interleaving.
    """

    def __init__(
        self,
        *,
        experiment: str,
        targets: list[CameraTarget],
        source_factory: Callable[[], FrameSource],
        writer_factory: WriterFactory = Hdf5StackWriter,
    ) -> None:
        self._experiment = experiment
        self._targets = targets
        self._source_factory = source_factory
        self._writer_factory = writer_factory
        self._session: ScanCaptureSession | None = None

    def __call__(self, name: str, doc: Document) -> None:
        """Handle one document from the stream."""
        try:
            if name == "start":
                self._on_start(doc)
            elif name == "stop":
                self._on_stop(doc)
        except Exception:
            logger.exception("capture daemon failed handling %s document", name)

    def _on_start(self, doc: Document) -> None:
        save_paths = doc.get("nonscalar_save_paths")
        if not isinstance(save_paths, Mapping) or not save_paths:
            return
        uid = doc.get("uid")
        if not isinstance(uid, str):
            return
        if self._session is not None:
            logger.error(
                "start %s arrived with session %s open — closing un-finalized",
                uid[:8],
                self._session.run_uid[:8],
            )
            self._session.close(finalized=False)
            self._session = None
        scan_number = doc.get("scan_number")
        start_time = doc.get("time")
        self._session = ScanCaptureSession(
            run_uid=uid,
            experiment=str(doc.get("experiment") or self._experiment),
            scan_number=scan_number if isinstance(scan_number, int) else None,
            start_time=float(start_time)
            if isinstance(start_time, (int, float))
            else time.time(),
            targets=self._targets,
            save_paths={str(k): str(v) for k, v in save_paths.items()},
            source=self._source_factory(),
            writer_factory=self._writer_factory,
        )

    def _on_stop(self, doc: Document) -> None:
        session = self._session
        if session is None:
            return
        run_start = doc.get("run_start")
        if isinstance(run_start, str) and run_start != session.run_uid:
            logger.warning(
                "stop for %s does not match open session %s — closing anyway",
                run_start[:8],
                session.run_uid[:8],
            )
        summary = session.close(finalized=True)
        self._session = None
        for device, counters in sorted(summary.items()):
            logger.info(
                "capture reconciliation %s: written=%d received=%d dup=%d "
                "stale=%d shape_err=%d q_drops=%d disconnects=%d",
                device,
                counters["frames_written"],
                counters["frames_received"],
                counters["duplicates_dropped"],
                counters["stale_skipped"],
                counters["shape_errors"],
                counters["writer_queue_drops"],
                counters["disconnect_events"],
            )

    def shutdown(self) -> None:
        """Close any open session un-finalized (daemon exit)."""
        if self._session is not None:
            self._session.close(finalized=False)
            self._session = None
