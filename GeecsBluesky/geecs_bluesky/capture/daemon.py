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

- The daemon NEVER creates scan folders or device directories — and because
  the engine creates the device dirs *after* the start document (the
  save-enable plan runs post-trigger-setup under ``defer_save_on``), writers
  are constructed **lazily on the first accepted frame**, on the writer
  thread. First frames can only arrive after save-on, so the directory
  exists by then; a still-missing directory drops that device's frames with
  a counted failure, never a ``mkdir``.
- Frames dedupe on ``acq_timestamp`` (the device re-pushes its last frame
  with an unchanged timestamp when idle) and frames older than the run
  start are the gateway's cached pre-scan frame — skipped, counted. The
  gateway's ``(1,1)`` placeholder initial post carries timestamp 0.0 and is
  therefore always stale-filtered before it could fix dataset geometry.
- Every h5py call happens on ONE thread (the writer thread), except
  finalize/abort which run strictly after that thread has been joined; if
  the join times out (wedged writer — e.g. a NAS hang inside a flush), the
  files are left un-finalized and untouched rather than contended from a
  second thread.
- Every frame is accounted: the per-device counter identity is
  ``received == written + duplicates_dropped + stale_skipped +
  shape_errors + queue_drops + late_frames + writer_create_failures +
  append_failures`` (see ``FORMAT.md``).
- Phase-1 dual-write doctrine: the LV per-shot file save stays ON; this
  daemon runs alongside and its files are the diff surface, so a daemon
  failure can never lose data.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .discovery import CameraTarget
from .subscriber import FrameSource
from .writer import FrameStackWriter, Hdf5StackWriter

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

Document = Mapping[str, object]

# Frames stamped earlier than run start minus this margin are the gateway's
# cached pre-scan frame (or clock skew); the t0-sync doctrine bounds
# inter-machine skew well under this. Caveats documented in FORMAT.md: a
# viewer-held gate can keep the pre-scan cache <2 s old, and a camera-server
# clock lagging by >2 s would drop real first frames — both show up as
# attributable counter entries, not silent loss.
STALE_MARGIN_S = 2.0

# At 1 Hz the queue never holds more than a few frames; the bound exists so
# a wedged writer surfaces as a counted drop, not unbounded RAM growth.
WRITER_QUEUE_MAX = 256

WriterFactory = Callable[..., FrameStackWriter]


@dataclass
class _DeviceCapture:
    """Per-device state within one scan."""

    target: CameraTarget
    save_dir: Path
    writer: FrameStackWriter | None = None
    writer_failure_logged: bool = False
    seen_ts: set[float] = field(default_factory=set)
    received: int = 0
    written: int = 0
    duplicates_dropped: int = 0
    stale_skipped: int = 0
    shape_errors: int = 0
    queue_drops: int = 0
    late_frames: int = 0
    writer_create_failures: int = 0
    append_failures: int = 0
    disconnect_events: int = 0
    initial_disconnect_absorbed: bool = False

    def counters(self) -> dict[str, int]:
        """Snapshot the reconciliation counters (call with the session lock held)."""
        return {
            "frames_received": self.received,
            "frames_written": self.written,
            "duplicates_dropped": self.duplicates_dropped,
            "stale_skipped": self.stale_skipped,
            "shape_errors": self.shape_errors,
            "queue_drops": self.queue_drops,
            "late_frames": self.late_frames,
            "writer_create_failures": self.writer_create_failures,
            "append_failures": self.append_failures,
            "disconnect_events": self.disconnect_events,
        }


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
        self._experiment = experiment
        self._scan_number = scan_number
        self._start_time = start_time
        self._writer_factory = writer_factory
        self._devices: dict[str, _DeviceCapture] = {}
        self._queue: queue.Queue[tuple[str, "np.ndarray", float, float] | None] = (
            queue.Queue(maxsize=WRITER_QUEUE_MAX)
        )
        self._closing = threading.Event()
        self._lock = threading.Lock()
        self._source = source

        for target in targets:
            save_path = save_paths.get(target.device)
            if save_path is None:
                continue
            # No writer construction here: at start-doc time the engine has
            # not created the device dirs yet (save-enable runs later).
            self._devices[target.device] = _DeviceCapture(
                target=target, save_dir=Path(save_path)
            )

        self._writer_thread = threading.Thread(
            target=self._drain, name="capture-writer", daemon=True
        )
        self._writer_thread.start()
        if self._devices:
            try:
                self._source.subscribe(
                    [dev.target for dev in self._devices.values()],
                    self._on_frame,
                    self._on_connection,
                )
            except Exception:
                # Fail the session cleanly: stop the writer thread first so
                # nothing leaks, then let the daemon's catch log it.
                self._stop_writer_thread()
                raise
        logger.info(
            "capture session %s: scan %s, %d camera(s): %s",
            run_uid[:8],
            scan_number,
            len(self._devices),
            ", ".join(sorted(self._devices)) or "none",
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
                dev.queue_drops += 1
                dev.seen_ts.discard(acq_ts)  # not written — keep books honest
            logger.error("capture writer queue full — dropped %s frame", device)

    def _on_connection(self, device: str, connected: bool) -> None:
        dev = self._devices.get(device)
        if dev is None or connected:
            return
        with self._lock:
            # p4p's notify_disconnect delivers one initial Disconnected at
            # subscribe time on every healthy monitor — absorb it so
            # disconnect_events counts real losses only (FORMAT.md).
            if not dev.initial_disconnect_absorbed and dev.received == 0:
                dev.initial_disconnect_absorbed = True
                return
            dev.disconnect_events += 1

    # -- writer thread (the ONLY thread that touches h5py while running) ----

    def _drain(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._closing.is_set():
                    return
                continue
            if item is None:
                return
            device, frame, acq_ts, recv_ts = item
            dev = self._devices[device]
            if dev.writer is None and not self._create_writer(dev, acq_ts):
                continue
            try:
                dev.writer.append(frame, acq_ts, recv_ts)  # type: ignore[union-attr]
                with self._lock:
                    dev.written += 1
            except ValueError:
                with self._lock:
                    dev.shape_errors += 1
                logger.warning("capture %s: frame shape mismatch dropped", device)
            except Exception:  # noqa: BLE001 - one bad frame must not kill the scan
                with self._lock:
                    dev.append_failures += 1
                logger.exception("capture %s: writer append failed", device)

    def _create_writer(self, dev: _DeviceCapture, acq_ts: float) -> bool:
        """Lazily build *dev*'s writer; on failure drop the frame, counted."""
        try:
            dev.writer = self._writer_factory(
                dev.save_dir,
                device=dev.target.device,
                experiment=self._experiment,
                scan_number=self._scan_number,
                source_pv=dev.target.pv,
            )
            return True
        except Exception:  # noqa: BLE001 - incl. FileNotFoundError, OSError, h5py errors
            with self._lock:
                dev.writer_create_failures += 1
                dev.seen_ts.discard(acq_ts)
            if not dev.writer_failure_logged:
                dev.writer_failure_logged = True
                logger.error(
                    "capture %s: writer creation failed (dir %s) — frames will "
                    "drop, counted, until it succeeds; the daemon never "
                    "creates directories",
                    dev.target.device,
                    dev.save_dir,
                    exc_info=True,
                )
            return False

    def _stop_writer_thread(self) -> bool:
        """Signal and join the writer thread; return True if it exited."""
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass  # the closing event below still ends the drain loop
        self._closing.set()
        self._writer_thread.join(timeout=30.0)
        return not self._writer_thread.is_alive()

    # -- lifecycle ----------------------------------------------------------

    def close(self, *, finalized: bool) -> dict[str, dict[str, int]]:
        """Unsubscribe, drain, finalize (or abort) writers; return counters."""
        self._source.close()
        drained = self._stop_writer_thread()
        if not drained:
            logger.error(
                "capture writer thread did not drain within 30 s — leaving "
                "files un-finalized and untouched (a wedged writer may still "
                "hold the HDF5 lock)"
            )
        # Frames still in the queue were never written: late deliveries after
        # unsubscribe (p4p can deliver a few in-flight events after close())
        # or residue behind a wedged writer. Count them so the books close.
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            dev = self._devices.get(item[0])
            if dev is not None:
                with self._lock:
                    dev.late_frames += 1
        summary: dict[str, dict[str, int]] = {}
        for device, dev in self._devices.items():
            with self._lock:
                counters = dev.counters()
            summary[device] = counters
            if dev.writer is None or not drained:
                continue  # no file, or unsafe to touch it from this thread
            try:
                if finalized:
                    dev.writer.finalize(counters)
                else:
                    dev.writer.abort()
            except Exception:  # noqa: BLE001 - close every writer regardless
                logger.exception("capture %s: writer close failed", device)
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
        # Prefer the engine's explicit capture list (md["capture_devices"] +
        # scan_folder — present since the native_image_save toggle landed;
        # with native saving off, captured cameras leave
        # nonscalar_save_paths entirely). Fall back to the LV-saving list
        # for older engines (dual-write inference).
        save_paths: Mapping[str, object] | None = None
        capture_names = doc.get("capture_devices")
        scan_folder = doc.get("scan_folder")
        if (
            isinstance(capture_names, (list, tuple))
            and isinstance(scan_folder, str)
            and scan_folder
        ):
            save_paths = {
                str(name): str(Path(scan_folder) / str(name)) for name in capture_names
            }
        else:
            raw = doc.get("nonscalar_save_paths")
            if isinstance(raw, Mapping):
                save_paths = raw
        if not save_paths:
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
        matched = not isinstance(run_start, str) or run_start == session.run_uid
        if not matched:
            logger.warning(
                "stop for %s does not match open session %s — closing "
                "UN-finalized (never stamp the wrong run's files)",
                str(run_start)[:8],
                session.run_uid[:8],
            )
        summary = session.close(finalized=matched)
        self._session = None
        for device, counters in sorted(summary.items()):
            logger.info(
                "capture reconciliation %s: written=%d received=%d dup=%d "
                "stale=%d shape_err=%d q_drops=%d late=%d create_fail=%d "
                "append_fail=%d disconnects=%d",
                device,
                counters["frames_written"],
                counters["frames_received"],
                counters["duplicates_dropped"],
                counters["stale_skipped"],
                counters["shape_errors"],
                counters["queue_drops"],
                counters["late_frames"],
                counters["writer_create_failures"],
                counters["append_failures"],
                counters["disconnect_events"],
            )

    def shutdown(self) -> None:
        """Close any open session un-finalized (daemon exit)."""
        if self._session is not None:
            self._session.close(finalized=False)
            self._session = None
