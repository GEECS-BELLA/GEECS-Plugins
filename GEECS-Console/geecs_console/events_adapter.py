"""Bridge the engine's ``on_event`` stream into Qt signals.

:class:`ScanEventsAdapter.handle` is the callback handed to the engine; it
runs on the scan thread and re-emits each event as a Qt signal, which Qt
delivers queued to the main thread.  Dispatch is by event class *name* and
duck-typed attributes rather than ``isinstance`` against
``geecs_bluesky.events`` — this module must import without the ``ca`` extra,
and fakes with the same class names test it hermetically.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal


class ScanEventsAdapter(QObject):
    """Turns ScanEvent objects into Qt signals for the Now panel (R6)."""

    state_changed = Signal(str)
    """The scan state pill text (a ``ScanState`` value, e.g. ``"running"``)."""

    totals_known = Signal(int)
    """Total shots for the scan (from the INITIALIZING lifecycle event)."""

    progress = Signal(int, int, int)
    """``(step_index, total_steps, shots_completed)`` from step events."""

    error = Signal(str)
    """A ``ScanErrorEvent`` message."""

    log_line = Signal(str)
    """One human-readable line per event, for the compact log tail."""

    def handle(self, event: Any) -> None:
        """Consume one engine event (the ``on_event`` callback).

        Parameters
        ----------
        event : Any
            A ``geecs_bluesky.events.ScanEvent`` (or a duck-typed fake with
            the same class name and attributes).
        """
        kind = type(event).__name__
        if kind == "ScanLifecycleEvent":
            state = getattr(event, "state", None)
            state_text = getattr(state, "value", str(state))
            self.state_changed.emit(state_text)
            total_shots = getattr(event, "total_shots", 0)
            if total_shots:
                self.totals_known.emit(int(total_shots))
            self.log_line.emit(f"scan {state_text}")
        elif kind == "ScanStepEvent":
            step_index = int(getattr(event, "step_index", 0))
            total_steps = int(getattr(event, "total_steps", 0))
            shots_completed = int(getattr(event, "shots_completed", 0))
            self.progress.emit(step_index, total_steps, shots_completed)
            phase = getattr(event, "phase", "")
            self.log_line.emit(
                f"step {step_index + 1}/{total_steps} {phase} ({shots_completed} shots)"
            )
        elif kind == "ScanErrorEvent":
            message = str(getattr(event, "message", ""))
            self.error.emit(message)
            self.log_line.emit(f"error: {message}")
        elif kind == "ScanRestoreFailedEvent":
            device = getattr(event, "device", "")
            message = getattr(event, "message", "")
            self.log_line.emit(f"restore failed: {device}: {message}")
        elif kind == "ScanDialogEvent":
            # Not yet rendered as a modal dialog (stubbed seam — see
            # CLAUDE.md); the engine's wait falls back to its default.
            request = getattr(event, "request", None)
            exc = getattr(request, "exc", None)
            self.log_line.emit(f"operator question (unanswered): {exc}")
        else:
            self.log_line.emit(kind)
