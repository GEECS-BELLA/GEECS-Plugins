"""The R6 now panel: state pill, progress bar, scan number, log tail.

Owns the rendering of the console's "what is happening right now" region
so the main window stays a composition root.  The window's scan-lifecycle
hub (``_on_scan_state``) remains in the window — it fans out to R5 button
gating and dialog teardown as well — and delegates the R6 rendering here.

The log tail this panel appends to is NOT the Python log; see the R6
bullet in this package's CLAUDE.md for the canonical description of the
two streams and why their shot counts must never be reconciled.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import QObject, Qt, QTimer, Slot
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QProgressBar

from geecs_console.services.background import BackgroundResult

logger = logging.getLogger(__name__)

#: A displayed live scan number expires to "(previous)" after this long.
_SCAN_NUMBER_EXPIRY_MS = 10_000

#: Screen-map palette (see app/style.qss header; a shared palette module
#: is a recorded deferral).
_COLOR_GREY = "#b9c0c7"
_COLOR_GREEN = "#2f9e63"
_COLOR_AMBER = "#d9a21b"
_COLOR_RED = "#c4453a"

#: ScanState value → state-pill dot color (grey covers idle/unknown).
_STATE_DOT_COLORS = {
    "running": _COLOR_GREEN,
    "done": _COLOR_GREEN,
    "complete": _COLOR_GREEN,
    "completed": _COLOR_GREEN,
    "initializing": _COLOR_AMBER,
    "stopping": _COLOR_AMBER,
    "pausing": _COLOR_AMBER,
    "paused": _COLOR_AMBER,
    "aborted": _COLOR_RED,
    "error": _COLOR_RED,
    "failed": _COLOR_RED,
}


class NowPanelController(QObject):
    """Render the R6 region on behalf of the main window.

    A ``QObject`` with **no Qt parent**, kept alive only by the window's
    Python reference (``self._now``); the window's ``closeEvent`` calls
    :meth:`dispose` to sever every controller → window edge so the dead
    window is freed by refcount, not the cyclic GC (the actions-menu
    controller's documented lifetime rule).

    Parameters
    ----------
    state_pill, progress_bar, scan_number_label, log_tail : QWidget
        The R6 widgets (bound from the ``.ui`` by the window, which
        remains their attribute home for tests and tooltips).
    current_experiment : callable
        Returns the currently selected experiment name ("" for none).
    resolve_lookup : callable
        Returns the idle scan-number lookup to use for one probe —
        resolved at call time so test patching of the module-level
        default keeps working.
    """

    def __init__(
        self,
        *,
        state_pill: QLabel,
        progress_bar: QProgressBar,
        scan_number_label: QLabel,
        log_tail: QPlainTextEdit,
        current_experiment: Callable[[], str],
        resolve_lookup: Callable[[], Callable[[str], Optional[int]]],
    ) -> None:
        super().__init__()
        self._state_pill = state_pill
        self._progress_bar = progress_bar
        self._scan_number_label = scan_number_label
        self._log_tail = log_tail
        self._current_experiment = current_experiment
        self._resolve_lookup = resolve_lookup

        #: Total shots announced by the running scan (0 = not announced).
        self._total_shots = 0

        self._scan_number_timer = QTimer(self)
        self._scan_number_timer.setSingleShot(True)
        self._scan_number_timer.setInterval(_SCAN_NUMBER_EXPIRY_MS)
        self._scan_number_timer.timeout.connect(self._expire_scan_number)

        self._worker = BackgroundResult()
        self._worker.result_ready.connect(
            self._apply_idle_scan_number, Qt.ConnectionType.QueuedConnection
        )
        #: Experiment tag of an in-flight idle probe, or None — same
        #: one-per-experiment dedupe as the actions-menu fetch (startup
        #: requests twice back-to-back; concurrent daemon threads racing
        #: a lazy native first import can abort the process).
        self._probe_inflight: Optional[str] = None

    # ------------------------------------------------------------------
    # Log tail + scan number
    # ------------------------------------------------------------------

    def append_log(self, line: str) -> None:
        """Append one line to the compact log tail.

        Parameters
        ----------
        line : str
            The text to append (one log-tail row).
        """
        self._log_tail.appendPlainText(line)

    def set_scan_number(self, number: int) -> None:
        """Show the current scan number, expiring to 'previous' after 10 s.

        Parameters
        ----------
        number : int
            The claimed scan number.
        """
        self._scan_number_label.setText(f"Scan {number:03d}")
        self._scan_number_timer.start()

    def _expire_scan_number(self) -> None:
        """Mark the displayed scan number as previous once the timer fires."""
        text = self._scan_number_label.text()
        if text.startswith("Scan ") and not text.endswith("(previous)"):
            self._scan_number_label.setText(f"{text} (previous)")

    def start_idle_probe(self) -> None:
        """Peek at today's scans dir for the idle display — read-only.

        The lookup resolves today's daily ``scans/`` folder and lists it
        for the highest existing ``ScanNNN`` — resolution and listdir
        only, never creating anything on that path (repo scan-folder
        invariant).  The data root is typically a network mount, so the
        blocking lookup runs on a short-lived daemon thread (via the
        :class:`BackgroundResult` worker) and reports back queued to
        :meth:`_apply_idle_scan_number`; one probe in flight per
        experiment.
        """
        experiment = self._current_experiment()
        if self._probe_inflight == experiment:
            return
        self._probe_inflight = experiment
        lookup = self._resolve_lookup()

        def probe() -> tuple[str, Optional[int]]:
            try:
                number = lookup(experiment)
            except Exception as exc:  # noqa: BLE001 — a flaky mount is not a crash
                logger.info("idle scan-number lookup failed: %s", exc)
                number = None
            return (experiment, number)

        self._worker.run_async(probe, name="console-idle-scan-probe")

    @Slot(object)
    def _apply_idle_scan_number(self, payload: object) -> None:
        """Render the idle scan-number peek (GUI-thread slot, delivered queued).

        A live scan number on display (the 10 s expiry timer still
        running) is never clobbered, and a result tagged with an
        experiment that is no longer selected is dropped.

        Parameters
        ----------
        payload : tuple
            ``(experiment, int | None)`` from the daemon-thread probe.
        """
        experiment, number = payload
        if self._probe_inflight == experiment:
            self._probe_inflight = None
        if experiment != self._current_experiment():
            return
        if self._scan_number_timer.isActive():
            return
        if number is None:
            self._scan_number_label.setText("No scans today")
        else:
            self._scan_number_label.setText(f"Scan {number:03d} (previous)")

    # ------------------------------------------------------------------
    # State pill + progress
    # ------------------------------------------------------------------

    def set_state_pill(self, state: str) -> None:
        """Render the state pill: colored dot + uppercase state word.

        Parameters
        ----------
        state : str
            A ``ScanState`` value (e.g. ``"running"``); the dot color
            comes from ``_STATE_DOT_COLORS`` (grey covers idle/unknown).
        """
        word = (state or "idle").strip()
        color = _STATE_DOT_COLORS.get(word.lower(), _COLOR_GREY)
        self._state_pill.setText(
            f'<span style="color:{color};">●</span> {word.upper()}'
        )

    def set_totals(self, total_shots: int) -> None:
        """Size the progress bar once the scan announces its totals.

        Parameters
        ----------
        total_shots : int
            The scan's announced shot total.
        """
        self._total_shots = total_shots
        self._progress_bar.setMaximum(max(1, total_shots))
        self._progress_bar.setValue(0)

    def update_progress(
        self, step_index: int, total_steps: int, shots_completed: int
    ) -> None:
        """Advance the progress bar from one step event.

        Parameters
        ----------
        step_index : int
            Zero-based index of the current step.
        total_steps : int
            The step total ("0" when unannounced).
        shots_completed : int
            Shots completed so far.
        """
        if self._total_shots:
            self._progress_bar.setValue(min(shots_completed, self._total_shots))
        elif total_steps:
            self._progress_bar.setMaximum(total_steps)
            self._progress_bar.setValue(min(step_index + 1, total_steps))

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        """Sever every controller → window reference; idempotent.

        Called from the window's ``closeEvent``.  Stops the expiry timer,
        detaches the probe worker so a straggling result lands nowhere,
        and drops the widget/closure references so the dead window is
        freed by refcount rather than the cyclic GC.
        """
        self._scan_number_timer.stop()
        try:
            self._worker.result_ready.disconnect(self._apply_idle_scan_number)
        except (RuntimeError, TypeError):
            pass
        self._current_experiment = lambda: ""
        self._resolve_lookup = lambda: (lambda experiment: None)
