"""The per-action Run/Preview dialog behind the Actions menu (G-actions v1).

Clicking an action-plan name in the Actions menu opens one
:class:`ActionRunDialog`: the plan's dry-run step table (the engine's
``describe_action``) plus a Run button gated by the menu's
"Enable action execution" arming switch.  The dialog is deliberately thin —
it owns no engine knowledge beyond the two callables it is constructed with,
so tests drive it with plain fakes and the window wires it to the real
:class:`~geecs_console.submission.Submitter` methods.

Both engine calls are blocking, so each runs on a short-lived daemon thread
through a dialog-owned :class:`~geecs_console.services.background.BackgroundResult`
worker (never a window-owned signal — issue #510), delivered queued back to
the GUI thread.  ``closeEvent`` disconnects the workers, so closing during a
slow describe/run returns immediately and the straggler result lands nowhere.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from geecs_console.services.background import BackgroundResult

#: The engine's dry-run step-dict keys, in table-column order (the
#: ``describe_action`` contract; ``from_plan`` is ``None`` for top-level
#: steps).  Rows appear in execution order.
STEP_KEYS = ("kind", "device", "variable", "value", "wait_s", "from_plan")

_STEP_HEADERS = ("#", "kind", "device", "variable", "value", "wait (s)", "from plan")


def _cell_text(value: object) -> str:
    """Render one step-dict value for the table (``None`` as an em-dash)."""
    return "—" if value is None else str(value)


class ActionRunDialog(QDialog):
    """Preview-and-run dialog for one named action plan.

    Parameters
    ----------
    parent : QWidget or None
        The owning window (the dialog is shown non-modally, so the caller
        must also keep a Python reference — the PySide6 GC hazard).
    name : str
        The action-plan name (dialog title; passed to both callables).
    describe : callable
        ``(name) -> list[dict]`` — the engine's blocking dry-run
        (:meth:`Submitter.describe_action`); called once at construction on
        a daemon thread.  A raised exception becomes the inline message.
    run : callable
        ``(name) -> None`` — the engine's blocking execution
        (:meth:`Submitter.run_action`); raises with an operator-readable
        message on refusal/failure (e.g. the scan-in-progress refusal).
    execution_enabled : bool
        The Actions menu's arming state at open; the window pushes later
        toggles through :meth:`set_execution_enabled`.
    report : callable
        ``(message) -> None`` — the window's status-bar/log reporter for
        the running/done/failed lines.
    request_during_scan : callable, optional
        ``(name) -> None`` — :meth:`Submitter.request_action_during_scan`,
        the during-scan path (pause → decide → run).  When a scan is
        active the Run button uses this instead of *run*; without it (or
        without *is_scanning*) the dialog only ever runs idle.
    is_scanning : callable, optional
        ``() -> bool`` — whether a scan is active *now*, so the Run button
        picks the path and shows the right label.  The window also pushes
        state changes via :meth:`set_scanning`.
    """

    def __init__(
        self,
        parent: Optional[QWidget],
        name: str,
        describe: Callable[[str], list],
        run: Callable[[str], None],
        execution_enabled: bool,
        report: Callable[[str], None],
        request_during_scan: Optional[Callable[[str], None]] = None,
        is_scanning: Optional[Callable[[], bool]] = None,
    ) -> None:
        super().__init__(parent)
        self._name = name
        self._describe = describe
        self._run = run
        self._report = report
        self._request_during_scan = request_during_scan
        self._is_scanning = is_scanning
        self._execution_enabled = bool(execution_enabled)
        self._run_in_flight = False
        #: Run stays disabled until the dry-run preview loads — a slow
        #: describe must never leave Run clickable beside an empty table
        #: (issue #575 misfire hardening).
        self._preview_loaded = False
        self._step_count = 0
        self._scanning = bool(is_scanning()) if is_scanning is not None else False

        self.setWindowTitle(f"Action: {name}")
        self._build_widgets()
        self._refresh_run_enabled()

        # Dialog-owned workers: the daemon thread emits on the worker, the
        # queued delivery paints on the GUI thread, and closeEvent
        # disconnects both so a slow engine call never blocks closing.
        self._describe_worker = BackgroundResult()
        self._describe_worker.result_ready.connect(
            self._apply_describe_result, Qt.ConnectionType.QueuedConnection
        )
        self._run_worker = BackgroundResult()
        self._run_worker.result_ready.connect(
            self._apply_run_result, Qt.ConnectionType.QueuedConnection
        )
        self._start_describe()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        """Create the name label, steps table, message line, and buttons."""
        layout = QVBoxLayout(self)

        # The action name is unmissable (issue #575: the misfire is
        # selection-adjacency — a habituated operator clicks Run beside the
        # wrong name).  Larger type, name on its own line.
        title = QLabel(f"Run action<br><b style='font-size:16px;'>{self._name}</b>")
        title.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(title)

        self.steps_table = QTableWidget(0, len(_STEP_HEADERS), self)
        self.steps_table.setHorizontalHeaderLabels(list(_STEP_HEADERS))
        self.steps_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.steps_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.steps_table.verticalHeader().setVisible(False)
        self.steps_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.steps_table)

        # Two separate lines, deliberately: the async preview result and the
        # async run outcome would otherwise race for one label, and a late
        # preview must never clobber a run failure/refusal (which has to
        # stay visible).
        self.preview_label = QLabel("Loading steps…")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.run_button = QPushButton("Run", self)
        self.run_button.clicked.connect(self._on_run_clicked)
        buttons.addWidget(self.run_button)
        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self.resize(640, 360)

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def set_execution_enabled(self, enabled: bool) -> None:
        """Apply the Actions menu's arming state (pushed by the window).

        Parameters
        ----------
        enabled : bool
            The "Enable action execution" checkbox state.
        """
        self._execution_enabled = bool(enabled)
        self._refresh_run_enabled()

    def set_scanning(self, scanning: bool) -> None:
        """Push the live scan state so Run picks the path and label.

        Parameters
        ----------
        scanning : bool
            Whether a scan is active (the window pushes lifecycle changes).
        """
        self._scanning = bool(scanning)
        self._refresh_run_enabled()

    def _during_scan(self) -> bool:
        """Whether the Run button should use the pause-scan path."""
        return self._scanning and self._request_during_scan is not None

    def _refresh_run_enabled(self) -> None:
        """Gate Run and set its label — armed, previewed, not in flight."""
        ready = (
            self._execution_enabled and self._preview_loaded and not self._run_in_flight
        )
        self.run_button.setEnabled(ready)
        # Step count on the button (issue #575): "Run 'name' (N steps)".
        steps = f" ({self._step_count} step{'s' if self._step_count != 1 else ''})"
        if self._during_scan():
            self.run_button.setText(f"Pause scan & run{steps}")
        else:
            self.run_button.setText(f"Run{steps if self._preview_loaded else ''}")

        if not self._execution_enabled:
            self.run_button.setToolTip(
                "Check 'Enable action execution' in the Actions menu to arm "
                "Run (preview is always available)."
            )
        elif not self._preview_loaded:
            self.run_button.setToolTip("Waiting for the dry-run preview to load…")
        elif self._during_scan():
            self.run_button.setToolTip(
                "Pause the running scan at its next safe point, then decide "
                "whether to run this action (a pop-up asks execute / ignore / "
                "abort).  Refused if the action writes the scan's trigger."
            )
        else:
            self.run_button.setToolTip(
                "Execute this action plan through the scan engine now."
            )

    # ------------------------------------------------------------------
    # Dry-run preview (describe_action)
    # ------------------------------------------------------------------

    def _start_describe(self) -> None:
        """Fetch the dry-run step list on a daemon thread (once, at open)."""
        describe = self._describe
        name = self._name

        def fetch() -> tuple[bool, object]:
            try:
                return (True, describe(name))
            except Exception as exc:  # noqa: BLE001 — any failure is an inline message
                return (False, str(exc))

        self._describe_worker.run_async(fetch, name="console-action-describe")

    @Slot(object)
    def _apply_describe_result(self, payload: object) -> None:
        """Render the dry-run steps, or the failure inline (GUI-thread slot).

        Parameters
        ----------
        payload : tuple
            ``(ok, steps | message)`` from the describe worker.
        """
        ok, value = payload
        if not ok:
            self.preview_label.setText(f"Preview failed: {value}")
            return
        steps = list(value)
        self.steps_table.setRowCount(len(steps))
        for row, step in enumerate(steps):
            cells = [str(row + 1)] + [_cell_text(step.get(key)) for key in STEP_KEYS]
            for column, text in enumerate(cells):
                self.steps_table.setItem(row, column, QTableWidgetItem(text))
        self.steps_table.resizeColumnsToContents()
        count = len(steps)
        self.preview_label.setText(
            f"{count} step{'s' if count != 1 else ''} (dry-run preview)."
        )
        # The preview loaded — arm Run (if the menu allows) with the step
        # count now on its label.
        self._preview_loaded = True
        self._step_count = count
        self._refresh_run_enabled()

    # ------------------------------------------------------------------
    # Execution (run_action)
    # ------------------------------------------------------------------

    def _on_run_clicked(self) -> None:
        """Dispatch the action — idle ``run_action`` or the pause-scan path."""
        if (
            self._run_in_flight
            or not self._execution_enabled
            or not self._preview_loaded
        ):
            return
        self._run_in_flight = True
        self._refresh_run_enabled()
        name = self._name

        if self._during_scan():
            # During a scan: ask the engine to pause and stage the action;
            # the operator's execute/ignore/abort decision then arrives as a
            # separate three-way modal on the window.  This call returns
            # promptly (or raises a refusal).
            message = f"pausing scan to run '{name}'…"
            self._report(message)
            self.message_label.setText(message)
            request = self._request_during_scan
            done_ok = "scan pausing — decide in the pop-up"

            def call() -> tuple[bool, str]:
                try:
                    request(name)
                except Exception as exc:  # noqa: BLE001 — refusals are messages
                    return (False, str(exc))
                return (True, done_ok)

        else:
            message = f"running action '{name}'…"
            self._report(message)
            self.message_label.setText(message)
            run = self._run

            def call() -> tuple[bool, str]:
                try:
                    run(name)
                except Exception as exc:  # noqa: BLE001 — refusals are messages
                    return (False, str(exc))
                return (True, f"action '{name}' done")

        self._run_worker.run_async(call, name="console-action-run")

    @Slot(object)
    def _apply_run_result(self, payload: object) -> None:
        """Report one finished run and re-arm the button (GUI-thread slot).

        Success shows "action '<name>' done"; a failure or refusal (e.g.
        the engine's "scan in progress — action not started") shows the
        raised message — in the status bar *and* inline, so a refusal is
        never silent.

        Parameters
        ----------
        payload : tuple
            ``(ok, message)`` from the run worker.
        """
        _ok, message = payload
        self._run_in_flight = False
        self._report(message)
        self.message_label.setText(message)
        self._refresh_run_enabled()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt override
        """Disconnect the workers so a straggling engine call lands nowhere.

        Never joins the daemon threads — closing during a slow describe or
        run must return immediately.
        """
        for worker, slot in (
            (self._describe_worker, self._apply_describe_result),
            (self._run_worker, self._apply_run_result),
        ):
            try:
                worker.result_ready.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        super().closeEvent(event)
