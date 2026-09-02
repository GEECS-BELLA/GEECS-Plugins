"""The R8 queue panel: what the manager holds — running, waiting, finished.

The RE Manager exposes the queue as data (``queue_get`` / ``history_get``)
and ships no operator view of it; this panel is the console's.  It
renders three things the R6 now panel deliberately does not: the items
*waiting* behind the running scan (any client's — the MCP's, a
notebook's), the recent *history* with each item's exit status, and the
one recovery verb the queue needs, **Clear queue** (the failed-item-at-
front trap: a failed plan returns to the queue front and would re-run
before the next submission — see ``qs_client``).

Refresh policy: the 1 s manager status poll the scan monitor already
runs is the trigger.  A refresh (three bounded 0MQ round trips on a
daemon thread) runs when the snapshot's queue-shaped fields change
(``items_in_queue`` / ``running_item_uid`` / ``re_state``) and, as a
fallback for edits that change nothing the status shows (a reorder by
another client), every :data:`FALLBACK_REFRESH_S`.  One fetch in flight
at a time; a fetch that fails renders as "unavailable", never as an
empty queue.

Controller rules (#534 checklist): ``QObject`` with **no Qt parent**;
injected widgets and callables, never the window; daemon threads emit
on worker-owned signals connected ``QueuedConnection``; an idempotent
:meth:`dispose` the window's ``closeEvent`` calls.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, Qt, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from geecs_console.services.background import BackgroundResult

logger = logging.getLogger(__name__)

#: Finished items shown (newest first).  The manager's history is
#: oldest-first and grows for the life of its Redis store.
HISTORY_ROWS = 10

#: Fallback refresh cadence (seconds) when the status poll shows no change.
FALLBACK_REFRESH_S = 5.0

#: Table columns, left to right.
COLUMNS = ("State", "Item", "By", "Detail")

#: Screen-map palette (see app/style.qss header; a shared palette module
#: is a recorded deferral — now_panel.py carries its own copy).  Grey is
#: the QSS ``--w-dim`` text tone, not the pill's ``--w-line`` dot grey:
#: this one colors words.
_COLOR_GREY = "#6b7681"
_COLOR_GREEN = "#2f9e63"
_COLOR_AMBER = "#d9a21b"
_COLOR_RED = "#c4453a"

#: Row state → State-cell color.  ``exit_status`` values are the
#: manager's (completed / failed / stopped / aborted / halted / unknown).
_STATE_COLORS = {
    "running": _COLOR_GREEN,
    "completed": _COLOR_GREEN,
    "queued": _COLOR_GREY,
    "stopped": _COLOR_AMBER,
    "failed": _COLOR_RED,
    "aborted": _COLOR_RED,
    "halted": _COLOR_RED,
}


@dataclass(frozen=True)
class QueueSnapshot:
    """One fetch of the manager's queue-shaped data (or why it failed).

    ``history`` is ``None`` when the fetch deliberately skipped it (the
    fallback tick re-reads the queue only — the history can change only
    when the status key does, and the manager's history is unbounded).
    """

    running: Optional[dict] = None
    queued: list[dict] = field(default_factory=list)
    history: Optional[list[dict]] = None
    error: str = ""


@dataclass(frozen=True)
class QueueRow:
    """One rendered table row."""

    state: str
    item: str
    user: str
    detail: str = ""


# ----------------------------------------------------------------------
# Pure summarizers (tested without Qt)
# ----------------------------------------------------------------------


def _fmt(value: Any) -> str:
    """Format a position number compactly (``0.5``, ``10``, ``1e-06``)."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _summarize_axis(axis: dict) -> str:
    """Render one ``ScanAxis`` dict: ``jet_x 0 → 5 step 0.5`` or ``jet_x [4 values]``."""
    variable = str(axis.get("variable") or "?")
    positions = axis.get("positions") or {}
    if isinstance(positions, dict) and "values" in positions:
        values = positions.get("values") or []
        return f"{variable} [{len(values)} values]"
    if isinstance(positions, dict) and "start" in positions:
        return (
            f"{variable} {_fmt(positions.get('start'))} → "
            f"{_fmt(positions.get('end'))} step {_fmt(positions.get('step'))}"
        )
    return variable


def _shots_per_step(request: dict) -> Optional[int]:
    """The request's shots per step (v2+ ``capture`` block, else the v1 flat key)."""
    capture = request.get("capture")
    raw = (
        capture.get("shots_per_step")
        if isinstance(capture, dict)
        else request.get("shots_per_step")
    )
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def summarize_request(request: dict) -> str:
    """One line for a ``ScanRequest`` dict as the manager stored it.

    Tolerant of every schema version the queue may hold — a history item
    outlives the console that submitted it.

    Parameters
    ----------
    request : dict
        The plan's first positional argument (the request JSON).

    Returns
    -------
    str
        e.g. ``jet_x 0 → 5 step 0.5 · 10 shots/step — "focus check"``.
    """
    mode = str(request.get("mode") or "").lower()
    axes = [a for a in (request.get("axes") or []) if isinstance(a, dict)]
    if mode == "optimize" or request.get("optimization"):
        spec = request.get("optimization") or {}
        objectives = spec.get("objectives") if isinstance(spec, dict) else None
        names = ", ".join(objectives) if isinstance(objectives, dict) else ""
        head = f"Optimize {names}".rstrip()
    elif axes:
        head = " × ".join(_summarize_axis(a) for a in axes)
    else:
        head = "No-scan"
    parts = [head]
    shots = _shots_per_step(request)
    if shots is not None:
        parts.append(f"{shots} shots" if head == "No-scan" else f"{shots} shots/step")
    if request.get("background"):
        parts.insert(0, "Background")
    line = " · ".join(parts)
    description = str(request.get("description") or "").strip()
    if description:
        line = f'{line} — "{description}"'
    return line


def summarize_item(item: dict) -> str:
    """One line for a manager queue/history item.

    Parameters
    ----------
    item : dict
        The manager's item shape (``name`` / ``args`` / ``kwargs`` /
        ``item_type`` / ``user`` / ``item_uid`` [/ ``result``]).

    Returns
    -------
    str
        The GEECS plans render by their arguments (a scan by its request,
        an action by its name); anything else by its plan name.
    """
    name = str(item.get("name") or "")
    args = item.get("args") or []
    kwargs = item.get("kwargs") or {}
    if name == "geecs_scan_request_plan" and args and isinstance(args[0], dict):
        return summarize_request(args[0])
    if name == "geecs_run_action_plan" and args:
        return f"Action: {args[0]}"
    # Forward-compatible with the named plans (GeecsBluesky ≥ 0.73.0:
    # geecs_noscan_plan / geecs_scan_plan / geecs_optimize_plan take the
    # request's parts as keyword arguments — axes / capture / optimization).
    if (
        name.startswith("geecs_")
        and isinstance(kwargs, dict)
        and any(key in kwargs for key in ("capture", "axes", "optimization"))
    ):
        return summarize_request(kwargs)
    kind = str(item.get("item_type") or "item")
    return f"{name or '?'} ({kind})"


def _clock(epoch: Any) -> str:
    """``HH:MM:SS`` local time for a manager epoch timestamp ('' when absent)."""
    try:
        return datetime.fromtimestamp(float(epoch)).strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def _history_row(item: dict) -> QueueRow:
    """Render one finished item: exit status, finish time, first message line."""
    result = item.get("result") or {}
    if not isinstance(result, dict):
        result = {}
    status = str(result.get("exit_status") or "unknown")
    when = _clock(result.get("time_stop"))
    message = str(result.get("msg") or "").strip()
    first_line = message.splitlines()[0] if message else ""
    detail = " · ".join(part for part in (when, first_line) if part)
    return QueueRow(status, summarize_item(item), str(item.get("user") or ""), detail)


def build_rows(
    snapshot: QueueSnapshot, *, history_rows: int = HISTORY_ROWS
) -> list[QueueRow]:
    """Order the snapshot for display: running, waiting (front first), finished (newest first).

    Parameters
    ----------
    snapshot : QueueSnapshot
        A successful fetch (``error`` is not consulted here).
    history_rows : int
        How many finished items to keep.

    Returns
    -------
    list of QueueRow
        The table rows, top to bottom.
    """
    rows: list[QueueRow] = []
    if snapshot.running:
        item = snapshot.running
        rows.append(
            QueueRow("running", summarize_item(item), str(item.get("user") or ""))
        )
    for position, item in enumerate(snapshot.queued, start=1):
        rows.append(
            QueueRow(
                "queued",
                summarize_item(item),
                str(item.get("user") or ""),
                f"#{position}",
            )
        )
    history = snapshot.history or []
    recent = list(history)[-history_rows:] if history_rows > 0 else []
    rows.extend(_history_row(item) for item in reversed(recent))
    return rows


def summarize_counts(snapshot: QueueSnapshot) -> str:
    """The one-line summary above the table (``Running · 2 waiting``)."""
    parts = []
    if snapshot.running:
        parts.append("Running")
    waiting = len(snapshot.queued)
    if waiting:
        parts.append(f"{waiting} waiting")
    if not parts:
        return "Queue empty"
    return " · ".join(parts)


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------


class QueuePanelController(QObject):
    """Render the R8 queue region on behalf of the main window.

    Parameters
    ----------
    table, summary_label, clear_button : QWidget
        The R8 widgets (bound from the ``.ui`` by the window, which
        remains their attribute home for tests and tooltips).
    client_provider : callable
        Returns the window's current queue client (``None`` before one
        exists).  Called on the GUI thread; only the returned client's
        read verbs run on the daemon thread.
    confirm : callable
        ``confirm(title, message) -> bool`` — the Clear-queue question
        (the window's ``_ask_binary``); ``False`` leaves the queue as-is.
    report : callable
        Status-bar/log-tail line sink (the window's ``_report``).
    history_rows : int
        Finished items shown.
    fallback_refresh_s : float
        Refresh cadence when the status poll shows no queue change.
    """

    def __init__(
        self,
        *,
        table: QTableWidget,
        summary_label: QLabel,
        clear_button: QPushButton,
        client_provider: Callable[[], Any],
        confirm: Callable[[str, str], bool],
        report: Callable[[str], None],
        history_rows: int = HISTORY_ROWS,
        fallback_refresh_s: float = FALLBACK_REFRESH_S,
    ) -> None:
        super().__init__()
        self._table = table
        self._summary_label = summary_label
        self._clear_button = clear_button
        self._client_provider = client_provider
        self._confirm = confirm
        self._report = report
        self._history_rows = history_rows
        self._fallback_refresh_s = fallback_refresh_s

        self._last_key: Optional[tuple] = None
        self._last_refresh_at = float("-inf")
        self._items_in_queue = 0
        self._fetch_inflight = False
        #: A refresh asked for while one was in flight (and whether it
        #: wants the history) — honoured as soon as the in-flight one lands.
        self._refresh_pending = False
        self._pending_history = False
        #: The last fetched history (a queue-only refresh reuses it).
        self._history: list[dict] = []
        self._clear_inflight = False
        self._disposed = False

        self._fetch_worker = BackgroundResult()
        self._fetch_worker.result_ready.connect(
            self._apply_snapshot, Qt.ConnectionType.QueuedConnection
        )
        self._clear_worker = BackgroundResult()
        self._clear_worker.result_ready.connect(
            self._apply_clear_result, Qt.ConnectionType.QueuedConnection
        )
        self._clear_button.clicked.connect(self._on_clear_clicked)
        self._clear_button.setEnabled(False)

        self._setup_table()
        self._render_message("Queue: waiting for the manager")

    def _setup_table(self) -> None:
        """Read-only, row-selecting table with the Item column taking the width."""
        table = self._table
        table.setColumnCount(len(COLUMNS))
        table.setHorizontalHeaderLabels(list(COLUMNS))
        table.setRowCount(0)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def on_status(self, status: Any) -> None:
        """Consume one manager status snapshot (GUI thread; the poll's fan-out).

        Parameters
        ----------
        status : QueueStatus
            The scan monitor's polled snapshot.
        """
        if self._disposed:
            return
        if not status.connected:
            self._last_key = None
            self._items_in_queue = 0
            self._clear_button.setEnabled(False)
            self._render_message("Queue: manager unreachable")
            return
        self._items_in_queue = int(status.items_in_queue or 0)
        self._refresh_clear_enabled()
        key = (status.items_in_queue, status.running_item_uid, status.re_state)
        if key != self._last_key:
            # The key is what the history can change on (an item finished
            # or started): full refresh.  Committed before the call — a
            # refresh that cannot start now is queued behind the one in
            # flight (``_refresh_pending``), never lost.
            self._last_key = key
            self.refresh(include_history=True)
        elif time.monotonic() - self._last_refresh_at >= self._fallback_refresh_s:
            # Nothing the status shows changed: re-read the queue only (a
            # reorder by another client); the cached history stands.
            self.refresh(include_history=False)

    def refresh(self, *, include_history: bool = True) -> None:
        """Fetch the queue on a daemon thread.

        While a fetch is in flight the request is remembered and honoured
        when that fetch lands (``_apply_snapshot``), so a queue change
        arriving mid-fetch is never dropped until the fallback tick.

        Parameters
        ----------
        include_history : bool
            Also re-read the manager's (unbounded) history; ``False``
            re-renders with the cached one.
        """
        if self._disposed:
            return
        if self._fetch_inflight:
            self._refresh_pending = True
            self._pending_history = self._pending_history or include_history
            return
        client = self._client_provider()
        if client is None:
            return
        self._fetch_inflight = True
        self._last_refresh_at = time.monotonic()

        def fetch() -> QueueSnapshot:
            try:
                return QueueSnapshot(
                    running=client.running_item(),
                    queued=list(client.queue_items()),
                    history=list(client.history_items()) if include_history else None,
                )
            except Exception as exc:  # noqa: BLE001 — rendered, never raised
                return QueueSnapshot(error=str(exc) or exc.__class__.__name__)

        self._fetch_worker.run_async(fetch, name="console-queue-fetch")

    def _run_pending_refresh(self) -> None:
        """Start the refresh that was asked for while the last one was in flight."""
        if not self._refresh_pending:
            return
        include_history = self._pending_history
        self._refresh_pending = False
        self._pending_history = False
        self.refresh(include_history=include_history)

    # ------------------------------------------------------------------
    # Rendering (GUI thread)
    # ------------------------------------------------------------------

    @Slot(object)
    def _apply_snapshot(self, snapshot: object) -> None:
        """Render one fetch result (queued delivery from the daemon thread)."""
        self._fetch_inflight = False
        if self._disposed or not isinstance(snapshot, QueueSnapshot):
            return
        if snapshot.error:
            self._render_message(f"Queue: unavailable ({snapshot.error})")
            self._run_pending_refresh()
            return
        if snapshot.history is not None:
            self._history = list(snapshot.history)
        rows = build_rows(
            replace(snapshot, history=self._history), history_rows=self._history_rows
        )
        self._render_rows(rows)
        self._summary_label.setText(summarize_counts(snapshot))
        self._run_pending_refresh()

    def _render_rows(self, rows: list[QueueRow]) -> None:
        table = self._table
        table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            cells = (row.state, row.item, row.user, row.detail)
            for column, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                cell.setToolTip(text)
                if column == 0:
                    color = _STATE_COLORS.get(row.state.lower(), _COLOR_GREY)
                    cell.setForeground(QColor(color))
                table.setItem(index, column, cell)

    def _render_message(self, text: str) -> None:
        """Replace the table contents with nothing and say why in the summary."""
        self._table.setRowCount(0)
        self._summary_label.setText(text)

    def _refresh_clear_enabled(self) -> None:
        self._clear_button.setEnabled(
            self._items_in_queue > 0 and not self._clear_inflight
        )

    # ------------------------------------------------------------------
    # Clear queue
    # ------------------------------------------------------------------

    @Slot()
    def _on_clear_clicked(self) -> None:
        """Ask, then remove every waiting item on a daemon thread."""
        if self._disposed or self._clear_inflight:
            return
        client = self._client_provider()
        if client is None:
            return
        count = self._items_in_queue
        if not self._confirm(
            "Clear queue",
            (
                f"Remove all {count} waiting item(s) from the queue? A running "
                "scan is not affected — use Stop for that."
            ),
        ):
            return
        self._clear_inflight = True
        self._refresh_clear_enabled()

        def clear() -> tuple[bool, str]:
            try:
                return client.clear_queue()
            except Exception as exc:  # noqa: BLE001 — rendered, never raised
                return False, str(exc) or exc.__class__.__name__

        self._clear_worker.run_async(clear, name="console-queue-clear")

    @Slot(object)
    def _apply_clear_result(self, result: object) -> None:
        """Report the clear's outcome and force a refresh (queued delivery)."""
        self._clear_inflight = False
        if self._disposed:
            return
        ok, message = result if isinstance(result, tuple) else (False, str(result))
        self._report(message if ok else f"Clear queue failed: {message}")
        # The next status poll re-derives the button; the queue itself is
        # re-read now rather than waiting for the poll to notice.
        self._last_key = None
        self.refresh()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        """Sever every controller → window reference; idempotent.

        Detaches both workers so a straggling result lands nowhere and
        replaces the injected closures — the only Python edges back to
        the window — so the dead window is freed by refcount.
        """
        if self._disposed:
            return
        self._disposed = True
        for worker, slot in (
            (self._fetch_worker, self._apply_snapshot),
            (self._clear_worker, self._apply_clear_result),
        ):
            try:
                worker.result_ready.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        try:
            self._clear_button.clicked.disconnect(self._on_clear_clicked)
        except (RuntimeError, TypeError):
            pass
        self._client_provider = lambda: None
        self._confirm = lambda title, message: False
        self._report = lambda message: None
