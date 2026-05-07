"""Status overview dialog for LiveWatch analysis.

Opens a modeless dialog showing a scan × analyzer grid for the selected day,
with colour-coded cells for each task state.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from geecs_data_utils import ScanTag
from scan_analysis.task_queue import TaskStatus, read_day_statuses

logger = logging.getLogger(__name__)

# State → (background colour, display label)
_STATE_STYLE: Dict[str, Tuple[QColor, str]] = {
    "done": (QColor(144, 238, 144), "done"),
    "failed": (QColor(255, 160, 122), "failed"),
    "claimed": (QColor(255, 215, 0), "claimed"),
    "no_data": (QColor(176, 196, 222), "no data"),
    "queued": (QColor(220, 220, 220), "queued"),
}
_DEFAULT_STYLE: Tuple[QColor, str] = (QColor(245, 245, 245), "—")

_AUTO_REFRESH_MS = 10_000


class StatusDialog(QDialog):
    """Modeless scan × analyzer status grid.

    Parameters
    ----------
    date_tag : ScanTag
        Identifies the day and experiment to inspect.
    analyzer_ids : list[str]
        Ordered list of analyzer IDs (column headers).  Any extra IDs found
        in on-disk status files are appended as additional columns.
    base_directory : Path or None
        Root data directory; ``None`` defers to the configured default.
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(
        self,
        date_tag: ScanTag,
        analyzer_ids: List[str],
        base_directory: Optional[Path],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Analysis Status")
        self.setMinimumSize(700, 320)
        self.resize(900, 480)
        # Keep the dialog on top of its parent but allow it to be moved freely.
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        self._date_tag = date_tag
        self._analyzer_ids = list(analyzer_ids)
        self._base_directory = base_directory

        self._timer = QTimer(self)
        self._timer.setInterval(_AUTO_REFRESH_MS)
        self._timer.timeout.connect(self._refresh)

        self._build_ui()
        self._check_auto.setChecked(True)
        self._refresh()

    # ------------------------------------------------------------------
    # Public API — called by the parent window to update config
    # ------------------------------------------------------------------

    def update_config(
        self,
        date_tag: ScanTag,
        analyzer_ids: List[str],
        base_directory: Optional[Path],
    ) -> None:
        """Apply new date / analyzer list and immediately refresh."""
        self._date_tag = date_tag
        self._analyzer_ids = list(analyzer_ids)
        self._base_directory = base_directory
        self._refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── top bar ──────────────────────────────────────────────────
        top = QHBoxLayout()

        self._label_summary = QLabel("—")
        self._label_summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top.addWidget(self._label_summary)

        self._check_auto = QCheckBox("Auto-refresh (10 s)")
        self._check_auto.toggled.connect(self._on_auto_toggled)
        top.addWidget(self._check_auto)

        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self._refresh)
        top.addWidget(self._btn_refresh)

        layout.addLayout(top)

        # ── legend ───────────────────────────────────────────────────
        legend = QHBoxLayout()
        legend.setSpacing(12)
        for state, (colour, label) in _STATE_STYLE.items():
            chip = QLabel(f"  {label}  ")
            chip.setAutoFillBackground(True)
            p = chip.palette()
            p.setColor(chip.backgroundRole(), colour)
            chip.setPalette(p)
            chip.setAlignment(Qt.AlignCenter)
            legend.addWidget(chip)
        legend.addStretch()
        layout.addLayout(legend)

        # ── table ────────────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._table)

    # ------------------------------------------------------------------
    # Auto-refresh
    # ------------------------------------------------------------------

    def _on_auto_toggled(self, checked: bool) -> None:
        if checked:
            self._timer.start()
        else:
            self._timer.stop()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_statuses(
        self,
    ) -> Tuple[List[int], List[str], Dict[int, Dict[str, TaskStatus]]]:
        """Return (sorted scan numbers, ordered analyzer IDs, scan→id→status map)."""
        scan_map = read_day_statuses(
            self._date_tag, base_directory=self._base_directory
        )

        # Merge any extra analyzer IDs found on disk into the column list.
        known = set(self._analyzer_ids)
        extra: List[str] = []
        for by_id in scan_map.values():
            for aid in by_id:
                if aid not in known:
                    extra.append(aid)
                    known.add(aid)

        return list(scan_map), list(self._analyzer_ids) + extra, scan_map

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        scan_numbers, analyzer_ids, scan_map = self._load_statuses()
        self._populate_table(scan_numbers, analyzer_ids, scan_map)
        self._update_summary(scan_numbers, analyzer_ids, scan_map)

    def _populate_table(
        self,
        scan_numbers: List[int],
        analyzer_ids: List[str],
        scan_map: Dict[int, Dict[str, TaskStatus]],
    ) -> None:
        n_rows = len(scan_numbers)
        n_cols = 1 + len(analyzer_ids)  # scan-number column + one per analyzer

        self._table.setRowCount(n_rows)
        self._table.setColumnCount(n_cols)

        headers = ["Scan #"] + analyzer_ids
        self._table.setHorizontalHeaderLabels(headers)

        now = datetime.now(timezone.utc)

        for row, scan_num in enumerate(scan_numbers):
            # Scan number cell
            num_item = QTableWidgetItem(str(scan_num))
            num_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 0, num_item)

            by_id = scan_map.get(scan_num, {})
            for col_offset, aid in enumerate(analyzer_ids):
                status = by_id.get(aid)
                item = self._make_cell(status, now)
                self._table.setItem(row, 1 + col_offset, item)

        self._table.resizeColumnsToContents()

    @staticmethod
    def _make_cell(status: Optional[TaskStatus], now: datetime) -> QTableWidgetItem:
        if status is None:
            colour, label = _DEFAULT_STYLE
            tooltip = ""
        else:
            colour, label = _STATE_STYLE.get(status.state, _DEFAULT_STYLE)
            tooltip = StatusDialog._build_tooltip(status, now)

        item = QTableWidgetItem(label)
        item.setTextAlignment(Qt.AlignCenter)
        item.setBackground(colour)
        if tooltip:
            item.setToolTip(tooltip)
        return item

    @staticmethod
    def _build_tooltip(status: TaskStatus, now: datetime) -> str:
        if status.state == "failed" and status.error:
            return f"Error: {status.error}"
        if status.state == "claimed":
            parts: List[str] = []
            if status.claimed_by:
                parts.append(f"Worker: {status.claimed_by}")
            hb_str = status.last_heartbeat or status.claimed_at
            if hb_str:
                try:
                    hb_dt = datetime.fromisoformat(hb_str)
                    if hb_dt.tzinfo is None:
                        hb_dt = hb_dt.replace(tzinfo=timezone.utc)
                    elapsed = int((now - hb_dt).total_seconds())
                    parts.append(f"Last heartbeat: {elapsed}s ago")
                except Exception:
                    pass
            return "\n".join(parts)
        return ""

    # ------------------------------------------------------------------
    # Summary label
    # ------------------------------------------------------------------

    def _update_summary(
        self,
        scan_numbers: List[int],
        analyzer_ids: List[str],
        scan_map: Dict[int, Dict[str, TaskStatus]],
    ) -> None:
        if not scan_numbers:
            date_str = (
                f"{self._date_tag.year:04d}-{self._date_tag.month:02d}"
                f"-{self._date_tag.day:02d}"
            )
            self._label_summary.setText(f"No scans found for {date_str}")
            return

        counts: Dict[str, int] = {s: 0 for s in _STATE_STYLE}
        total = 0
        for scan_num in scan_numbers:
            by_id = scan_map.get(scan_num, {})
            for aid in analyzer_ids:
                status = by_id.get(aid)
                state = status.state if status else "queued"
                total += 1
                if state in counts:
                    counts[state] += 1

        parts = [f"{len(scan_numbers)} scan(s)"]
        parts.append(f"{counts['done']}/{total} done")
        if counts["claimed"]:
            parts.append(f"{counts['claimed']} running")
        if counts["failed"]:
            parts.append(f"{counts['failed']} failed")
        if counts["no_data"]:
            parts.append(f"{counts['no_data']} no data")
        if counts["queued"]:
            parts.append(f"{counts['queued']} queued")

        self._label_summary.setText("  |  ".join(parts))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Stop the auto-refresh timer before closing."""
        self._timer.stop()
        super().closeEvent(event)
