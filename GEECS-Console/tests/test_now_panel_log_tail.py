"""The R6 log tail's consecutive-duplicate suppression (PR #624).

The tail has two writers — the events adapter's narration and the
window's direct status lines — so dedupe lives at their convergence
point, ``NowPanelController.append_log``, where "consecutive" means
consecutive-in-the-tail. Field report (Scan004, 2026-08-19): "scan
running" x2 and the doubled final step line.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QProgressBar

from geecs_console.app.now_panel import NowPanelController


@pytest.fixture
def panel(qapp):
    tail = QPlainTextEdit()
    controller = NowPanelController(
        state_pill=QLabel(),
        progress_bar=QProgressBar(),
        scan_number_label=QLabel(),
        log_tail=tail,
        current_experiment=lambda: "",
        resolve_lookup=lambda: (lambda experiment: None),
    )
    yield controller, tail
    controller.dispose()


def _lines(tail: QPlainTextEdit) -> list[str]:
    text = tail.toPlainText()
    return text.splitlines() if text else []


def test_consecutive_duplicates_collapse(panel) -> None:
    controller, tail = panel
    controller.append_log("step 1/1 completed (10 shots)")
    controller.append_log("step 1/1 completed (10 shots)")
    assert _lines(tail) == ["step 1/1 completed (10 shots)"]


def test_interleaved_writers_measure_against_the_tail(panel) -> None:
    """Adapter line / direct window line / identical adapter line: the
    third line is NOT a tail-consecutive duplicate and must render
    (review finding 1's false-suppression scenario — the re-asked
    operator question after an interleaved 'operator: continue')."""
    controller, tail = panel
    controller.append_log("operator question: boom")
    controller.append_log("operator: continue")
    controller.append_log("operator question: boom")
    assert _lines(tail) == [
        "operator question: boom",
        "operator: continue",
        "operator question: boom",
    ]


def test_direct_writer_duplicates_also_collapse(panel) -> None:
    controller, tail = panel
    controller.append_log("operator: continue")
    controller.append_log("operator: continue")
    assert _lines(tail) == ["operator: continue"]


def test_new_scan_rearms_the_cache(panel) -> None:
    """Scan N's last line must never swallow scan N+1's first (review
    finding 3): set_totals — the new scan announcing itself — re-arms."""
    controller, tail = panel
    controller.append_log("scan done")
    controller.set_totals(10)
    controller.append_log("scan done")  # pathological but must render
    assert _lines(tail) == ["scan done", "scan done"]


def test_alternating_lines_never_suppressed(panel) -> None:
    controller, tail = panel
    controller.append_log("step 1/1 completed (1 shots)")
    controller.append_log("step 1/1 completed (2 shots)")
    assert _lines(tail) == [
        "step 1/1 completed (1 shots)",
        "step 1/1 completed (2 shots)",
    ]
