"""Generate the annotated B1-B7 screen-map figure for the scan-browser docs.

Same technique as ``generate_screen_map.py`` (headless, reproducible):
opens the real ScanBrowserWindow offscreen, populated from the test
suite's synthetic FakeCatalog so every region has real content, and draws
the region highlights from the live widget geometry (the union bounding
box of each ``bN_``-prefixed widget).  Re-run after user-visible UI
changes:

    QT_QPA_PLATFORM=offscreen poetry run python \
        scripts/generate_browser_screen_map.py \
        ../docs/geecs_console/assets/browser_screen_map.png
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QPainter, QPen  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

# The synthetic catalog lives with the tests (it IS the documented fake).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

REGION_COLORS = {
    1: QColor(66, 133, 244),  # controls bar — blue
    2: QColor(52, 168, 83),  # run list — green
    3: QColor(244, 180, 0),  # run header — amber
    4: QColor(171, 71, 188),  # plot — purple
    5: QColor(234, 67, 53),  # table — red
    6: QColor(0, 172, 193),  # drift — cyan
    7: QColor(255, 112, 67),  # scan metadata — orange
}


def _region_rect(window: QWidget, region: int) -> QRect | None:
    """Union bounding box of the visible ``b{region}_*`` widgets."""
    rect: QRect | None = None
    for child in window.findChildren(QWidget):
        name = child.objectName()
        if not name.startswith(f"b{region}_") or not child.isVisible():
            continue
        top_left = child.mapTo(window, QPoint(0, 0))
        child_rect = QRect(top_left, child.size())
        rect = child_rect if rect is None else rect.united(child_rect)
    return rect


def main(out_path: str) -> None:
    """Capture the populated browser and write the annotated PNG."""
    # Isolate QSettings so the capture never touches real user settings
    # (the same redirection the test conftest uses).
    tmp = tempfile.mkdtemp(prefix="browser_screen_map_")
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, tmp)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    from fake_catalog import TEST_DAY, FakeCatalog, make_detail

    from geecs_console.browser.browser_window import ScanBrowserWindow
    from geecs_console.services.settings import ConsoleSettings

    catalog = FakeCatalog(
        [
            make_detail(1, hour=9, minute=11, description="warmup"),
            make_detail(
                2,
                hour=9,
                minute=27,
                motor="jet_x-position",
                num_points=5,
                shots_per_step=10,
                description="jet x alignment",
            ),
            make_detail(3, hour=10, minute=2, description="reference"),
        ]
    )
    window = ScanBrowserWindow(
        catalog=catalog,
        settings=ConsoleSettings(),
        today=TEST_DAY,
        scan_folder_resolver=lambda detail, day: None,
    )
    window.resize(1500, 950)
    window.show()

    def settle(condition, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not condition():
            if time.monotonic() > deadline:
                raise SystemExit("browser did not settle — condition timed out")
            app.processEvents()
            time.sleep(0.02)

    settle(lambda: window.b2_run_list.count() == 3)
    window.b2_run_list.setCurrentRow(1)  # the 1D alignment scan
    settle(lambda: window._detail is not None)
    window.add_series("cam-counts")  # a plotted Y series, so B4 shows data
    for _ in range(10):
        app.processEvents()

    pixmap = window.grab()
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    badge_font = QFont()
    badge_font.setBold(True)
    badge_font.setPointSize(13)

    for region, color in REGION_COLORS.items():
        rect = _region_rect(window, region)
        if rect is None:  # UI evolved: fail loudly, don't draw a lie
            raise SystemExit(f"no visible b{region}_* widgets — update this script")
        rect = rect.adjusted(-4, -4, 4, 4)
        fill = QColor(color)
        fill.setAlpha(26)
        painter.setPen(QPen(color, 3))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 6, 6)

        # Top-right, except B4 whose top row holds controls: drop that
        # badge below the X/Y selector row so it covers only plot canvas.
        badge_top = rect.top() + 40 if region == 4 else rect.top() - 2
        badge = QRect(rect.right() - 42, badge_top, 44, 30)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(badge, 8, 8)
        painter.setPen(QPen(QColor("white")))
        painter.setFont(badge_font)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, f"B{region}")

    painter.end()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(out), "PNG"):
        raise SystemExit(f"failed to write {out}")
    print(f"wrote {out} ({pixmap.width()}x{pixmap.height()})")
    window.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "browser_screen_map.png")
