"""Generate the annotated R1-R7 screen-map figure for the docs.

Headless and reproducible (the docs/CLAUDE.md screenshot convention):
opens the real MainWindow offscreen with its offline stub seams, grabs a
screenshot, and draws the region highlights **from the live widget
geometry** (each region's ``rN_group`` container), so the figure stays
pixel-accurate as the UI evolves.  Re-run after user-visible UI changes:

    QT_QPA_PLATFORM=offscreen poetry run python scripts/generate_screen_map.py \
        ../docs/geecs_console/assets/console_screen_map.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QPainter, QPen  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from geecs_console.app.main_window import MainWindow, load_stylesheet  # noqa: E402

#: One distinct hue per region (fill drawn at low alpha, badge at full).
REGION_COLORS = {
    1: QColor(66, 133, 244),  # session bar — blue
    2: QColor(52, 168, 83),  # save sets — green
    3: QColor(244, 180, 0),  # scan form — amber
    4: QColor(171, 71, 188),  # presets — purple
    5: QColor(234, 67, 53),  # start/stop — red
    6: QColor(0, 172, 193),  # now panel — cyan
    7: QColor(255, 112, 67),  # device panel — orange
}


def main(out_path: str) -> None:
    """Capture the window and write the annotated PNG to *out_path*."""
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyleSheet(load_stylesheet())
    window = MainWindow()
    window.resize(1340, 920)
    # The status bar shows local filesystem paths — never ship those in a
    # committed screenshot (public repo).
    window.statusBar().hide()
    window.show()
    for _ in range(8):
        app.processEvents()

    pixmap = window.grab()
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    badge_font = QFont()
    badge_font.setBold(True)
    badge_font.setPointSize(13)

    for region, color in REGION_COLORS.items():
        group = window.findChild(QWidget, f"r{region}_group")
        if group is None:  # UI evolved: fail loudly, don't draw a lie
            raise SystemExit(f"r{region}_group not found — update this script")
        top_left = group.mapTo(window, QPoint(0, 0))
        rect = QRect(top_left, group.size()).adjusted(2, 2, -2, -2)

        fill = QColor(color)
        fill.setAlpha(28)
        painter.setPen(QPen(color, 3))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 6, 6)

        # Top-right corner: group-box titles live at the top-left.
        badge = QRect(rect.right() - 42, rect.top() - 2, 44, 30)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(badge, 8, 8)
        painter.setPen(QPen(QColor("white")))
        painter.setFont(badge_font)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, f"R{region}")

    painter.end()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(out), "PNG"):
        raise SystemExit(f"failed to write {out}")
    print(f"wrote {out} ({pixmap.width()}x{pixmap.height()})")
    window.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "console_screen_map.png")
