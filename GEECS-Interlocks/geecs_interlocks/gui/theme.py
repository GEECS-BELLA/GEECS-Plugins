"""The window's colours and its one stylesheet: the state words, and the quiet chrome around them.

The states carry the same words and order as the monitor (``STATES``) and
the web client; only their colours live here.
"""

from __future__ import annotations

from geecs_interlocks.monitor import State

#: (background, text) of each state's chip and of the verdict strip.
STATE_COLOURS: dict[State, tuple[str, str]] = {
    "tripped": ("#FCEBEB", "#A32D2D"),
    "waiting": ("#F1EFE8", "#5F5E5A"),
    "settling": ("#FAEEDA", "#854F0B"),
    "ok": ("#EAF3DE", "#3B6D11"),
    "off": ("#F1EFE8", "#888780"),
}
#: The verdict strip when everything armed is OK.
SAFE_COLOURS = ("#EAF3DE", "#3B6D11")
#: The log pane's line colours: trips and warnings, UDP packets, the rest.
LOG_WARNING = "#A32D2D"
LOG_UDP = "#185FA5"
LOG_INFO = "#5F5E5A"
#: Muted text (footnotes, the "watched" markers).
MUTED = "#888780"

STYLESHEET = """
QWidget { font-size: 12px; }
QGroupBox { border: 1px solid #D3D1C7; border-radius: 6px; margin-top: 14px; padding: 8px 8px 6px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; font-weight: 600; }
QTreeWidget, QListWidget, QPlainTextEdit { border: 1px solid #D3D1C7; border-radius: 4px; }
QPlainTextEdit#log { font-family: Menlo, Consolas, monospace; font-size: 11px; background: #FAFAF7; }
QLabel#footnote { color: #888780; font-size: 11px; }
QPushButton#primary { background: #2C2C2A; color: white; border: none; border-radius: 4px; padding: 4px 12px; }
QPushButton#primary:disabled { background: #B4B2A9; }
QFrame#verdict { border-radius: 6px; }
"""

__all__ = [
    "LOG_INFO",
    "LOG_UDP",
    "LOG_WARNING",
    "MUTED",
    "SAFE_COLOURS",
    "STATE_COLOURS",
    "STYLESHEET",
]
