"""Shared fixtures: force the offscreen Qt platform before any QApplication."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
