"""Shared pytest fixtures for ScanAnalysis tests."""

from __future__ import annotations

import os

import pytest

# The legacy scan wrappers render per-bin figures from worker threads, which
# the interactive macOS backend cannot do (it aborts the interpreter). CI is
# headless already; this keeps a developer's local run headless too.
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication instance for Qt widget tests.

    Sets QT_QPA_PLATFORM=offscreen so tests run on headless servers.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
