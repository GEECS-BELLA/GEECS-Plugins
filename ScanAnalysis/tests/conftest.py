"""Shared pytest fixtures for ScanAnalysis tests."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication instance for Qt widget tests.

    Sets QT_QPA_PLATFORM=offscreen so tests run on headless servers.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
