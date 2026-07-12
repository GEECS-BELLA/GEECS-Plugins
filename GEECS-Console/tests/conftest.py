"""Shared fixtures: force the offscreen Qt platform before any QApplication."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path):
    """Point QSettings' user scope at a per-test tmp dir (hermetic settings).

    ``ConsoleSettings`` uses the INI format precisely because INI honors
    ``QSettings.setPath`` — so no test can read or write the developer's
    real ``GEECS/GEECS-Console`` settings, even when a window is built
    without an injected settings object.
    """
    from PySide6.QtCore import QSettings

    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path / "qsettings"),
    )
