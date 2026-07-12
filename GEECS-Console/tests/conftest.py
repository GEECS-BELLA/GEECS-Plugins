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


@pytest.fixture(autouse=True)
def _offline_window_defaults(monkeypatch):
    """Neutralize MainWindow's network-touching default seams (hermetic).

    A window built without an injected ``completions_factory`` /
    ``scan_number_lookup`` would otherwise dispatch daemon threads at the
    real ``GeecsDb`` and the real data root on every construction.  The
    module-level defaults are resolved lazily at fetch time, so patching
    them here keeps every test offline; tests of the features themselves
    inject fakes through the constructor parameters.
    """
    from geecs_console.app import main_window
    from geecs_console.services.device_completions import EmptyCompletions

    monkeypatch.setattr(
        main_window,
        "_default_completions_factory",
        lambda experiment: EmptyCompletions(),
    )
    monkeypatch.setattr(main_window, "_idle_scan_lookup", lambda experiment: None)
