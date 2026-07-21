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

    class _OfflineActionStore:
        """No-op stand-in for the default ``ActionLibraryStore``.

        The real default's ``list_names`` resolves the configs repo —
        reading the developer's actual user config and lazily importing
        ``geecs_bluesky`` on a daemon thread per window construction.
        Tests of the Actions menu inject their own store; every other
        window must stay offline.
        """

        def __init__(self, experiment: str = "", experiments_root=None) -> None:
            self.experiment = experiment

        def set_experiment(self, experiment: str) -> None:
            self.experiment = experiment

        def list_names(self) -> list:
            return []

    monkeypatch.setattr(main_window, "ActionLibraryStore", _OfflineActionStore)
