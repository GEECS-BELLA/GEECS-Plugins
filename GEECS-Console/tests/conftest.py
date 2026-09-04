"""Shared fixtures: force the offscreen Qt platform before any QApplication."""

import gc
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Free a dropped window at a safe point, not under Qt's event dispatch.

    A ``MainWindow`` built inside a test body is Python-owned and cyclic
    (window ↔ controllers through the injected bound-method callbacks), so
    dropping the last reference at ``return`` leaves it to the cyclic GC —
    which runs at an arbitrary allocation, including inside pytest-qt's
    post-call ``processEvents`` while Qt is delivering an event to that
    very window (a polish walk, a queued poll result).  shiboken then
    deletes the C++ window under the running dispatch: segfault, at a
    random test, in isolated ``test_main_window.py`` runs (#767 follow-up
    — the 1 Hz status poll's fetch/render allocations made the GC land
    there).  Collecting here — after the body, before pytest-qt's own
    hook processes events (its wrapper is ``tryfirst``, so its post-yield
    runs after this one) — deletes such windows with no Qt frame on the
    stack.  Tests that drop a window *mid*-body and then pump events must
    still collect on their own.
    """
    try:
        return (yield)
    finally:
        gc.collect()


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
