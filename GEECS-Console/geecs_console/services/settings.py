"""Tiny persistent GUI-state helper (QSettings-backed) — not a framework.

One class, one key so far: the last selected experiment, remembered across
sessions so the console reopens pointed where the operator left it (health
probe and configs included).  Future GUI state (window geometry, last
preset, …) belongs here as more properties — resist anything grander.

The backing store is ``QSettings("GEECS", "GEECS-Console")`` in INI format
(INI honors ``QSettings.setPath``, so tests can redirect the user scope to a
tmp dir; the platform-native macOS plist store ignores it).  Tests can also
inject their own ``QSettings`` instance directly.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings

_ORGANIZATION = "GEECS"
_APPLICATION = "GEECS-Console"
_LAST_EXPERIMENT_KEY = "session/last_experiment"


class ConsoleSettings:
    """Persisted console GUI state.

    Parameters
    ----------
    qsettings : QSettings, optional
        The backing store; tests inject one pointed at a tmp INI file.
        Defaults to the user-scope ``GEECS / GEECS-Console`` INI store.
    """

    def __init__(self, qsettings: Optional[QSettings] = None) -> None:
        self._settings = (
            qsettings
            if qsettings is not None
            else QSettings(
                QSettings.Format.IniFormat,
                QSettings.Scope.UserScope,
                _ORGANIZATION,
                _APPLICATION,
            )
        )

    @property
    def last_experiment(self) -> str:
        """The experiment selected when the console last changed it ("" if never)."""
        return str(self._settings.value(_LAST_EXPERIMENT_KEY, "") or "")

    @last_experiment.setter
    def last_experiment(self, experiment: str) -> None:
        self._settings.setValue(_LAST_EXPERIMENT_KEY, experiment)
        self._settings.sync()
