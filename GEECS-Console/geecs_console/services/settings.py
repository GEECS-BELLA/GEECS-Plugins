"""Tiny persistent GUI-state helper (QSettings-backed) — not a framework.

One class, a handful of keys: the last selected experiment (remembered
across sessions so the console reopens pointed where the operator left it,
health probe and configs included) and the Preferences beep options.
Future GUI state (window geometry, last preset, …) belongs here as more
properties — resist anything grander.

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
_PER_SHOT_BEEP_KEY = "preferences/per_shot_beep"
_RANDOMIZED_BEEPS_KEY = "preferences/randomized_beeps"


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

    def _get_bool(self, key: str) -> bool:
        """Read one boolean key (INI stores it as text; ``type=bool`` parses)."""
        return bool(self._settings.value(key, False, type=bool))

    def _set_bool(self, key: str, value: bool) -> None:
        """Write and sync one boolean key."""
        self._settings.setValue(key, bool(value))
        self._settings.sync()

    @property
    def per_shot_beep(self) -> bool:
        """Whether the Preferences per-shot beep is on (default off)."""
        return self._get_bool(_PER_SHOT_BEEP_KEY)

    @per_shot_beep.setter
    def per_shot_beep(self, value: bool) -> None:
        self._set_bool(_PER_SHOT_BEEP_KEY, value)

    @property
    def randomized_beeps(self) -> bool:
        """Whether beeps fire only for a random ~1-in-4 subset of shots (default off)."""
        return self._get_bool(_RANDOMIZED_BEEPS_KEY)

    @randomized_beeps.setter
    def randomized_beeps(self, value: bool) -> None:
        self._set_bool(_RANDOMIZED_BEEPS_KEY, value)
