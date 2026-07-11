"""get_action_control constructs lazily post-G1 (review fix on the G1 PR).

The legacy backend built ActionControl at RunControl init; the bluesky-only
RunControl must build it on first use (ActionControl/ActionManager are
backend-independent keeps) — returning None would crash the ActionLibrary's
perform-action path with AttributeError.

``geecs_scanner.app``'s package __init__ imports the PyQt5 editor windows, so
``run_control`` is loaded straight from its file (its own module-level imports
are Qt-free) to keep this test hermetic on non-Windows CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

# run_control imports geecs_scanner.app.lib.action_control, which triggers
# geecs_scanner.app.__init__ and its PyQt5 editor imports. Stub Qt with
# MagicMock modules so the chain loads headless (same approach as the G1
# import sanity checks); arbitrary attribute/call chains all resolve.
for _qt_mod in ("PyQt5", "PyQt5.QtWidgets", "PyQt5.QtCore", "PyQt5.QtGui"):
    sys.modules.setdefault(_qt_mod, MagicMock(name=_qt_mod))

_RUN_CONTROL_PATH = (
    Path(__file__).parents[2] / "geecs_scanner" / "app" / "run_control.py"
)
_spec = importlib.util.spec_from_file_location(
    "_run_control_under_test", _RUN_CONTROL_PATH
)
_run_control = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_run_control)
RunControl = _run_control.RunControl


class _FakeActionControl:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name


def _bare_run_control() -> "RunControl":
    rc = RunControl.__new__(RunControl)
    rc.experiment_name = "TestExp"
    rc.action_control = None
    return rc


def test_lazy_construction_on_first_use(monkeypatch):
    monkeypatch.setattr(_run_control, "ActionControl", _FakeActionControl)
    rc = _bare_run_control()
    ac = rc.get_action_control()
    assert isinstance(ac, _FakeActionControl)
    assert ac.experiment_name == "TestExp"
    # cached: same instance on subsequent calls
    assert rc.get_action_control() is ac


def test_refresh_rebuilds_for_new_experiment(monkeypatch):
    monkeypatch.setattr(_run_control, "ActionControl", _FakeActionControl)
    rc = _bare_run_control()
    first = rc.get_action_control()
    refreshed = rc.get_action_control(experiment_name_refresh="OtherExp")
    assert refreshed is not first
    assert refreshed.experiment_name == "OtherExp"
    assert rc.get_action_control() is refreshed
