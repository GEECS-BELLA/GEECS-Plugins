"""BlueskyScanner.request_action_during_scan (G-actions v2, #552, PR-3).

The bridge half of the during-scan action path: fail-fast validation and
the shot-control-device refusal (owner decision 11) on the GUI thread, the
PAUSING emission + deferred pause request, single-slot staging, and the
no-scan refusal.  The pause-window execution itself is pinned in
``test_pause_supervisor.py``; here we drive the bridge with fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from geecs_bluesky.events import ScanState
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.pause_supervisor import PauseSupervisor
from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner
from geecs_schemas import (
    ActionPlan,
    ScanRequest,
    TriggerProfile,
)


class _FakeSignal:
    def set(self, value):  # pragma: no cover - never executed here
        raise NotImplementedError


class _FakeFactory:
    def __init__(self) -> None:
        self.unreachable: set = set()

    def get_settable(self, device, variable):
        if (device, variable) in self.unreachable:
            raise RuntimeError("no such PV")
        return _FakeSignal()

    def get_readable(self, device, variable):
        return _FakeSignal()


class _FakeResolver:
    def __init__(self, plans, profiles=None) -> None:
        self._plans = plans
        self._profiles = profiles or {}

    def resolve_action_plan(self, name):
        return self._plans[name]

    def resolve_trigger_profile(self, name):
        return self._profiles[name]


class _FakeSession:
    def __init__(self, resolver) -> None:
        self._resolver = resolver
        self.RE = SimpleNamespace(loop=None)
        self.factory = _FakeFactory()
        self.disconnected: list = []
        self._shot_controller = None

    def _resolve_action(self, name, resolver):
        r = resolver or self._resolver
        return r.resolve_action_plan(name), {
            n: r.resolve_action_plan(n) for n in getattr(r, "_plans", {})
        }

    def action_signal_factory(self):
        return self.factory

    def disconnect(self, *devices):
        self.disconnected.extend(devices)


def _make_running_scanner(session, request, resolver, *, scanning=True):
    scanner = BlueskyScanner.__new__(BlueskyScanner)
    scanner._session = session
    scanner._experiment_dir = "TestExp"
    scanner._scan_request = request
    scanner._request_resolver = resolver
    scanner._abort_requested = False
    scanner._scan_number = None
    scanner.events: list = []
    scanner._on_event = scanner.events.append
    scanner._current_state = ScanState.RUNNING
    scanner._pause_supervisor = PauseSupervisor(
        acquisition="free_run",
        shot_controller=lambda: None,
        ask=scanner._ask_action_decision,
        should_abort=lambda: scanner._abort_requested,
    )
    pauses: list = []
    scanner._RE = SimpleNamespace(
        loop=None, request_pause=lambda defer=True: pauses.append(defer)
    )
    scanner._pauses = pauses
    # Force is_scanning_active without a real thread.
    scanner._scan_thread = SimpleNamespace(
        is_alive=lambda: scanning, join=lambda timeout=None: None
    )
    return scanner


_JET_ON = ActionPlan.model_validate(
    {"steps": [{"do": "set", "device": "U_Valve", "variable": "Open", "value": 1}]}
)
_TOUCHES_DG = ActionPlan.model_validate(
    {"steps": [{"do": "set", "device": "U_DG645", "variable": "Amp", "value": 9}]}
)
_PROFILE = TriggerProfile(
    name="HTU",
    states={
        "SCAN": [{"device": "U_DG645", "variable": "Amplitude.Ch AB", "value": "4.0"}]
    },
)


def _request(**kw) -> ScanRequest:
    base = dict(
        mode="noscan", shots_per_step=1, acquisition="free_run", save_sets=["s"]
    )
    base.update(kw)
    return ScanRequest.model_validate(base)


def test_requests_deferred_pause_and_emits_pausing() -> None:
    resolver = _FakeResolver({"jet_on": _JET_ON})
    session = _FakeSession(resolver)
    scanner = _make_running_scanner(session, _request(), resolver)
    scanner.request_action_during_scan("jet_on")
    assert scanner._pauses == [True]  # deferred pause requested
    assert scanner._current_state == ScanState.PAUSING
    assert scanner._pause_supervisor._pending is not None


def test_refused_when_action_touches_shot_control_device() -> None:
    resolver = _FakeResolver({"bad": _TOUCHES_DG}, profiles={"HTU": _PROFILE})
    request = _request(trigger_profile="HTU")
    session = _FakeSession(resolver)
    scanner = _make_running_scanner(session, request, resolver)
    with pytest.raises(GeecsConfigurationError, match="shot-control"):
        scanner.request_action_during_scan("bad")
    assert scanner._pauses == []  # never paused
    assert scanner._pause_supervisor._pending is None


def test_unreachable_target_refused_before_pausing() -> None:
    resolver = _FakeResolver({"jet_on": _JET_ON})
    session = _FakeSession(resolver)
    session.factory.unreachable.add(("U_Valve", "Open"))
    scanner = _make_running_scanner(session, _request(), resolver)
    with pytest.raises(GeecsConfigurationError, match="cannot reach"):
        scanner.request_action_during_scan("jet_on")
    assert scanner._pauses == []
    assert session.disconnected  # the factory was released


def test_refused_when_no_scan_active() -> None:
    resolver = _FakeResolver({"jet_on": _JET_ON})
    session = _FakeSession(resolver)
    scanner = _make_running_scanner(session, _request(), resolver, scanning=False)
    with pytest.raises(RuntimeError, match="no scan in progress"):
        scanner.request_action_during_scan("jet_on")


def test_single_action_per_pause_window() -> None:
    resolver = _FakeResolver({"jet_on": _JET_ON})
    session = _FakeSession(resolver)
    scanner = _make_running_scanner(session, _request(), resolver)
    scanner.request_action_during_scan("jet_on")
    with pytest.raises(RuntimeError, match="already awaiting"):
        scanner.request_action_during_scan("jet_on")


_TOUCHES_DG_LOWER = ActionPlan.model_validate(
    {"steps": [{"do": "set", "device": "u_dg645", "variable": "Amp", "value": 9}]}
)


def test_case_mismatched_shot_control_device_still_refused() -> None:
    """Decision 11 is case-insensitive (GEECS configs disagree on case)."""
    resolver = _FakeResolver({"bad": _TOUCHES_DG_LOWER}, profiles={"HTU": _PROFILE})
    request = _request(trigger_profile="HTU")  # profile writes 'U_DG645'
    scanner = _make_running_scanner(_FakeSession(resolver), request, resolver)
    with pytest.raises(GeecsConfigurationError, match="shot-control"):
        scanner.request_action_during_scan("bad")  # 'u_dg645' vs 'U_DG645'


def test_stop_during_pause_does_not_abort_the_re_itself() -> None:
    """#552: while paused, stop_scanning_thread lets the supervisor own the
    abort — a second RE.abort() here would truncate the finalize cleanup."""
    aborts: list = []
    resolver = _FakeResolver({"jet_on": _JET_ON})
    scanner = _make_running_scanner(_FakeSession(resolver), _request(), resolver)
    scanner._RE = SimpleNamespace(
        state="paused",
        request_pause=lambda defer=True: None,
        abort=lambda reason=None: aborts.append(reason),
    )
    scanner._set_state = lambda *a, **k: None  # skip event plumbing
    scanner.stop_scanning_thread()
    assert scanner._abort_requested is True  # flag set (wakes the park loop)
    assert aborts == []  # but the RE was NOT aborted from here (supervisor owns it)


def test_console_less_bridge_gives_supervisor_no_ask() -> None:
    """A bridge with no on_event consumer hands the supervisor ask=None, so
    a pause window defaults to 'ignore' rather than parking forever."""
    resolver = _FakeResolver({"jet_on": _JET_ON})
    scanner = _make_running_scanner(_FakeSession(resolver), _request(), resolver)
    scanner._on_event = None  # no consumer
    sup = scanner._make_pause_supervisor()
    assert sup._ask is None


def test_request_pause_arms_manual_pause_and_defers() -> None:
    """The operator Pause button: arm a manual pause + deferred RE pause."""
    resolver = _FakeResolver({})
    scanner = _make_running_scanner(_FakeSession(resolver), _request(), resolver)
    scanner.request_pause()
    assert scanner._pauses == [True]  # deferred pause requested
    assert scanner._current_state == ScanState.PAUSING
    assert scanner._pause_supervisor._manual_pause is True


def test_request_resume_signals_the_supervisor() -> None:
    resolver = _FakeResolver({})
    scanner = _make_running_scanner(_FakeSession(resolver), _request(), resolver)
    scanner._pause_supervisor.arm_manual_pause()
    assert not scanner._pause_supervisor._resume_event.is_set()
    scanner.request_resume()
    assert scanner._pause_supervisor._resume_event.is_set()


def test_request_pause_no_op_without_active_scan() -> None:
    resolver = _FakeResolver({})
    scanner = _make_running_scanner(
        _FakeSession(resolver), _request(), resolver, scanning=False
    )
    scanner.request_pause()
    assert scanner._pauses == []  # nothing paused
