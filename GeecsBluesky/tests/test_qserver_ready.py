"""Hermetic tests for ``geecs-qserver-ensure-ready`` (#793 part 1).

The manager transport is a scripted fake ``request(method, params)``; no
manager, no zmq, no ``bluesky_queueserver`` import (the real transport is
built only in ``main``).  Sleeps are patched to zero.
"""

from __future__ import annotations

import pytest

from geecs_bluesky import qserver_ready
from geecs_bluesky.plan_names import GEECS_PLAN_NAMES
from geecs_bluesky.qserver_ready import NotReady, ensure_ready


def _plans(names):
    return {"success": True, "plans_allowed": {n: {"name": n} for n in names}}


def _status(*, exists: bool, manager="idle", env=None, re_state=None):
    return {
        "success": True,
        "manager_state": manager,
        "worker_environment_exists": exists,
        "worker_environment_state": env or ("idle" if exists else "closed"),
        "re_state": re_state if re_state is not None else ("idle" if exists else None),
    }


class _Manager:
    """Scripted manager: a queue of status snapshots, recording every call."""

    def __init__(
        self, statuses, *, plans=GEECS_PLAN_NAMES, open_reply=None, answer=True
    ):
        self.statuses = list(statuses)
        self.plans = list(plans)
        self.open_reply = open_reply or {"success": True}
        self.answer = answer
        self.calls: list[tuple[str, dict | None]] = []

    def __call__(self, method, params):
        self.calls.append((method, params))
        if not self.answer:
            return None, "timeout"
        if method == "status":
            snap = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return snap, ""
        if method == "environment_open":
            return self.open_reply, ""
        if method == "plans_allowed":
            return _plans(self.plans), ""
        raise AssertionError(f"unexpected method {method!r}")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(qserver_ready.time, "sleep", lambda s: None)


def test_closed_environment_is_opened_then_plans_asserted() -> None:
    manager = _Manager(
        [
            _status(exists=False),  # the fresh-restart state
            _status(exists=False),  # first poll after open: still the old snapshot
            _status(exists=True, manager="creating_environment", env="initializing"),
            _status(exists=True),
        ]
    )
    log = []
    allowed = ensure_ready(manager, timeout_s=30, log=log.append)

    methods = [m for m, _ in manager.calls]
    assert methods.index("environment_open") < methods.index("plans_allowed")
    assert methods.count("environment_open") == 1
    assert ("plans_allowed", {"user_group": "primary"}) in manager.calls
    assert allowed == sorted(GEECS_PLAN_NAMES)
    assert any("opening it" in line for line in log)
    assert log[-1].startswith("ready: 5 allowed plans")


def test_already_open_skips_the_open() -> None:
    manager = _Manager([_status(exists=True)])
    ensure_ready(manager, timeout_s=30, log=lambda s: None)
    assert "environment_open" not in [m for m, _ in manager.calls]


def test_missing_plan_is_not_ready_and_names_it() -> None:
    manager = _Manager([_status(exists=True)], plans=["geecs_run_action_plan"])
    with pytest.raises(NotReady) as excinfo:
        ensure_ready(manager, timeout_s=30, log=lambda s: None)
    message = str(excinfo.value)
    assert "geecs_scan_request_plan" in message
    assert "geecs_noscan_plan" in message
    assert "geecs_run_action_plan" not in message.split("(listed")[0]
    assert "user_group_permissions.yaml" in message


def test_empty_plan_list_after_open_is_not_ready() -> None:
    """The exact #793 invariant: open 'succeeded', plan list still empty."""
    manager = _Manager([_status(exists=False), _status(exists=True)], plans=[])
    with pytest.raises(NotReady, match="listed: none"):
        ensure_ready(manager, timeout_s=30, log=lambda s: None)


def test_manager_never_answering_is_not_ready(monkeypatch) -> None:
    ticks = iter(range(0, 100000, 5))
    monkeypatch.setattr(qserver_ready.time, "monotonic", lambda: float(next(ticks)))
    manager = _Manager([], answer=False)
    with pytest.raises(NotReady, match="did not answer 'status'"):
        ensure_ready(manager, timeout_s=20, log=lambda s: None)


def test_open_refused_by_the_manager_is_not_ready() -> None:
    manager = _Manager(
        [_status(exists=False)], open_reply={"success": False, "msg": "locked"}
    )
    with pytest.raises(NotReady, match="refused 'environment_open': locked"):
        ensure_ready(manager, timeout_s=30, log=lambda s: None)


def test_open_that_collapses_back_to_closed_is_not_ready() -> None:
    """A startup-profile import error: the env reads closed/idle again, for good."""
    manager = _Manager([_status(exists=False)])  # every poll: closed + idle
    with pytest.raises(NotReady, match="startup profile failed to import"):
        ensure_ready(manager, timeout_s=30, log=lambda s: None)


def test_idle_never_reached_before_deadline_is_not_ready(monkeypatch) -> None:
    ticks = iter(range(0, 100000, 5))
    monkeypatch.setattr(qserver_ready.time, "monotonic", lambda: float(next(ticks)))
    manager = _Manager(
        [
            _status(exists=False),
            _status(exists=True, manager="creating_environment", env="initializing"),
        ]
    )
    with pytest.raises(NotReady, match="not idle before the deadline"):
        ensure_ready(manager, timeout_s=20, log=lambda s: None)


def test_busy_manager_without_environment_is_not_something_open_fixes() -> None:
    manager = _Manager([_status(exists=False, manager="closing_environment")])
    with pytest.raises(NotReady, match="not a state 'environment open' can fix"):
        ensure_ready(manager, timeout_s=30, log=lambda s: None)


class TestMain:
    def test_exit_codes(self, monkeypatch) -> None:
        outcomes = {"ready": None}

        def fake_ensure(request, *, user_group, timeout_s):
            if outcomes["ready"] is None:
                raise NotReady("nope")
            return []

        monkeypatch.setattr(qserver_ready, "_zmq_request", lambda addr: object())
        monkeypatch.setattr(qserver_ready, "ensure_ready", fake_ensure)
        assert qserver_ready.main([]) == 1
        outcomes["ready"] = True
        assert (
            qserver_ready.main(["--control-addr", "tcp://x:1", "--timeout", "5"]) == 0
        )
        with pytest.raises(SystemExit) as excinfo:
            qserver_ready.main(["--timeout", "0"])
        assert excinfo.value.code == 2

    def test_missing_queueserver_is_exit_2(self, monkeypatch) -> None:
        def no_zmq(addr):
            raise ImportError("no bluesky_queueserver")

        monkeypatch.setattr(qserver_ready, "_zmq_request", no_zmq)
        assert qserver_ready.main([]) == 2

    def test_control_addr_env_default(self, monkeypatch) -> None:
        seen = {}
        monkeypatch.setenv("QS_CONTROL_ADDR", "tcp://worker:60615")
        monkeypatch.setattr(
            qserver_ready, "_zmq_request", lambda addr: seen.setdefault("addr", addr)
        )
        monkeypatch.setattr(qserver_ready, "ensure_ready", lambda *a, **k: [])
        assert qserver_ready.main([]) == 0
        assert seen["addr"] == "tcp://worker:60615"
