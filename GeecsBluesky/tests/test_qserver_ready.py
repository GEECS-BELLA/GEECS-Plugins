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
        self,
        statuses,
        *,
        plans=GEECS_PLAN_NAMES,
        open_reply=None,
        answer=True,
        plans_answer=True,
    ):
        self.statuses = list(statuses)
        self.plans = list(plans)
        self.open_reply = open_reply or {"success": True}
        self.answer = answer
        self.plans_answer = plans_answer
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
            if not self.plans_answer:
                return None, "timeout"
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
    assert "permissions file" in message
    assert "user_group='primary'" in message


def test_empty_plan_list_after_open_is_not_ready() -> None:
    """The exact #793 invariant: open 'succeeded', plan list still empty."""
    manager = _Manager([_status(exists=False), _status(exists=True)], plans=[])
    with pytest.raises(NotReady, match="lists no allowed plans"):
        ensure_ready(manager, timeout_s=30, log=lambda s: None)


def test_unanswered_plan_list_is_not_ready() -> None:
    """Shared-verdict rule: env up but plans_allowed unanswered → NOT READY."""
    manager = _Manager([_status(exists=True)], plans_answer=False)
    log = []
    with pytest.raises(NotReady, match="could not be read"):
        ensure_ready(manager, timeout_s=30, log=log.append)
    assert any("plans_allowed unanswered" in line for line in log)


def test_verdict_is_the_shared_one() -> None:
    """ensure_ready's sentence IS qs_client.readiness_verdict's — one definition."""
    from geecs_bluesky.qs_client.client import QueueStatus, readiness_verdict

    manager = _Manager([_status(exists=True)], plans=["geecs_run_action_plan"])
    with pytest.raises(NotReady) as excinfo:
        ensure_ready(manager, timeout_s=30, log=lambda s: None)
    expected = readiness_verdict(
        QueueStatus(connected=True, worker_exists=True),
        {"geecs_run_action_plan": {}},
        list(GEECS_PLAN_NAMES),
    ).detail
    assert str(excinfo.value).startswith(expected)


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


def test_manager_executing_a_plan_is_ready_too() -> None:
    """A by-hand re-run mid-scan must not report NOT READY: env up, plans served."""
    manager = _Manager(
        [
            _status(
                exists=True,
                manager="executing_queue",
                env="executing_plan",
                re_state="running",
            )
        ]
    )
    ensure_ready(manager, timeout_s=30, log=lambda s: None)
    assert "environment_open" not in [m for m, _ in manager.calls]


def test_environment_never_up_before_deadline_is_not_ready(monkeypatch) -> None:
    ticks = iter(range(0, 100000, 5))
    monkeypatch.setattr(qserver_ready.time, "monotonic", lambda: float(next(ticks)))
    manager = _Manager(
        [
            _status(exists=False),
            _status(exists=True, manager="creating_environment", env="initializing"),
        ]
    )
    with pytest.raises(NotReady, match="not up before the deadline"):
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

    def test_control_addr_precedence(self, monkeypatch) -> None:
        """flag > QS_CONTROL_ADDR > [qserver] control_addr/host > loopback."""
        from geecs_bluesky.qs_client.client import QserverConfig

        seen = {}
        monkeypatch.setattr(
            qserver_ready, "_zmq_request", lambda addr: seen.__setitem__("addr", addr)
        )
        monkeypatch.setattr(qserver_ready, "ensure_ready", lambda *a, **k: [])
        monkeypatch.delenv("QS_CONTROL_ADDR", raising=False)

        monkeypatch.setattr(qserver_ready, "read_qserver_config", lambda: None)
        assert qserver_ready.main([]) == 0
        assert seen["addr"] == qserver_ready.DEFAULT_CONTROL_ADDR

        monkeypatch.setattr(
            qserver_ready,
            "read_qserver_config",
            lambda: QserverConfig(
                "tcp://192.168.6.14:60615", "tcp://h:60625", "h:5568"
            ),
        )
        assert qserver_ready.main([]) == 0
        assert seen["addr"] == "tcp://192.168.6.14:60615"

        monkeypatch.setenv("QS_CONTROL_ADDR", "tcp://worker:60615")
        assert qserver_ready.main([]) == 0
        assert seen["addr"] == "tcp://worker:60615"

        assert qserver_ready.main(["--control-addr", "tcp://flag:1"]) == 0
        assert seen["addr"] == "tcp://flag:1"
