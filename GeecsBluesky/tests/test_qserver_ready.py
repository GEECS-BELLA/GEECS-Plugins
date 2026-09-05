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
        plans_sequence=None,
        open_reply=None,
        answer=True,
        plans_answer=True,
    ):
        self.statuses = list(statuses)
        self.plans = list(plans)
        # Successive plans_allowed replies (the last one repeats) — the
        # manager's plan list landing *after* the environment reads up.
        self.plans_sequence = [list(p) for p in (plans_sequence or [])]
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
            if self.plans_sequence:
                head = self.plans_sequence[0]
                if len(self.plans_sequence) > 1:
                    self.plans_sequence.pop(0)
                return _plans(head), ""
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
    """The exact #793 invariant: open 'succeeded', plan list still empty.

    The fake stays empty across the post-open settle re-reads, so the
    conclusion is reached only after the settle window is exhausted.
    """
    manager = _Manager([_status(exists=False), _status(exists=True)], plans=[])
    log = []
    with pytest.raises(NotReady, match="lists no allowed plans"):
        ensure_ready(manager, timeout_s=30, log=log.append)
    reads = [m for m, _ in manager.calls].count("plans_allowed")
    assert reads == qserver_ready.PLAN_LIST_SETTLE_POLLS + 1
    assert (
        sum("re-reading" in line for line in log)
        == qserver_ready.PLAN_LIST_SETTLE_POLLS
    )


def test_plan_list_landing_after_the_open_is_ready() -> None:
    """The manager reports the env up before its plan-list download lands.

    A fresh manager reads an EMPTY list in that window and a deploy that
    adds a plan reads the STALE one; both settle within a few polls.
    """
    manager = _Manager(
        [_status(exists=False), _status(exists=True)],
        plans_sequence=[[], ["geecs_run_action_plan"], GEECS_PLAN_NAMES],
    )
    log = []
    allowed = ensure_ready(manager, timeout_s=30, log=log.append)
    assert allowed == sorted(GEECS_PLAN_NAMES)
    assert [m for m, _ in manager.calls].count("plans_allowed") == 3
    assert any("re-reading" in line for line in log)
    assert log[-1].startswith("ready: 5 allowed plans")


def test_no_settle_window_without_our_open() -> None:
    """An environment someone else opened is judged on the first read."""
    manager = _Manager([_status(exists=True)], plans=[])
    with pytest.raises(NotReady, match="lists no allowed plans"):
        ensure_ready(manager, timeout_s=30, log=lambda s: None)
    assert [m for m, _ in manager.calls].count("plans_allowed") == 1


def test_unanswered_plan_list_after_open_is_not_retried() -> None:
    """plans_unknown is never a settling state — unanswered is not ready."""
    manager = _Manager(
        [_status(exists=False), _status(exists=True)], plans_answer=False
    )
    with pytest.raises(NotReady, match="could not be read"):
        ensure_ready(manager, timeout_s=30, log=lambda s: None)
    assert [m for m, _ in manager.calls].count("plans_allowed") == 1


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


def test_closing_environment_is_waited_out_then_opened() -> None:
    """Unit started mid-`environment close`: wait for closed+idle, then open."""
    manager = _Manager(
        [
            _status(exists=False, manager="closing_environment", env="closing"),
            _status(exists=False, manager="closing_environment", env="closing"),
            _status(exists=False),  # settled: closed + idle → ours to open
            _status(exists=True),
        ]
    )
    ensure_ready(manager, timeout_s=30, log=lambda s: None)
    methods = [m for m, _ in manager.calls]
    assert methods.count("environment_open") == 1
    # the open came after the closing polls, not on the first snapshot
    assert methods.index("environment_open") > 2


def test_environment_closed_by_hand_mid_run_is_reopened_once() -> None:
    """`qserver environment close` after startup: the next run opens exactly once."""
    manager = _Manager([_status(exists=False), _status(exists=True)])
    ensure_ready(manager, timeout_s=30, log=lambda s: None)
    assert [m for m, _ in manager.calls].count("environment_open") == 1


def test_busy_manager_without_environment_waits_until_the_deadline(
    monkeypatch,
) -> None:
    ticks = iter(range(0, 100000, 5))
    monkeypatch.setattr(qserver_ready.time, "monotonic", lambda: float(next(ticks)))
    manager = _Manager([_status(exists=False, manager="closing_environment")])
    with pytest.raises(NotReady, match="not up before the deadline"):
        ensure_ready(manager, timeout_s=20, log=lambda s: None)
    assert "environment_open" not in [m for m, _ in manager.calls]


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
        """flag > QS_CONTROL_ADDR > loopback — the client-side config is ignored.

        The unit asserts the manager on THIS host; a service account's
        ``config.ini`` naming another worker (the client-side address)
        must never redirect it.
        """
        from geecs_bluesky.qs_client import client as qs_client
        from geecs_bluesky.qs_client.client import QserverConfig

        seen = {}
        monkeypatch.setattr(
            qserver_ready, "_zmq_request", lambda addr: seen.__setitem__("addr", addr)
        )
        monkeypatch.setattr(qserver_ready, "ensure_ready", lambda *a, **k: [])
        monkeypatch.delenv("QS_CONTROL_ADDR", raising=False)
        monkeypatch.setattr(
            qs_client,
            "read_qserver_config",
            lambda: QserverConfig("tcp://other-box:60615", "tcp://h:60625", "h:5568"),
        )

        assert qserver_ready.main([]) == 0
        assert seen["addr"] == qserver_ready.DEFAULT_CONTROL_ADDR
        assert seen["addr"].startswith("tcp://localhost:")

        monkeypatch.setenv("QS_CONTROL_ADDR", "tcp://worker:60615")
        assert qserver_ready.main([]) == 0
        assert seen["addr"] == "tcp://worker:60615"

        assert qserver_ready.main(["--control-addr", "tcp://flag:1"]) == 0
        assert seen["addr"] == "tcp://flag:1"

    def test_default_addr_port_is_the_client_modules(self) -> None:
        from geecs_bluesky.qs_client.client import _CONTROL_PORT

        assert qserver_ready.DEFAULT_CONTROL_ADDR.endswith(f":{_CONTROL_PORT}")
