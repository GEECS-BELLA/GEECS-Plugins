"""The queue client's environment verbs: close, open (each waits until settled), queue start.

No manager: the real client over the recording fake of
``test_queue_client``, its status answers scripted poll by poll.
"""

from __future__ import annotations

import pytest

pytest.importorskip("bluesky_queueserver_api")

from geecs_bluesky.qs_client import StubQueueClient, ZmqQueueClient

from .test_queue_client import _client, _FakeManagerAPI


def _env(exists: bool, manager: str = "idle", worker: str | None = None) -> dict:
    return {
        "re_state": "idle" if exists else None,
        "manager_state": manager,
        "worker_environment_exists": exists,
        "worker_environment_state": worker or ("idle" if exists else "closed"),
        "items_in_queue": 0,
    }


class _EnvFake(_FakeManagerAPI):
    def environment_close(self) -> None:
        self.calls.append(("environment_close",))

    def environment_open(self) -> None:
        self.calls.append(("environment_open",))


def _env_client(fake: _EnvFake) -> ZmqQueueClient:
    client = _client(fake)
    client._environment_poll_s = 0
    return client


def test_close_waits_until_the_manager_is_idle_with_no_environment():
    fake = _EnvFake()
    fake.status_payloads = [
        _env(True, "closing_environment", "closing"),
        _env(False, "closing_environment", "closing"),
        _env(False),
    ]
    ok, message = _env_client(fake).environment_close()
    assert ok, message
    assert fake.calls[0] == ("environment_close",)
    assert fake.status_payloads == []  # every intermediate state was waited out


class _Manager:
    """The readiness unit's transport, scripted: ``(method, params) -> (msg, err)``; a failure is ``(None, err)``."""

    def __init__(self, statuses: list[dict], plan_lists: list[list[str]], restore: tuple | None = None) -> None:  # fmt: skip
        self.statuses, self.plan_lists, self.restore = statuses, plan_lists, restore
        self.calls: list[str] = []

    def __call__(self, method: str, params: dict | None) -> tuple[dict | None, str]:
        self.calls.append(method)
        if method == "status":
            return (
                self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            ), ""
        if method == "plans_allowed":
            names = (
                self.plan_lists.pop(0)
                if len(self.plan_lists) > 1
                else self.plan_lists[0]
            )
            return {"success": True, "plans_allowed": {n: {} for n in names}}, ""
        if method == "permissions_reload" and self.restore is not None:
            return self.restore
        return {"success": True, "msg": ""}, ""


@pytest.fixture
def transport(monkeypatch):
    """The open goes over ready.zmq_request to the client's own control address (its per-method budgets)."""
    from geecs_bluesky.qs_client import ready

    monkeypatch.setattr(ready, "POLL_S", 0)
    seen: dict = {}

    def install(manager: _Manager) -> dict:
        def zmq_request(addr: str):
            seen["addr"] = addr
            return manager

        monkeypatch.setattr(ready, "zmq_request", zmq_request)
        return seen

    return install


def test_open_waits_for_the_plan_list_not_merely_the_environment(transport):
    """#838: the environment reads up before its plan list lands; ready is the plan list too."""
    from geecs_bluesky.plan_names import GEECS_PLAN_NAMES

    manager = _Manager(
        [_env(False), _env(False, "creating_environment", "initializing"), _env(True)],
        [[], list(GEECS_PLAN_NAMES)],
    )
    seen = transport(manager)
    ok, message = _client(_EnvFake()).environment_open()
    assert ok, message
    assert seen["addr"] == "tcp://x:1"  # the client's control address
    assert manager.calls.count("environment_open") == 1
    assert manager.calls.count("plans_allowed") == 2  # re-read until settled


def test_a_plan_list_that_never_lands_is_restored_once_from_disk(transport):
    from geecs_bluesky.plan_names import GEECS_PLAN_NAMES

    manager = _Manager([_env(False), _env(True)], [[]] * 6 + [list(GEECS_PLAN_NAMES)])
    transport(manager)
    ok, message = _client(_EnvFake()).environment_open()
    assert ok, message
    assert manager.calls.count("permissions_reload") == 1


def test_a_restore_that_times_out_is_a_failure_never_a_raise(transport):
    manager = _Manager([_env(True)], [[]], restore=(None, "timeout during receive"))
    transport(manager)
    ok, message = _client(_EnvFake()).environment_open()
    assert not ok and "restore from disk failed" in message


def test_an_open_that_settles_closed_names_the_startup_profile(transport):
    transport(_Manager([_env(False)], [[]]))
    ok, message = _client(_EnvFake()).environment_open()
    assert not ok and "startup profile" in message


def test_a_refused_request_fails_and_waits_for_nothing():
    class Refusing(_EnvFake):
        def environment_close(self) -> None:
            raise RuntimeError("Environment does not exist")

    fake = Refusing()
    ok, message = _env_client(fake).environment_close()
    assert not ok and "does not exist" in message
    assert ("status",) not in fake.calls


def test_the_wait_is_bounded():
    fake = _EnvFake()
    fake.status_payloads = [_env(True, "closing_environment", "closing")] * 1000
    ok, message = _env_client(fake).environment_close(timeout=0)
    assert not ok and "not settled" in message


def test_queue_start_and_the_stub():
    fake = _EnvFake()
    assert _client(fake).queue_start() == (True, "queue started")
    assert fake.calls == [("queue_start",)]
    stub = StubQueueClient()
    assert not stub.environment_close()[0]
    assert not stub.environment_open()[0]
    assert not stub.queue_start()[0]
