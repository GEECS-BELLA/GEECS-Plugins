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


def test_open_waits_for_the_worker_idle_not_merely_existing():
    fake = _EnvFake()
    fake.status_payloads = [
        _env(False),  # a poll before the manager took the request
        _env(False, "creating_environment", "initializing"),
        _env(True, "creating_environment", "initializing"),
        _env(True, "idle", "initializing"),
        _env(True),
    ]
    ok, message = _env_client(fake).environment_open()
    assert ok, message
    assert fake.calls[0] == ("environment_open",)
    assert fake.status_payloads == []


def test_an_open_that_settles_closed_names_the_startup_profile():
    fake = _EnvFake()
    fake.status_payloads = [_env(False)] * 5
    ok, message = _env_client(fake).environment_open()
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
