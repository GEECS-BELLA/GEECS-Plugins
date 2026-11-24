"""``FakeManagerAPI``: a recording stand-in for the RE Manager's ``REManagerAPI``.

Test code only. The real :class:`~geecs_bluesky.qs_client.client.ZmqQueueClient`
runs over it through its ``_api`` cache (the documented injection point),
so a test drives the client without a manager or a network. Nothing here
imports pytest or the queueserver API.
"""

from __future__ import annotations

from typing import Any

from geecs_bluesky.qs_client.client import QserverConfig, ZmqQueueClient

#: The plan list ``plans_allowed`` answers unless a test sets its own.
DEFAULT_PLANS: dict[str, dict] = {"scan": {"name": "scan"}, "count": {"name": "count"}}


class FakeManagerAPI:
    """Record every call as a tuple in ``calls``; answer from scripted state.

    ``status`` pops ``status_payloads`` one per call (an idle manager once
    they run out) or raises ``raise_on_status``; ``plans_allowed`` answers
    ``plans`` or raises ``plans_error``; the queue verbs keep
    ``queue_items`` and ``running``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.status_payloads: list[dict] = []
        self.queue_items: list[dict] = []
        self.running: dict | None = None
        self.raise_on_status: Exception | None = None
        self.plans: dict | None = dict(DEFAULT_PLANS)
        self.plans_error: Exception | None = None

    def status(self) -> dict:
        """The next scripted status payload."""
        self.calls.append(("status",))
        if self.raise_on_status is not None:
            raise self.raise_on_status
        if self.status_payloads:
            return self.status_payloads.pop(0)
        return {
            "re_state": "idle",
            "manager_state": "idle",
            "worker_environment_exists": True,
            "items_in_queue": len(self.queue_items),
        }

    def queue_get(self) -> dict:
        """The queue and the running item."""
        self.calls.append(("queue_get",))
        return {
            "items": list(self.queue_items),
            "running_item": dict(self.running) if self.running else {},
        }

    def queue_clear(self) -> None:
        """Empty the queue."""
        self.calls.append(("queue_clear",))
        self.queue_items = []

    def item_add(self, item: Any, *, user: str | None = None) -> dict:  # Any: a BPlan
        """Record the item (as its dict) and answer a fixed uid."""
        self.calls.append(("item_add", item.to_dict(), user))
        return {"item": {"item_uid": "uid-1"}}

    def queue_start(self) -> None:
        """Record a start."""
        self.calls.append(("queue_start",))

    def item_remove(self, *, uid: str | None = None, pos: object = None) -> None:
        """Drop the item with *uid* from the queue."""
        self.calls.append(("item_remove", uid))
        self.queue_items = [i for i in self.queue_items if i["item_uid"] != uid]

    def queue_stop(self) -> None:
        """Record a queue stop."""
        self.calls.append(("queue_stop",))

    def re_pause(self, option: str | None = None) -> None:
        """Record a pause and its option."""
        self.calls.append(("re_pause", option))

    def re_resume(self) -> None:
        """Record a resume."""
        self.calls.append(("re_resume",))

    def re_stop(self) -> None:
        """Record a stop."""
        self.calls.append(("re_stop",))

    def plans_allowed(
        self, *, reload: bool = False, user_group: str | None = None
    ) -> dict:
        """The scripted plan list."""
        self.calls.append(("plans_allowed",))
        if self.plans_error is not None:
            raise self.plans_error
        return {"success": True, "plans_allowed": self.plans}

    def devices_allowed(
        self, *, reload: bool = False, user_group: str | None = None
    ) -> dict:
        """A two-device tree."""
        self.calls.append(("devices_allowed",))
        return {
            "success": True,
            "devices_allowed": {
                "U_S1H": {"components": {"current": {}, "scalars": {}}},
                "UC_Cam": {"components": {"scalars": {}}},
            },
        }

    def environment_close(self) -> None:
        """Record an environment close."""
        self.calls.append(("environment_close",))

    def environment_open(self) -> None:
        """Record an environment open."""
        self.calls.append(("environment_open",))

    def close(self) -> None:
        """Record the client's close."""
        self.calls.append(("close",))


def client_over(api: object, *, user: str | None = None) -> ZmqQueueClient:
    """A ``ZmqQueueClient`` over *api* in place of a real ``REManagerAPI``."""
    config = QserverConfig("tcp://x:1", "tcp://x:2", "x:3")
    client = (
        ZmqQueueClient(config) if user is None else ZmqQueueClient(config, user=user)
    )
    client._api = api
    return client
