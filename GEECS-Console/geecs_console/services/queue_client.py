"""Queueserver manager client seam (#648): the console as a peer RE Manager client.

The console submits scans to a bluesky-queueserver RE Manager (the GEECS
worker, ``GeecsBluesky/qserver/``) instead of an in-process engine.  This
module is the one place that speaks ``bluesky-queueserver-api``:

- :class:`QueueClient` — the protocol the window and controllers depend on;
- :class:`ZmqQueueClient` — the real client (0MQ control socket, lazy
  imports so the module stays import-safe offline);
- :class:`StubQueueClient` — the offline/test default (disconnected status,
  every verb refuses with a clear message);
- :func:`read_qserver_config` — the ``[qserver]`` section of the shared
  ``~/.config/geecs_python_api/config.ini``.

Threading contract (the console's standing rules): every method here
**blocks** — 0MQ request/reply with a short timeout, or a polled
``function_execute`` task.  :meth:`QueueClient.status` is cheap and bounded
(one request, ``timeout_recv``) and is polled from a background thread
(``HealthPoller`` pattern); the submit/stop/manual-verb calls run on
``BackgroundResult`` workers.  Nothing here touches Qt.

Queue semantics this client owns (#648 item 3): on plan failure the manager
returns the failed item to the **front** of the queue (``ignore_failures``
default false), so a client that blindly add-and-starts re-runs the failed
item.  :meth:`ZmqQueueClient.submit_scan` therefore surfaces the queue's
front items to the caller (``pending_items``) and only clears them when
told to (``clear_pending=True``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: The manager's default 0MQ control / console-info ports and the document
#: stream's default out port (see GeecsBluesky/qserver/deploy/DEPLOYMENT.md,
#: "Network ports").
_CONTROL_PORT = 60615
_INFO_PORT = 60625
_DOC_PORT = 5568

#: Bound on one control-socket round trip (seconds). Chosen above the api's
#: 2 s default recv timeout so a wedged manager degrades to "disconnected"
#: instead of stacking retries.
_RECV_TIMEOUT_S = 2.0

#: How long a foreground ``function_execute`` task may run before the client
#: reports a timeout. Manual moves ride GEECS blocking sets — a slow magnet
#: legitimately takes tens of seconds.
_TASK_TIMEOUT_S = 120.0

#: How long a stop-while-running may wait for the deferred pause to land
#: before giving up. The pause waits out an in-flight blocking move (the
#: engine's pause-latency floor), so this must cover the slowest move.
_STOP_PAUSE_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class QserverConfig:
    """Client-side addresses for one GEECS RE Manager."""

    control_addr: str
    info_addr: str
    doc_addr: str


def read_qserver_config() -> Optional[QserverConfig]:
    """Read the ``[qserver]`` section of the shared GEECS config file.

    Keys: ``host`` (required — the worker machine) plus optional full
    overrides ``control_addr`` (``tcp://host:60615``), ``info_addr``
    (``tcp://host:60625``), and ``doc_addr`` (``host:5568``).  Defaults are
    derived from ``host`` with the deployment-standard ports.

    Returns
    -------
    QserverConfig or None
        ``None`` when the file or section is absent — the console then runs
        with the stub client (submission disabled, everything else works).
    """
    import configparser

    from geecs_console.services.ops_paths import USER_CONFIG_PATH

    path = USER_CONFIG_PATH.expanduser()
    if not path.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(path)
    except (OSError, configparser.Error) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
    if not parser.has_section("qserver"):
        return None
    host = parser.get("qserver", "host", fallback="").strip()
    control = parser.get("qserver", "control_addr", fallback="").strip()
    info = parser.get("qserver", "info_addr", fallback="").strip()
    doc = parser.get("qserver", "doc_addr", fallback="").strip()
    if not host and not control:
        logger.warning("[qserver] section present but no host/control_addr set")
        return None
    return QserverConfig(
        control_addr=control or f"tcp://{host}:{_CONTROL_PORT}",
        info_addr=info or f"tcp://{host}:{_INFO_PORT}",
        doc_addr=doc or f"{host}:{_DOC_PORT}",
    )


@dataclass(frozen=True)
class QueueStatus:
    """One poll's snapshot of the manager, GUI-shaped.

    ``connected=False`` means the manager did not answer — every other
    field is then meaningless and ``detail`` says why.  ``re_state`` is the
    worker RunEngine state string (``idle`` / ``running`` / ``paused`` /
    ``None`` while the worker environment is closed).
    """

    connected: bool = False
    re_state: Optional[str] = None
    manager_state: Optional[str] = None
    worker_exists: bool = False
    items_in_queue: int = 0
    running_item_uid: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of a queue submission attempt.

    ``ok=False`` with non-empty ``pending_items`` means the queue already
    held items (typically a failed item returned to the front — the #648
    item-3 trap) and nothing was submitted: the caller surfaces them and
    retries with ``clear_pending=True`` once the operator agrees.
    """

    ok: bool
    message: str = ""
    item_uid: Optional[str] = None
    pending_items: list[dict] = field(default_factory=list)


@runtime_checkable
class QueueClient(Protocol):
    """What the console needs from a RE Manager (all methods block; see module doc)."""

    def status(self) -> QueueStatus:
        """Return the manager snapshot. Never raises."""
        ...

    def submit_scan(
        self, request: dict, *, clear_pending: bool = False
    ) -> SubmitResult:
        """Queue ``geecs_scan_request_plan(request)`` and start the queue."""
        ...

    def submit_action(self, name: str) -> SubmitResult:
        """Queue ``geecs_run_action_plan(name)`` and start the queue."""
        ...

    def request_pause(self) -> tuple[bool, str]:
        """Deferred-pause the running plan. Returns ``(ok, message)``."""
        ...

    def request_resume(self) -> tuple[bool, str]:
        """Resume a paused plan. Returns ``(ok, message)``."""
        ...

    def stop_scan(self) -> tuple[bool, str]:
        """Gracefully stop: from paused directly, from running via pause."""
        ...

    def move_variable(self, name: str, value: float) -> dict:
        """Run ``geecs_move_variable`` on the worker; raises on refusal/failure."""
        ...

    def describe_action(self, name: str) -> list[dict]:
        """Run ``geecs_describe_action`` on the worker; raises on failure."""
        ...


_STUB_MESSAGE = (
    "no queueserver configured — add a [qserver] section (host = <worker>) "
    "to ~/.config/geecs_python_api/config.ini"
)


class StubQueueClient:
    """The offline default: disconnected status, every verb refuses clearly."""

    def status(self) -> QueueStatus:
        """Return a disconnected snapshot naming the missing config."""
        return QueueStatus(connected=False, detail=_STUB_MESSAGE)

    def submit_scan(
        self, request: dict, *, clear_pending: bool = False
    ) -> SubmitResult:
        """Refuse with the missing-config message."""
        return SubmitResult(ok=False, message=_STUB_MESSAGE)

    def submit_action(self, name: str) -> SubmitResult:
        """Refuse with the missing-config message."""
        return SubmitResult(ok=False, message=_STUB_MESSAGE)

    def request_pause(self) -> tuple[bool, str]:
        """Refuse with the missing-config message."""
        return False, _STUB_MESSAGE

    def request_resume(self) -> tuple[bool, str]:
        """Refuse with the missing-config message."""
        return False, _STUB_MESSAGE

    def stop_scan(self) -> tuple[bool, str]:
        """Refuse with the missing-config message."""
        return False, _STUB_MESSAGE

    def move_variable(self, name: str, value: float) -> dict:
        """Refuse with the missing-config message."""
        raise RuntimeError(_STUB_MESSAGE)

    def describe_action(self, name: str) -> list[dict]:
        """Refuse with the missing-config message."""
        raise RuntimeError(_STUB_MESSAGE)


class ZmqQueueClient:
    """The real manager client over the 0MQ control socket.

    One :class:`bluesky_queueserver_api.zmq.REManagerAPI` is created lazily
    on first use and reused (it is thread-safe for the console's usage: the
    status poller and the one-at-a-time worker calls).  All
    ``bluesky_queueserver_api`` imports live inside methods — the module
    must import offline (console standing rule).

    Parameters
    ----------
    config : QserverConfig
        The manager addresses (see :func:`read_qserver_config`).
    user : str
        Submitted-as identity recorded by the manager on every queue item.
    """

    def __init__(self, config: QserverConfig, *, user: str = "geecs-console") -> None:
        self._config = config
        self._user = user
        self._api: Any = None

    # -- plumbing -----------------------------------------------------------

    def _manager(self) -> Any:
        """Return the cached ``REManagerAPI``, creating it on first use."""
        if self._api is None:
            from bluesky_queueserver_api.zmq import REManagerAPI

            self._api = REManagerAPI(
                zmq_control_addr=self._config.control_addr,
                zmq_info_addr=self._config.info_addr,
                timeout_recv=_RECV_TIMEOUT_S,
            )
        return self._api

    def _wait_for_task(self, task_uid: str, *, timeout_s: float) -> dict:
        """Poll ``task_result`` until the task completes; return its payload.

        Raises
        ------
        RuntimeError
            Task failure (with the worker's message) or timeout.
        """
        api = self._manager()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = api.task_result(task_uid)
            status = result.get("status")
            if status == "completed":
                task = result.get("result") or {}
                if not task.get("success", False):
                    raise RuntimeError(task.get("msg") or "worker task failed")
                return task.get("return_value")
            if status not in (None, "running", "accepted", "pending"):
                raise RuntimeError(f"worker task ended with status {status!r}")
            time.sleep(0.2)
        raise RuntimeError(f"worker task did not finish within {timeout_s:.0f} s")

    # -- protocol -----------------------------------------------------------

    def status(self) -> QueueStatus:
        """Return the manager snapshot; a failed request reads as disconnected."""
        try:
            raw = self._manager().status()
        except Exception as exc:
            return QueueStatus(connected=False, detail=str(exc))
        return QueueStatus(
            connected=True,
            re_state=raw.get("re_state"),
            manager_state=raw.get("manager_state"),
            worker_exists=bool(raw.get("worker_environment_exists")),
            items_in_queue=int(raw.get("items_in_queue") or 0),
            running_item_uid=raw.get("running_item_uid") or None,
            detail="",
        )

    def _submit_item(
        self, plan_name: str, args: list, *, clear_pending: bool
    ) -> SubmitResult:
        """Shared add-and-start with the failed-item-at-front guard."""
        from bluesky_queueserver_api import BPlan

        api = self._manager()
        try:
            queue = api.queue_get()
            pending = list(queue.get("items") or [])
            if pending and not clear_pending:
                return SubmitResult(
                    ok=False,
                    message=(
                        f"{len(pending)} item(s) already queued (a failed "
                        "item returns to the queue front) — clear before "
                        "resubmitting"
                    ),
                    pending_items=pending,
                )
            if pending:
                api.queue_clear()
            item = BPlan(plan_name, *args)
            response = api.item_add(item, user=self._user)
            item_uid = (response.get("item") or {}).get("item_uid")
            api.queue_start()
            return SubmitResult(ok=True, message="queued", item_uid=item_uid)
        except Exception as exc:
            return SubmitResult(ok=False, message=str(exc))

    def submit_scan(
        self, request: dict, *, clear_pending: bool = False
    ) -> SubmitResult:
        """Queue the one scan plan with *request* as its sole argument."""
        return self._submit_item(
            "geecs_scan_request_plan", [request], clear_pending=clear_pending
        )

    def submit_action(self, name: str) -> SubmitResult:
        """Queue the on-demand action plan (actions are queue items, decision 2)."""
        return self._submit_item("geecs_run_action_plan", [name], clear_pending=False)

    def request_pause(self) -> tuple[bool, str]:
        """Deferred pause (the stock verb; hard pause replays — never default it)."""
        try:
            self._manager().re_pause(option="deferred")
            return True, "pause requested"
        except Exception as exc:
            return False, str(exc)

    def request_resume(self) -> tuple[bool, str]:
        """Resume from the paused state (replays nothing on the deferred verb)."""
        try:
            self._manager().re_resume()
            return True, "resumed"
        except Exception as exc:
            return False, str(exc)

    def stop_scan(self) -> tuple[bool, str]:
        """Gracefully stop the current plan, preserving partial data.

        From ``paused``: ``re_stop`` directly (the live-verified
        stop-from-paused path — Scan003, 2026-08-21).  From ``running``:
        the manager only accepts stop while paused, so this sequences
        deferred-pause → wait for ``paused`` → ``re_stop``.  The wait is
        bounded but long (a deferred pause waits out an in-flight blocking
        move).  Also stops the queue so no follow-on item auto-starts.
        """
        api = self._manager()
        try:
            api.queue_stop()
        except Exception:  # queue may be empty/stopped already — irrelevant
            logger.debug("queue_stop refused (ignored)", exc_info=True)
        try:
            state = self.status()
            if not state.connected:
                return False, f"manager unreachable: {state.detail}"
            if state.re_state == "paused":
                api.re_stop()
                return True, "stop requested (from paused)"
            if state.re_state != "running":
                return False, f"nothing to stop (RE state: {state.re_state})"
            api.re_pause(option="deferred")
            deadline = time.monotonic() + _STOP_PAUSE_TIMEOUT_S
            while time.monotonic() < deadline:
                st = self.status()
                if st.connected and st.re_state == "paused":
                    api.re_stop()
                    return True, "stop requested (paused, then stopped)"
                if st.connected and st.re_state == "idle":
                    # The plan finished on its own while we waited.
                    return True, "scan already finished"
                time.sleep(0.5)
            return False, (
                "pause did not land within "
                f"{_STOP_PAUSE_TIMEOUT_S:.0f} s — scan still running"
            )
        except Exception as exc:
            return False, str(exc)

    def move_variable(self, name: str, value: float) -> dict:
        """Run the worker's manual move; idle-only (the manager enforces it)."""
        from bluesky_queueserver_api import BFunc

        api = self._manager()
        response = api.function_execute(
            BFunc("geecs_move_variable", name, value), user=self._user
        )
        return self._wait_for_task(response["task_uid"], timeout_s=_TASK_TIMEOUT_S)

    def describe_action(self, name: str) -> list[dict]:
        """Dry-run a named action against the worker's configs; idle-only."""
        from bluesky_queueserver_api import BFunc

        api = self._manager()
        response = api.function_execute(
            BFunc("geecs_describe_action", name), user=self._user
        )
        return self._wait_for_task(response["task_uid"], timeout_s=30.0)


def make_queue_client(user: str = "geecs-console") -> QueueClient:
    """Build the configured client, or the stub when no ``[qserver]`` exists."""
    config = read_qserver_config()
    if config is None:
        return StubQueueClient()
    return ZmqQueueClient(config, user=user)
