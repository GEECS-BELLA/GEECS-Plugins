"""The submission seam: what the console needs from the scan service.

Since the queueserver migration (#648) the console is a **peer client of
the RE Manager** — scans are queue items executed by the GEECS worker
(``GeecsBluesky/qserver/``), not an in-process engine.  :class:`Submitter`
is the protocol the window and controllers depend on;
:class:`QueueSubmitter` implements it over the
:class:`~geecs_console.services.queue_client.QueueClient`, and
:func:`make_queue_submitter` is the default factory (offline it wraps the
stub client, whose every verb refuses with the missing-``[qserver]``
message).

Scan *state* is deliberately not on this protocol: the window observes it
from the manager status poll and the document stream
(:class:`~geecs_console.app.scan_monitor.ScanMonitorController`), never by
asking the submitter.  The dropped pause-window action flow
(``request_action_during_scan``, decision 2 of the migration) has no
successor member — actions run as ordinary queue items when the manager is
idle.

Threading contract: every member here **blocks** (0MQ round trips; manual
moves poll a worker task to completion), so the window and controllers
dispatch them through their ``BackgroundResult`` workers — with two
deliberate exceptions, :meth:`Submitter.request_pause` and
:meth:`Submitter.request_resume`, which are single short-timeout requests
the window calls directly (the old prompt-returning pause semantics).
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from geecs_console.services.queue_client import (
    QueueClient,
    QueueStatus,
    SubmitResult,
)


@runtime_checkable
class Submitter(Protocol):
    """The scan-service surface the console submits through."""

    def submit(self, request: dict, *, clear_pending: bool = False) -> SubmitResult:
        """Queue one ``ScanRequest`` dict and start the queue.

        ``ok=False`` with ``pending_items`` means the queue already held
        items (the failed-item-at-front trap) and nothing was submitted —
        the window asks the operator, then retries with
        ``clear_pending=True``.
        """
        ...

    def stop_scan(self) -> tuple[bool, str]:
        """Gracefully stop the current scan (partial data preserved).

        From paused: stop directly.  From running: sequenced deferred
        pause → stop, waiting out an in-flight blocking move — may take
        tens of seconds, so the window dispatches it on the stop worker.
        """
        ...

    def request_pause(self) -> tuple[bool, str]:
        """Deferred-pause the running scan (returns promptly)."""
        ...

    def request_resume(self) -> tuple[bool, str]:
        """Resume a paused scan — replays nothing (returns promptly)."""
        ...

    def run_action(self, name: str) -> None:
        """Queue action plan *name* as its own item (idle manager only).

        Raises ``RuntimeError`` with an operator-readable message when the
        submission is refused (manager unreachable, queue busy).
        Completion/failure of the action itself is observed through the
        manager status, not this call.
        """
        ...

    def describe_action(self, name: str) -> list[dict]:
        """Dry-run action plan *name* against the worker's configs.

        Blocking (a worker ``function_execute`` task; idle manager only).
        Returns one dict per step in execution order, with keys ``kind``,
        ``device``, ``variable``, ``value``, ``wait_s``, ``from_plan``.
        Raises ``RuntimeError`` with an operator-readable message on
        refusal or failure.
        """
        ...

    def move_variable(self, name: str, value: float) -> dict:
        """Move one catalog scan variable (or raw ``Device:Variable``) now.

        Runs the worker's ``geecs_move_variable`` (scan-identical
        completion semantics; idle manager only) and blocks until the move
        lands — dispatch off the GUI thread.  Returns ``{"variable",
        "kind", "value", "targets"}``; raises ``RuntimeError`` with the
        worker's refusal/failure message (e.g. the engine's exact
        ``"scan in progress — move not started"``).
        """
        ...

    def status(self) -> QueueStatus:
        """One manager status snapshot (never raises) — the poller's probe."""
        ...


class QueueSubmitter:
    """The real :class:`Submitter`: a thin adapter over the queue client.

    Also carries the stream addresses
    (:attr:`info_addr` / :attr:`doc_addr`, ``None`` when unconfigured) so
    the window can build its
    :class:`~geecs_console.app.scan_monitor.ScanMonitorController` from the
    same configuration in one place.
    """

    def __init__(
        self,
        client: QueueClient,
        *,
        info_addr: Optional[str] = None,
        doc_addr: Optional[str] = None,
    ) -> None:
        self.client = client
        self.info_addr = info_addr
        self.doc_addr = doc_addr

    def submit(self, request: dict, *, clear_pending: bool = False) -> SubmitResult:
        """Queue the scan request (see :class:`Submitter`)."""
        return self.client.submit_scan(request, clear_pending=clear_pending)

    def stop_scan(self) -> tuple[bool, str]:
        """Gracefully stop the current scan (see :class:`Submitter`)."""
        return self.client.stop_scan()

    def request_pause(self) -> tuple[bool, str]:
        """Deferred-pause the running scan."""
        return self.client.request_pause()

    def request_resume(self) -> tuple[bool, str]:
        """Resume a paused scan."""
        return self.client.request_resume()

    def run_action(self, name: str) -> None:
        """Queue the action item; raise the refusal message on failure."""
        result = self.client.submit_action(name)
        if not result.ok:
            if result.pending_items:
                raise RuntimeError(
                    f"queue not empty ({len(result.pending_items)} item(s) "
                    "pending) — clear the queue before running an action"
                )
            raise RuntimeError(result.message or "action submission refused")

    def describe_action(self, name: str) -> list[dict]:
        """Worker-side dry run (see :class:`Submitter`)."""
        return self.client.describe_action(name)

    def move_variable(self, name: str, value: float) -> dict:
        """Worker-side manual move (see :class:`Submitter`)."""
        return self.client.move_variable(name, value)

    def status(self) -> QueueStatus:
        """One manager status snapshot (never raises)."""
        return self.client.status()


def make_queue_submitter(experiment: str = "") -> QueueSubmitter:
    """Build the configured submitter (stub-backed when no ``[qserver]``).

    Parameters
    ----------
    experiment : str, optional
        The selected experiment.  Currently informational — one manager
        serves one experiment by deployment contract (``QS_EXPERIMENT``),
        and the ``[qserver]`` config names that manager; a mismatch
        surfaces as the worker refusing the request's names at validation.

    Returns
    -------
    QueueSubmitter
        Ready to use; unconfigured installs get the stub client (every
        verb refuses with the missing-config message) and no stream
        addresses.
    """
    from geecs_console.services.queue_client import (
        StubQueueClient,
        ZmqQueueClient,
        read_qserver_config,
    )

    config = read_qserver_config()
    if config is None:
        return QueueSubmitter(StubQueueClient())
    return QueueSubmitter(
        ZmqQueueClient(config),
        info_addr=config.info_addr,
        doc_addr=config.doc_addr,
    )
