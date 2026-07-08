"""Typed scan event vocabulary: the ScanEvent hierarchy, ScanState, DialogRequest.

This is the engine's event vocabulary (target-architecture vision §2): every
state transition, step boundary, and operator question a scan engine emits is
one of these types, delivered through an ``on_event`` callback injected by the
consumer.  Front-ends (the Scanner GUI, headless scripts, tests) import these
types *from the engine*; ``geecs_scanner.engine.scan_events`` and
``geecs_scanner.engine.dialog_request`` are re-export shims kept for
compatibility.

Historically these types lived in ``geecs_scanner`` (a front-end package),
forcing the engine to import its own vocabulary defensively via try/except.
They moved here verbatim — semantics unchanged — so the engine owns its
vocabulary and front-ends import from below.

The event sequence for a successful 1D scan looks like::

    ScanLifecycleEvent(INITIALIZING, total_shots=N)
    ScanLifecycleEvent(RUNNING)
    ScanStepEvent(step_index=0, total_steps=M, shots_completed=0,  phase="started")
    DeviceCommandEvent(device="X", variable="Y", outcome="accepted")
    ScanStepEvent(step_index=0, total_steps=M, shots_completed=k,  phase="completed")
    ...
    ScanLifecycleEvent(DONE)
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

__all__ = [
    "ScanState",
    "ScanEvent",
    "ScanLifecycleEvent",
    "ScanStepEvent",
    "DeviceCommandEvent",
    "ScanErrorEvent",
    "ScanRestoreFailedEvent",
    "ScanDialogEvent",
    "DialogRequest",
]


# ---------------------------------------------------------------------------
# ScanState
# ---------------------------------------------------------------------------


class ScanState(str, enum.Enum):
    """Lifecycle state of a scan.

    Inherits ``str`` so values serialise naturally to JSON / log messages.
    """

    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED_ON_ERROR = "paused_on_error"
    STOPPING = "stopping"
    DONE = "done"
    ABORTED = "aborted"


# ---------------------------------------------------------------------------
# Dialog request (worker → consumer operator question)
# ---------------------------------------------------------------------------


@dataclass
class DialogRequest:
    """Carries an operator question across the worker→main-thread boundary.

    Worker threads that need an operator decision create a
    :class:`DialogRequest`, emit it inside a :class:`ScanDialogEvent` through
    the ``on_event`` callback, and block on ``response_event.wait()``.  The
    consumer (e.g. the GUI, on the Qt main thread) shows the dialog and
    answers by writing ``abort[0]`` and setting ``response_event``.

    Parameters
    ----------
    exc :
        The exception that triggered the dialog.  For requests carrying
        their own ``title``/labels, ``str(exc)`` is the full operator-facing
        dialog body.
    context :
        Optional extra information shown in the dialog body — e.g. the full
        list of variables being set for a device when the error occurred.
    title :
        Optional dialog window title.  ``None`` (legacy device-command
        requests) keeps the title derived from the exception type.
    continue_label :
        Optional text for the non-abort button (e.g. ``"Drop && Continue"``).
        ``None`` keeps the default ``"Continue"``.  When set, the dialog body
        is ``str(exc)`` verbatim — the request author owns the wording of
        what "continue" means.
    abort_label :
        Optional text for the abort button.  ``None`` keeps ``"Abort"``.
    response_event :
        Set by the consumer once the user has responded.
    abort :
        Single-element list used as a mutable result container.
        ``True`` → user chose Abort; ``False`` → user chose Continue.
    """

    exc: Exception
    context: Optional[str] = None
    title: Optional[str] = None
    continue_label: Optional[str] = None
    abort_label: Optional[str] = None
    response_event: threading.Event = field(default_factory=threading.Event)
    abort: list[bool] = field(default_factory=lambda: [False])


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------


@dataclass
class ScanEvent:
    """Base class for all scan events.

    Parameters
    ----------
    timestamp :
        Wall-clock time of the event in seconds since the epoch
        (``time.time()``).  Set automatically if not provided.
    """

    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@dataclass
class ScanLifecycleEvent(ScanEvent):
    """Emitted at every :class:`ScanState` transition.

    Parameters
    ----------
    state :
        The state transitioned *to*.
    total_shots :
        Total number of shots in the scan (``shots_per_step × steps``).
        Non-zero only on the ``INITIALIZING`` event; zero on all others.
        Consumers should cache this value on the ``INITIALIZING`` event and
        use it to compute progress from subsequent :class:`ScanStepEvent` s.
    """

    state: ScanState = ScanState.IDLE
    total_shots: int = 0


# ---------------------------------------------------------------------------
# Step-level progress
# ---------------------------------------------------------------------------


@dataclass
class ScanStepEvent(ScanEvent):
    """Emitted at the start and completion of each scan step.

    Parameters
    ----------
    step_index :
        Zero-based index of the current step.
    total_steps :
        Total number of steps in the scan.
    shots_completed :
        Cumulative shots logged across all completed steps so far.
        Together with ``ScanLifecycleEvent.total_shots`` this lets
        consumers compute a shot-level progress fraction without polling.
    phase :
        ``"started"`` fires before the device set; ``"completed"`` fires
        after acquisition for that step has finished.
    """

    step_index: int = 0
    total_steps: int = 0
    shots_completed: int = 0
    phase: Literal["started", "completed"] = "started"


# ---------------------------------------------------------------------------
# Device commands
# ---------------------------------------------------------------------------


@dataclass
class DeviceCommandEvent(ScanEvent):
    """Emitted for every device command issued by the scan engine.

    Parameters
    ----------
    device :
        Device name (e.g. ``"U_ESP_JetXYZ"``).
    variable :
        Variable name (e.g. ``"Position.Axis 2"``).
    outcome :
        Result of the command attempt.

        - ``"sent"``     — command dispatched; response not yet known.
        - ``"accepted"`` — device acknowledged the command.
        - ``"rejected"`` — command rejected (transient comms).
        - ``"failed"``   — hardware error.
        - ``"timeout"``  — device did not respond in time.
    """

    device: str = ""
    variable: str = ""
    outcome: Literal["sent", "accepted", "rejected", "failed", "timeout"] = "sent"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@dataclass
class ScanErrorEvent(ScanEvent):
    """Emitted when the engine encounters a recoverable or fatal error.

    Parameters
    ----------
    message :
        Human-readable description of what went wrong.
    recoverable :
        ``True`` if the scan can continue (e.g. a skipped step);
        ``False`` if the scan is about to abort.
    exc :
        Originating exception, if available.
    """

    message: str = ""
    recoverable: bool = True
    exc: Optional[BaseException] = None


# ---------------------------------------------------------------------------
# Restore failures
# ---------------------------------------------------------------------------


@dataclass
class ScanRestoreFailedEvent(ScanEvent):
    """Emitted once per device that could not be restored after a scan.

    The GUI accumulates these and shows a single warning dialog on the
    ``ScanLifecycleEvent(DONE)`` transition.

    Parameters
    ----------
    device :
        Device name that failed to restore.
    message :
        Description of the failure (exception message or context).
    """

    device: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# Dialog events
# ---------------------------------------------------------------------------


@dataclass
class ScanDialogEvent(ScanEvent):
    """Carries an operator dialog request across the worker→consumer boundary.

    The GUI callback receives this event and posts it to the Qt main thread
    via ``QMetaObject.invokeMethod``; headless consumers may answer inline or
    ignore it (the worker's wait times out to a default).

    The consumer *must* call ``request.response_event.set()`` after writing
    the user's choice into ``request.abort[0]``; the worker thread is
    blocking on ``request.response_event.wait()``.

    Parameters
    ----------
    request :
        The :class:`DialogRequest` object shared between the worker and the
        consumer.
    """

    request: Optional[DialogRequest] = None
