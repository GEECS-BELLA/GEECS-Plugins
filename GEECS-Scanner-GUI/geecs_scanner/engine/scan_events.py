"""Typed scan event hierarchy and ScanState enum.

Every state transition and device command outcome in the scan engine emits one
of these events via the ``on_event`` callback injected into :class:`ScanManager`.

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
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from geecs_scanner.engine.dialog_request import DialogRequest


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
        - ``"rejected"`` — :class:`GeecsDeviceCommandRejected` raised.
        - ``"failed"``   — :class:`GeecsDeviceCommandFailed` raised.
        - ``"timeout"``  — :class:`GeecsDeviceExeTimeout` raised.
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
# Dialog requests (replaces dialog_queue in Block 7)
# ---------------------------------------------------------------------------


@dataclass
class ScanDialogEvent(ScanEvent):
    """Carries a device-error dialog request across the worker→consumer boundary.

    In Block 6 this event is emitted but the GUI still drains ``dialog_queue``
    as before.  In Block 7 the GUI callback receives this event and posts it
    to the Qt main thread via ``QMetaObject.invokeMethod``; ``dialog_queue``
    is removed at that point.

    The consumer *must* call ``request.response_event.set()`` after writing
    the user's choice into ``request.abort[0]``; the worker thread is
    blocking on ``request.response_event.wait()``.

    Parameters
    ----------
    request :
        The :class:`~geecs_scanner.engine.dialog_request.DialogRequest`
        object shared between the worker and the consumer.
    """

    request: Optional[DialogRequest] = None
