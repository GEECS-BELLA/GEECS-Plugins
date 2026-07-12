"""Fake ScanEvents: same class names/attributes as geecs_bluesky.events.

The adapter dispatches on class *name* and duck-typed attributes (see
events_adapter's module docstring), so these hermetic stand-ins exercise the
exact production dispatch without importing the engine.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScanLifecycleEvent:
    state: str = "idle"
    total_shots: int = 0
    scan_number: Optional[int] = None


@dataclass
class ScanStepEvent:
    step_index: int = 0
    total_steps: int = 0
    shots_completed: int = 0
    phase: str = "started"


@dataclass
class ScanErrorEvent:
    message: str = ""
    recoverable: bool = True


@dataclass
class ScanRestoreFailedEvent:
    device: str = ""
    message: str = ""


@dataclass
class _Request:
    """Duck-typed stand-in for geecs_bluesky.events.DialogRequest."""

    exc: Optional[Exception] = None
    title: Optional[str] = None
    continue_label: Optional[str] = None
    abort_label: Optional[str] = None
    response_event: threading.Event = field(default_factory=threading.Event)
    abort: list = field(default_factory=lambda: [False])


@dataclass
class ScanDialogEvent:
    request: Optional[_Request] = None
