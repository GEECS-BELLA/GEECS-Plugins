"""Re-export shim: the scan event vocabulary lives in ``geecs_bluesky.events``.

The typed ScanEvent hierarchy and ScanState moved down to the engine package
(target-architecture vision §2 — "the event vocabulary moves down"): the
engine owns its vocabulary; front-ends import it from below.  This module
stays so every existing import path keeps working verbatim::

    from geecs_scanner.engine.scan_events import ScanLifecycleEvent, ScanState

These are the *same class objects* as ``geecs_bluesky.events`` — isinstance
checks agree across both import paths.  New code should import from
``geecs_bluesky.events`` directly.
"""

from geecs_bluesky.events import (
    DeviceCommandEvent,
    ScanDialogEvent,
    ScanErrorEvent,
    ScanEvent,
    ScanLifecycleEvent,
    ScanRestoreFailedEvent,
    ScanState,
    ScanStepEvent,
)

__all__ = [
    "ScanState",
    "ScanEvent",
    "ScanLifecycleEvent",
    "ScanStepEvent",
    "DeviceCommandEvent",
    "ScanErrorEvent",
    "ScanRestoreFailedEvent",
    "ScanDialogEvent",
]
