"""Scan lifecycle state machine — owns scan state and emits ScanLifecycleEvent."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from geecs_scanner.engine.scan_events import ScanEvent, ScanLifecycleEvent, ScanState

logger = logging.getLogger(__name__)


class ScanLifecycleStateMachine:
    """Track scan lifecycle state and emit ScanLifecycleEvent on every transition.

    Parameters
    ----------
    on_event : callable, optional
        Receives each ScanEvent; exceptions are caught and logged at DEBUG.
    """

    def __init__(self, on_event: Optional[Callable[[ScanEvent], None]] = None) -> None:
        self._state: ScanState = ScanState.IDLE
        self._state_lock: threading.Lock = threading.Lock()
        self._on_event = on_event

    def set_state(self, new_state: ScanState, total_shots: int = 0) -> None:
        """Transition to *new_state* and emit a ScanLifecycleEvent."""
        with self._state_lock:
            self._state = new_state
        self._emit(ScanLifecycleEvent(state=new_state, total_shots=total_shots))

    @property
    def current_state(self) -> ScanState:
        """The current lifecycle state."""
        with self._state_lock:
            return self._state

    def _emit(self, event: ScanEvent) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                logger.debug("on_event callback raised; ignoring", exc_info=True)
