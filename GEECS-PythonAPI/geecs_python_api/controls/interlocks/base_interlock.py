"""Base class for interlock checks consumed by ``InterlockServer``.

An interlock is a single boolean condition that a downstream consumer (e.g.
Master Control) polls to decide whether some action — typically firing the
laser — is allowed.

Each subclass implements :meth:`BaseInterlock.check` and returns ``True`` when
the condition is **not** met (i.e. the consumer should hold off).  Freshness
tracking is provided here so subclasses do not have to reimplement
stale-data detection in every concrete interlock.

Concrete interlocks should keep ``__init__`` side-effect free so that they
can be constructed from YAML without touching the network or the GEECS
database.  Any device wiring (subscriptions, TCP connections) belongs in
:meth:`connect`, which the server calls once before the monitor loop starts.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseInterlock(ABC):
    """Abstract base class for an interlock condition.

    Parameters
    ----------
    name : str
        Identifier broadcast by the server; visible to clients.
    poll_interval : float, optional
        Seconds between successive ``check`` calls (default 0.5 s).
    stale_after : float, optional
        Maximum age in seconds of the most recent successful check before
        the server treats the interlock as unsafe.  Defaults to
        ``5 * poll_interval`` and can be tightened or loosened per
        subclass.

    Attributes
    ----------
    last_check_time : float or None
        Wall-clock time (``time.time()``) of the most recent successful
        :meth:`check` call.  Stamped by the server's monitor loop, not by
        subclasses.
    """

    def __init__(
        self,
        name: str,
        poll_interval: float = 0.5,
        stale_after: Optional[float] = None,
    ):
        self.name = name
        self.poll_interval = poll_interval
        self.stale_after = (
            stale_after if stale_after is not None else 10 * poll_interval
        )
        self.last_check_time: Optional[float] = None

    def connect(self) -> None:
        """Acquire any external resources (device subscriptions, sockets).

        Called once by the server before the monitor loop starts.  Default
        implementation is a no-op; subclasses override when they need to
        attach to GEECS devices or other live data sources.
        """

    @abstractmethod
    def check(self) -> bool:
        """Evaluate the interlock condition.

        Returns
        -------
        bool
            ``True`` when the condition is **not** met (consumer should
            hold off), ``False`` when conditions are satisfied.

        Notes
        -----
        Subclasses should raise on programmer errors (bad config, missing
        device) but return ``True`` for runtime states they consider
        unsafe (stale data, out-of-range value, etc.).  The server treats
        uncaught exceptions as unsafe.
        """

    def is_stale(self, now: Optional[float] = None) -> bool:
        """Return True when the last successful check is older than ``stale_after``.

        Parameters
        ----------
        now : float, optional
            Override for the current time; defaults to ``time.time()``.
            Useful for testing.
        """
        if self.last_check_time is None:
            return True
        current = now if now is not None else time.time()
        return (current - self.last_check_time) > self.stale_after
