"""Base class for interlock checks consumed by ``InterlockServer``.

An interlock is a single boolean condition that a downstream consumer
(e.g. Master Control) polls to decide whether some action — typically
firing the laser — is allowed.

Each subclass implements :meth:`BaseInterlock.check` and returns
``True`` when the condition is **not** met (i.e. the consumer should
hold off).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseInterlock(ABC):
    """Abstract base class for an interlock condition.

    Parameters
    ----------
    name : str
        Identifier broadcast by the server; visible to clients.
    poll_interval : float, optional
        Seconds between successive ``check`` calls (default 0.5 s).
    """

    def __init__(self, name: str, poll_interval: float = 0.5):
        self.name = name
        self.poll_interval = poll_interval

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
        Subclasses should raise on programmer errors (bad config,
        missing device) but return ``True`` for runtime states they
        consider unsafe (stale data, out-of-range value, etc.).  The
        server treats uncaught exceptions as unsafe.
        """
