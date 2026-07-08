"""Re-export shim for DialogRequest + legacy device-error escalation helpers.

:class:`DialogRequest` (the thread-safe worker→GUI operator question) moved
down to ``geecs_bluesky.events`` with the rest of the event vocabulary
(target-architecture vision §2); it is re-exported here so every existing
import path keeps working verbatim — and it is the *same class object*, so
isinstance checks agree across both import paths.  New code should import it
from ``geecs_bluesky.events``.

What genuinely stays here is the **legacy-engine** residue that cannot move
down because it depends on ``geecs_python_api`` (which geecs_bluesky must not
import): the :data:`DEVICE_COMMAND_ERRORS` tuple and the
:func:`escalate_device_error` helper used by the legacy engine's
device-command escalation path.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from geecs_bluesky.events import DialogRequest
from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceCommandFailed,
    GeecsDeviceCommandRejected,
    GeecsDeviceExeTimeout,
)

__all__ = [
    "DialogRequest",
    "DEVICE_COMMAND_ERRORS",
    "escalate_device_error",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Convenience tuple — use in except clauses to catch any device command error
# ---------------------------------------------------------------------------

DEVICE_COMMAND_ERRORS = (
    GeecsDeviceExeTimeout,
    GeecsDeviceCommandRejected,
    GeecsDeviceCommandFailed,
)


# ---------------------------------------------------------------------------
# Escalation helper
# ---------------------------------------------------------------------------


def escalate_device_error(
    exc: Exception,
    on_escalate: Optional[Callable[..., bool]],
    context: Optional[str] = None,
) -> bool:
    """Call *on_escalate* with *exc* and return its result.

    If no callback is wired (headless / test context), the error is logged
    and ``True`` (abort) is returned so the caller can stop the scan safely.

    Parameters
    ----------
    exc :
        The device exception to escalate.
    on_escalate :
        Callable that submits *exc* to the GUI dialog queue and blocks until
        the user responds.  Returns ``True`` → Abort, ``False`` → Continue.
    context :
        Optional extra information shown in the dialog body — e.g. the full
        list of variables that were being set for a device when the error
        occurred.

    Returns
    -------
    bool
        ``True`` if the scan should be aborted, ``False`` to continue.
    """
    if on_escalate is not None:
        return on_escalate(exc, context=context)
    logger.error("Device error with no escalation callback — auto-aborting: %s", exc)
    return True
