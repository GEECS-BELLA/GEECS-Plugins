"""The worker-wide default :class:`~geecs_bluesky.session.GeecsSession`.

A neutral home, because there are now **two** consumers with different
lifetimes: the ScanRequest funnel (``plans/scan_request_plan.py``, slated
for retirement with GEECS-Plugins#807 phase C) and the stock-plan preamble
preprocessor (``preprocessors.py``, which outlives it).  Keeping the global
in the funnel would make the preprocessor import a private name from the
module that is about to be deleted.
"""

from __future__ import annotations

from typing import Any

#: The worker-wide default session (installed once at worker startup).
_worker_session: Any | None = None


def set_plan_session(session: Any | None) -> None:
    """Install the worker-wide default :class:`GeecsSession` for the plans.

    The queueserver registers plans by name with JSON args, so the session
    cannot travel in the call — a worker startup script constructs one
    headless session and installs it here; ``RE(geecs_scan_request_plan(
    request))``, and any stock plan carrying ``md={"geecs": ...}``, then
    need nothing else.  Pass ``None`` to clear (tests).

    Parameters
    ----------
    session :
        The session whose RunEngine will execute the plan.  The plan's
        connect messages run on that engine's loop, so running the plan on
        a *different* RunEngine than ``session.RE`` is unsupported.
    """
    global _worker_session
    _worker_session = session


def get_plan_session() -> Any | None:
    """The installed worker-wide session, or ``None`` if none was installed."""
    return _worker_session
