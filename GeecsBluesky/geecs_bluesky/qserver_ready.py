"""``geecs-qserver-ensure-ready`` — a running ``geecs-qserver`` means *ready*.

bluesky-queueserver treats ``environment open`` as an operator gesture: the
manager starts knowing no plans, and only opening the worker environment
imports the startup profile and populates ``plans_allowed``.  Run as a
systemd service that is the wrong contract — a manager that restarted
unattended answers ``qserver status`` healthily and refuses every
submission with "Plan ... is not in the list of allowed plans" (#793, live
2026-09-04, after Phase 3 re-rendered the unit onto a fresh clone).

This entry point is the readiness assertion the ``geecs-qserver-ready``
oneshot unit runs after the manager (``qserver/deploy/``): wait for the
manager to answer, open the environment if it is closed, wait for the
worker environment to finish initializing, then **assert the manager lists
every plan this code implements** (:data:`~geecs_bluesky.plan_names.GEECS_PLAN_NAMES`).
The plan-list assertion is the point: an open that succeeded onto a partial
import, or a permissions file that excludes a plan, is still broken, and
only that check catches it.  Exit codes: 0 ready; 1 not ready (the message
says exactly what was found); 2 usage.

Talks to the manager over its 0MQ control socket through
``bluesky_queueserver.manager.comms.zmq_single_request`` — the ``qserver``
extra the worker host already installs; the ``qs-client`` extra is *not*
required.  The transport is injectable (``request=``) so the logic is
tested without a manager.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

from geecs_bluesky.plan_names import GEECS_PLAN_NAMES

logger = logging.getLogger(__name__)

#: Default control-socket address as seen from the worker host itself.
DEFAULT_CONTROL_ADDR = "tcp://localhost:60615"
#: The permissions group the plan list is asked for — the CLI's default;
#: ``user_group_permissions.yaml`` allows every ``geecs_*`` plan to it.
DEFAULT_USER_GROUP = "primary"
#: Overall budget: the environment open imports the startup profile (and
#: warms the optimize stack — torch/botorch), which takes minutes cold.
DEFAULT_TIMEOUT_S = 600.0
#: Poll interval while waiting on the manager.
POLL_S = 2.0

#: ``request(method, params) -> (msg, err_msg)``: the manager transport.
Request = Callable[[str, dict[str, Any] | None], tuple[dict[str, Any] | None, str]]


class NotReady(RuntimeError):
    """The manager is not (or could not be made) ready; message says why."""


def _zmq_request(control_addr: str) -> Request:
    """The real transport: one bounded request per call, socket per call."""
    from bluesky_queueserver.manager.comms import zmq_single_request

    def request(method: str, params: dict[str, Any] | None = None):
        return zmq_single_request(
            method,
            params or {},
            zmq_server_address=control_addr,
            timeout=int(POLL_S * 1000),
        )

    return request


def _call(request: Request, method: str, params: dict[str, Any] | None = None) -> dict:
    """One manager call; raises :class:`NotReady` on transport or ``success: False``."""
    msg, err = request(method, params)
    if msg is None:
        raise NotReady(f"manager did not answer {method!r}: {err or 'no reply'}")
    if not msg.get("success", True):
        raise NotReady(f"manager refused {method!r}: {msg.get('msg') or msg}")
    return msg


def _wait_for_manager(request: Request, deadline: float) -> dict:
    """Poll ``status`` until the manager answers or the deadline passes."""
    last = ""
    while True:
        msg, err = request("status", None)
        if msg is not None:
            return msg
        last = err or "no reply"
        if time.monotonic() >= deadline:
            raise NotReady(
                f"manager did not answer 'status' before the deadline: {last}"
            )
        time.sleep(POLL_S)


#: Worker-environment states in which the environment is still coming or going.
_TRANSIENT_ENV_STATES = ("initializing", "closing")


def _wait_for_environment(
    request: Request, deadline: float, *, closed_grace: int = 3
) -> dict:
    """Poll until the worker environment exists and has finished initializing.

    "Exists and not transient" — not "idle": a manager that is executing a
    plan when the readiness unit is (re)run by hand is ready too, and its
    plan list is served regardless.  A ``closed`` + ``idle`` snapshot
    *after* we asked for an open means the open failed and the manager
    settled back (startup profile import error) — but the first polls after
    the request may still read the pre-open state, so up to *closed_grace*
    consecutive such snapshots are tolerated before that is concluded.
    """
    closed_polls = 0
    while True:
        status = _wait_for_manager(request, deadline)
        exists = bool(status.get("worker_environment_exists"))
        manager_state = status.get("manager_state")
        env_state = status.get("worker_environment_state")
        if (
            exists
            and manager_state != "creating_environment"
            and env_state not in _TRANSIENT_ENV_STATES
        ):
            return status
        if env_state == "closed" and manager_state == "idle":
            closed_polls += 1
            if closed_polls > closed_grace:
                raise NotReady(
                    "the worker environment is closed after 'environment open' — "
                    "the startup profile failed to import (journalctl -u geecs-qserver)"
                )
        else:
            closed_polls = 0
        if time.monotonic() >= deadline:
            raise NotReady(
                "worker environment not up before the deadline "
                f"(worker_environment_exists={exists}, manager_state={manager_state!r}, "
                f"worker_environment_state={env_state!r})"
            )
        time.sleep(POLL_S)


def ensure_ready(
    request: Request,
    *,
    expected_plans: Sequence[str] = GEECS_PLAN_NAMES,
    user_group: str = DEFAULT_USER_GROUP,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    log: Callable[[str], None] = logger.info,
) -> list[str]:
    """Make the manager ready, or raise :class:`NotReady` saying why not.

    Parameters
    ----------
    request :
        The manager transport (``(method, params) -> (msg, err_msg)``).
    expected_plans :
        Plan names that must all appear in ``plans_allowed``.
    user_group :
        The permissions group the plan list is asked for.
    timeout_s :
        Overall wall-clock budget for answer + open + idle.
    log :
        Progress sink (one line per step).

    Returns
    -------
    list of str
        The allowed plan names the manager reported (sorted).

    Raises
    ------
    NotReady
        Manager unreachable, the open refused or collapsed, idle not
        reached in time, or the plan list missing an expected plan.
    """
    deadline = time.monotonic() + timeout_s
    status = _wait_for_manager(request, deadline)
    log(
        "manager answers: manager_state=%r worker_environment_exists=%s re_state=%r"
        % (
            status.get("manager_state"),
            status.get("worker_environment_exists"),
            status.get("re_state"),
        )
    )
    if not status.get("worker_environment_exists"):
        if status.get("manager_state") not in ("idle", "creating_environment"):
            raise NotReady(
                f"manager is {status.get('manager_state')!r} with no worker environment — "
                "not a state 'environment open' can fix"
            )
        if status.get("manager_state") == "idle":
            log("worker environment closed — opening it")
            _call(request, "environment_open", {})
    status = _wait_for_environment(request, deadline)
    log(
        "worker environment up (worker_environment_state=%r, re_state=%r)"
        % (status.get("worker_environment_state"), status.get("re_state"))
    )

    reply = _call(request, "plans_allowed", {"user_group": user_group})
    allowed = sorted((reply.get("plans_allowed") or {}).keys())
    missing = [name for name in expected_plans if name not in allowed]
    if missing:
        raise NotReady(
            f"plans_allowed lacks {', '.join(missing)} (listed: "
            f"{', '.join(allowed) or 'none'}; user_group={user_group!r}) — the "
            "startup profile did not register them or user_group_permissions.yaml "
            "excludes them; see qserver/README.md Troubleshooting"
        )
    log(
        "ready: %d allowed plans, all %d expected present"
        % (len(allowed), len(expected_plans))
    )
    return allowed


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: exit 0 when ready, 1 when not, 2 on usage errors."""
    parser = argparse.ArgumentParser(
        prog="geecs-qserver-ensure-ready",
        description=(
            "Wait for the GEECS RE Manager, open its worker environment if closed, "
            "wait for idle, and assert every GEECS plan is allowed."
        ),
    )
    parser.add_argument(
        "--control-addr",
        default=os.environ.get("QS_CONTROL_ADDR", DEFAULT_CONTROL_ADDR),
        help="manager 0MQ control socket (env QS_CONTROL_ADDR; default %(default)s)",
    )
    parser.add_argument(
        "--user-group",
        default=DEFAULT_USER_GROUP,
        help="permissions group to ask plans_allowed for (default %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="overall budget in seconds (default %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        ensure_ready(
            _zmq_request(args.control_addr),
            user_group=args.user_group,
            timeout_s=args.timeout,
        )
    except NotReady as exc:
        logger.error("NOT READY: %s", exc)
        return 1
    except ImportError as exc:
        logger.error(
            "bluesky-queueserver is not installed in this environment: %s", exc
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
