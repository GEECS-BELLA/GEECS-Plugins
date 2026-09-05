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
only that check catches it.  The plan list is read through the same
``qs_client.readiness_from_reads`` assembly the pre-submit ``worker_ready``
check runs (one definition of ready); after an open this run requested it
is re-read for a short settle window, because the manager reports the
environment up before its own plan-list download has landed.  Exit codes:
0 ready; 1 not ready (the message says exactly what was found); 2 usage.

The address asserted is the manager on **this** host (loopback), or
``QS_CONTROL_ADDR`` when set — never the client-side ``[qserver]`` config.

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
from geecs_bluesky.qs_client.client import (
    _CONTROL_PORT,
    queue_status_from_manager,
    readiness_from_reads,
)

logger = logging.getLogger(__name__)

#: The control socket of the manager on THIS host — the one the unit
#: asserts.  The port is the client module's (one definition).
DEFAULT_CONTROL_ADDR = f"tcp://localhost:{_CONTROL_PORT}"


def default_control_addr() -> str:
    """The manager address to assert against: ``QS_CONTROL_ADDR``, else loopback.

    Deliberately NOT the client-side ``[qserver]`` section of the service
    account's ``config.ini``: that names the worker *clients* talk to, and
    a carried-over config naming another box would make this unit report
    ready against the wrong manager.  The unit asserts the local manager;
    the only override is the environment variable, settable from
    ``site.env`` through the unit's ``EnvironmentFile=``.
    """
    env = os.environ.get("QS_CONTROL_ADDR", "").strip()
    return env or DEFAULT_CONTROL_ADDR


#: The permissions group the plan list is asked for — the CLI's default;
#: ``user_group_permissions.yaml`` allows every ``geecs_*`` plan to it.
DEFAULT_USER_GROUP = "primary"
#: Overall budget: the environment open imports the startup profile (and
#: warms the optimize stack — torch/botorch), which takes minutes cold.
DEFAULT_TIMEOUT_S = 600.0
#: Poll interval while waiting on the manager.
POLL_S = 2.0
#: After the environment came up on *our* open, the manager reports
#: ``worker_environment_exists`` before its own plan-list download task
#: has landed, so the first ``plans_allowed`` reads can still return the
#: pre-open list (empty on a fresh manager, stale after a deploy that adds
#: a plan).  Re-read up to this many times at ``POLL_S`` (~10 s) before
#: concluding ``plans_empty`` / ``plan_missing``.
PLAN_LIST_SETTLE_POLLS = 5
#: The verdict states the post-open settle re-reads (never ``plans_unknown``
#: — an unanswered list is not ready, full stop).
_SETTLING_STATES = ("plans_empty", "plan_missing")

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
    request: Request,
    deadline: float,
    *,
    log: Callable[[str], None],
    initial: dict | None = None,
    closed_grace: int = 3,
) -> tuple[dict, bool]:
    """Poll until the worker environment exists and has finished initializing.

    *initial* is a status snapshot already read (the manager's first
    answer), judged before any further poll.

    Opens the environment when (and only when) a poll reads it settled
    closed (``closed`` + ``idle``) before this run has asked for an open —
    the fresh-restart state, but also the state after someone ran
    ``qserver environment close``.  A manager still ``closing_environment``
    (or otherwise busy without an environment) is waited out, not
    refused: it settles to closed + idle, and then we open.

    "Exists and not transient" — not "idle": a manager that is executing a
    plan when the readiness unit is (re)run by hand is ready too, and its
    plan list is served regardless.  A ``closed`` + ``idle`` snapshot
    *after* our open means the open failed and the manager settled back
    (startup profile import error) — but the first polls after the
    request may still read the pre-open state, so up to *closed_grace*
    consecutive such snapshots are tolerated before that is concluded.

    Returns
    -------
    tuple of (dict, bool)
        The status snapshot that read up, and whether this run opened the
        environment (the post-open plan-list settle applies then).
    """
    opened = False
    closed_polls = 0
    status = initial
    while True:
        if status is None:
            status = _wait_for_manager(request, deadline)
        exists = bool(status.get("worker_environment_exists"))
        manager_state = status.get("manager_state")
        env_state = status.get("worker_environment_state")
        if (
            exists
            and manager_state != "creating_environment"
            and env_state not in _TRANSIENT_ENV_STATES
        ):
            return status, opened
        if not exists and env_state == "closed" and manager_state == "idle":
            if not opened:
                log("worker environment closed — opening it")
                _call(request, "environment_open", {})
                opened = True
            else:
                closed_polls += 1
                if closed_polls > closed_grace:
                    raise NotReady(
                        "the worker environment is closed after 'environment open' — "
                        "the startup profile failed to import "
                        "(journalctl -u geecs-qserver)"
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
        status = None


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
    status, opened = _wait_for_environment(request, deadline, log=log, initial=status)
    log(
        "worker environment up (worker_environment_state=%r, re_state=%r)"
        % (status.get("worker_environment_state"), status.get("re_state"))
    )

    def read_plans() -> dict | None:
        reply, err = request("plans_allowed", {"user_group": user_group})
        if reply is None or not reply.get("success", True):
            log("plans_allowed unanswered: %s" % (err or (reply or {}).get("msg")))
            return None
        return reply.get("plans_allowed") or {}

    # The ONE assembly of ready (qs_client.readiness_from_reads), shared
    # with the pre-submit worker_ready check: an unanswered plan list is
    # not ready.  After OUR open the manager's plan-list download may still
    # be in flight when the environment first reads up, so an empty or
    # incomplete list is re-read for a short settle window (F1, #795).
    settle_polls = PLAN_LIST_SETTLE_POLLS if opened else 0
    while True:
        verdict = readiness_from_reads(
            queue_status_from_manager(status), read_plans, list(expected_plans)
        )
        if verdict.ready:
            break
        if (
            verdict.state in _SETTLING_STATES
            and settle_polls > 0
            and time.monotonic() < deadline
        ):
            settle_polls -= 1
            log(
                "plan list not settled after the open (%s) — re-reading" % verdict.state
            )
            time.sleep(POLL_S)
            status = _wait_for_manager(request, deadline)
            continue
        raise NotReady(f"{verdict.detail} (user_group={user_group!r})")
    log(
        "ready: %d allowed plans, all %d expected present"
        % (len(verdict.allowed_plans), len(expected_plans))
    )
    return list(verdict.allowed_plans)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: exit 0 when ready, 1 when not, 2 on usage errors."""
    parser = argparse.ArgumentParser(
        prog="geecs-qserver-ensure-ready",
        description=(
            "Wait for the GEECS RE Manager, open its worker environment if closed, "
            "wait for it to come up, and assert every GEECS plan is allowed."
        ),
    )
    parser.add_argument(
        "--control-addr",
        default=None,
        help=(
            "manager 0MQ control socket; default: QS_CONTROL_ADDR, else "
            f"{DEFAULT_CONTROL_ADDR} (the manager on this host)"
        ),
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
    control_addr = args.control_addr or default_control_addr()
    logger.info("asserting readiness of the manager at %s", control_addr)
    try:
        ensure_ready(
            _zmq_request(control_addr),
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
