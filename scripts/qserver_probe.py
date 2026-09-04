#!/usr/bin/env python3
"""Is the queueserver ready, not merely listening (#793 part 3).

A running ``geecs-qserver`` unit answers on its control port while its RE
worker environment is closed — nothing opens the environment at service
start, and a closed environment means the manager knows **zero plans**, so
every submission fails with "Plan '…' is not in the list of allowed plans"
(live incident 2026-09-04). Port liveness therefore says nothing about
readiness. This probe asks the manager over 0MQ for its ``status`` and the
``plans_allowed`` of one user group and reduces both to a verdict:

    exit 0  ready     environment exists and at least one plan is allowed
    exit 5  notready  the manager answered but cannot accept a scan (reason follows)
    exit 1  down      no answer within the timeout

One status line on stdout: the verdict word, then a compact summary the
fleet table can carry as-is. Read-only — two bounded requests, no writes.
Run from an environment that has ``bluesky-queueserver`` installed (the
GeecsBluesky env; ``scripts/lib/net_probes.sh`` finds one).
"""

from __future__ import annotations

import argparse
from typing import Any


def summarize(
    status: dict[str, Any] | None, plans_allowed: dict[str, Any] | None, user_group: str
) -> tuple[int, str]:
    """Reduce the manager's ``status`` + ``plans_allowed`` replies to a verdict.

    Parameters
    ----------
    status : dict[str, Any] | None
        The ``status`` reply (``None`` when the request got no answer).
    plans_allowed : dict[str, Any] | None
        The ``plans_allowed`` reply for ``user_group`` (``None`` when it got no
        answer or was not attempted).
    user_group : str
        The permission group the plan list was fetched for.

    Returns
    -------
    tuple[int, str]
        ``(exit_code, line)`` — ``0 ready …``, ``5 notready …`` or ``1 down …``.
    """
    if not status:
        return 1, "down RE Manager did not answer"
    env_exists = bool(status.get("worker_environment_exists"))
    env_state = status.get("worker_environment_state") or (
        "closed" if not env_exists else "?"
    )
    manager = status.get("manager_state") or "?"
    re_state = status.get("re_state")
    queue = status.get("items_in_queue")
    running = "running" if status.get("running_item_uid") else None
    n_plans = len((plans_allowed or {}).get("plans_allowed") or {})
    plans_ok = bool(plans_allowed and plans_allowed.get("success", True))

    summary = f"manager {manager} · environment {env_state}"
    if re_state:
        summary += f" · RE {re_state}"
    if plans_ok:
        summary += f" · {n_plans} plans allowed ({user_group})"
    else:
        summary += " · plans_allowed unanswered"
    if queue is not None:
        summary += f" · queue {queue}"
    if running:
        summary += " · item running"

    if not env_exists:
        return 5, (
            f"notready worker environment CLOSED — 0 plans, every submission is refused "
            f"as 'not in the list of allowed plans'; nothing opens it at service start "
            f"(`qserver environment open`; #793). {summary}"
        )
    if plans_ok and n_plans == 0:
        return 5, (
            f"notready environment {env_state} but the manager allows 0 plans for "
            f"'{user_group}' — profile import or permissions problem. {summary}"
        )
    return 0, f"ready {summary}"


def main(argv: list[str] | None = None) -> int:
    """Entry point: two 0MQ requests, one verdict line, exit code per the docstring.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (``None`` reads ``sys.argv``).

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--addr", required=True, help="tcp://host:60615 (the RE Manager control socket)"
    )
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="seconds per request (default 2)"
    )
    parser.add_argument(
        "--user-group",
        default="primary",
        help="permission group for plans_allowed (default: primary)",
    )
    args = parser.parse_args(argv)
    try:
        from bluesky_queueserver.manager.comms import zmq_single_request
    except ImportError:
        print(
            "down bluesky-queueserver is not importable in this interpreter (see /env-doctor)"
        )
        return 1
    timeout_ms = int(args.timeout * 1000)
    status, _err = zmq_single_request(
        "status", timeout=timeout_ms, zmq_server_address=args.addr
    )
    plans = None
    if status:
        plans, _err = zmq_single_request(
            "plans_allowed",
            {"user_group": args.user_group},
            timeout=timeout_ms,
            zmq_server_address=args.addr,
        )
    code, line = summarize(status, plans, args.user_group)
    print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
