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
    exit 6  unknown   ``status`` answered but ``plans_allowed`` did not (timed out
                      or ``success: false``) — readiness cannot be asserted
    exit 1  down      no answer within the timeout
    exit 2  no address to ask (no ``--addr`` and no ``[qserver]`` config)

One status line on stdout: the verdict word, then a compact summary the
fleet table can carry as-is. Read-only — two bounded requests, no writes.
Run from an environment that has ``bluesky-queueserver`` installed (the
GeecsBluesky env; ``scripts/lib/net_probes.sh`` finds one).

The manager address is ``--addr`` when given; otherwise the ``[qserver]``
section of ``~/.config/geecs_python_api/config.ini`` — its ``control_addr``
override verbatim, else ``tcp://<host>:60615`` — read through
``geecs_bluesky.qs_client.read_qserver_config`` when that imports here and
parsed directly otherwise (the same keys, the same precedence).
"""

from __future__ import annotations

import argparse
import configparser
import logging
from pathlib import Path
from typing import Any

#: The shared GEECS user config (the permanent fleet contract path).
USER_CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini")

#: The RE Manager's default 0MQ control port (GeecsBluesky/qserver/deploy/DEPLOYMENT.md).
CONTROL_PORT = 60615


def control_addr_from_ini(path: Path) -> str | None:
    """Return the manager control address named by a GEECS ``config.ini``.

    A direct parse of the ``[qserver]`` section with the precedence
    ``geecs_bluesky.qs_client.read_qserver_config`` applies: ``control_addr``
    verbatim when set, else ``tcp://<host>:60615`` from ``host``.

    Parameters
    ----------
    path : Path
        The config file (missing or unreadable ⇒ ``None``).

    Returns
    -------
    str | None
        ``tcp://host:port``, or ``None`` when the file has no usable
        ``[qserver]`` section.
    """
    parser = configparser.ConfigParser()
    try:
        if not parser.read(path):
            return None
    except configparser.Error:
        return None
    if not parser.has_section("qserver"):
        return None
    control = parser.get("qserver", "control_addr", fallback="").strip()
    host = parser.get("qserver", "host", fallback="").strip()
    if control:
        return control
    if host:
        return f"tcp://{host}:{CONTROL_PORT}"
    return None


def resolve_control_addr(addr: str | None) -> str | None:
    """Return the address to ask: ``addr`` if given, else the config's.

    Parameters
    ----------
    addr : str | None
        An explicit ``--addr`` (wins as-is).

    Returns
    -------
    str | None
        The control address, or ``None`` when nothing names one.
    """
    if addr:
        return addr
    try:
        from geecs_bluesky.qs_client import read_qserver_config

        config = read_qserver_config()
        if config is not None:
            return config.control_addr
    except Exception:  # noqa: BLE001 — not every probe python has geecs_bluesky; parse the INI ourselves
        pass
    return control_addr_from_ini(USER_CONFIG_PATH.expanduser())


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
        ``(exit_code, line)`` — ``0 ready …``, ``5 notready …``,
        ``6 unknown …`` or ``1 down …``.
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
    if not plans_ok:
        # status answered but the plan list did not (a second-request timeout,
        # or the manager refused it): the environment may be fine, but "ready"
        # is an assertion about plans, and there is none to make.
        reason = (plans_allowed or {}).get("msg") or "no answer within the timeout"
        return 6, (
            f"unknown plans_allowed unanswered for '{user_group}' ({reason}) — "
            f"readiness cannot be asserted; re-run, then check the manager log. "
            f"{summary}"
        )
    if n_plans == 0:
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
        "--addr",
        help="tcp://host:60615 (the RE Manager control socket); default: the "
        "[qserver] section of ~/.config/geecs_python_api/config.ini",
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
    addr = resolve_control_addr(args.addr)
    if not addr:
        print(
            "unknown no RE Manager address — pass --addr or set [qserver] host / "
            f"control_addr in {USER_CONFIG_PATH}"
        )
        return 2
    try:
        from bluesky_queueserver.manager.comms import zmq_single_request
    except ImportError:
        print(
            "down bluesky-queueserver is not importable in this interpreter (see /env-doctor)"
        )
        return 1
    # A timed-out request is this probe's "down" verdict, not an event: keep
    # the library's warning + traceback off stderr (it otherwise lands above
    # fleet_status's table).
    logging.getLogger("bluesky_queueserver").setLevel(logging.CRITICAL)
    timeout_ms = int(args.timeout * 1000)
    status, _err = zmq_single_request(
        "status", timeout=timeout_ms, zmq_server_address=addr
    )
    plans = None
    if status:
        plans, _err = zmq_single_request(
            "plans_allowed",
            {"user_group": args.user_group},
            timeout=timeout_ms,
            zmq_server_address=addr,
        )
    code, line = summarize(status, plans, args.user_group)
    print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
