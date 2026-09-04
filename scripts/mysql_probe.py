#!/usr/bin/env python3
"""Bounded MySQL reachability probe that completes the protocol handshake (#790).

Why not a bare TCP connect
--------------------------
MySQL counts every connection from a host that opens TCP to 3306 and drops
**without completing the handshake** against ``max_connect_errors`` (default
100). Past the limit the server refuses every connection from that host with
error 1129 ("Host is blocked because of many connection errors") until an
admin runs ``FLUSH HOSTS``. A ``/dev/tcp`` or ``nc`` port probe is exactly
that pattern — and the server sees every VPN client as the VPN pool's NAT
address, so one watch loop blocks the DB for everyone on VPN (live incident
2026-09-04). Successful logins reset the counter; a *failed authentication*
is not counted at all. Only connect-and-drop is.

What this does instead
----------------------
Opens a real MySQL connection through the same pure-Python connector
``geecs_core.db.geecs_db`` uses, bounded by ``--timeout``, and closes it
cleanly (``COM_QUIT``). Credentials come from GEECS's ``Configurations.INI``
via ``geecs_core.db.geecs_db._find_credentials`` when that resolves; otherwise
a throwaway probe user is offered — the server's ``Access denied`` still
proves the port answered and costs the host nothing.

Exit codes (one status word on stdout, detail after it)::

    0  ok       reachable — ``login`` succeeded or ``auth-refused`` (1045-class)
    1  down     no MySQL answered within the timeout (refused / timed out / reset)
    3  blocked  reachable, but THIS host is blocked (1129) — needs FLUSH HOSTS
    4  no-connector  mysql-connector-python is not importable here
"""

from __future__ import annotations

import argparse

#: Offered when no GEECS credentials resolve. A refused login is a complete
#: handshake, so it is *not* counted toward the host block.
PROBE_USER = "geecs-lab-status-probe"

#: Client-side connector errors that mean "nothing answered": can't connect
#: (2003), unknown host (2005), lost connection (2013), timeout (2055 is not
#: used by the pure connector; timeouts surface as 2003 with a reason).
_DOWN_ERRNOS = frozenset({2003, 2005, 2013})

#: Server refused *this host*: the max_connect_errors block.
_BLOCKED_ERRNO = 1129


def _credentials(host: str, port: int) -> dict[str, object]:
    """Return connector kwargs: GEECS credentials when resolvable, else the probe user.

    Parameters
    ----------
    host : str
        Host to probe (always wins over the host named in Configurations.INI —
        the caller decides which box is under test).
    port : int
        TCP port to probe.

    Returns
    -------
    dict[str, object]
        Keyword arguments for ``mysql.connector.connect``.
    """
    creds: dict[str, object] = {"user": PROBE_USER, "password": ""}
    try:
        from geecs_core.db.geecs_db import _find_credentials

        found = dict(_find_credentials())
        creds = {
            "user": found["user"],
            "password": found["password"],
            "database": found["database"],
        }
    except Exception:  # noqa: BLE001 — no config here is fine; the probe user suffices
        pass
    creds["host"] = host
    creds["port"] = port
    return creds


def probe(host: str, port: int, timeout: float) -> tuple[int, str]:
    """Attempt one handshake-completing connection.

    Parameters
    ----------
    host : str
        Host to probe.
    port : int
        TCP port to probe.
    timeout : float
        Connect timeout in seconds (the connector's ``connection_timeout``).

    Returns
    -------
    tuple[int, str]
        ``(exit_code, status_line)`` per the module docstring's table.
    """
    try:
        import mysql.connector
        from mysql.connector import errors as mysql_errors
    except ImportError:
        return (
            4,
            "no-connector mysql-connector-python not importable in this interpreter",
        )

    kwargs = _credentials(host, port)
    who = "geecs credentials" if kwargs["user"] != PROBE_USER else "probe user"
    try:
        conn = mysql.connector.connect(
            **kwargs,
            use_pure=True,  # the same implementation geecs_core.db.geecs_db uses
            connection_timeout=timeout,
        )
    except mysql_errors.Error as exc:
        errno = getattr(exc, "errno", None)
        if errno == _BLOCKED_ERRNO:
            return 3, (
                f"blocked {host}:{port} answered but refuses this host (MySQL 1129): "
                "too many aborted connects from this address — a DB admin must run "
                "FLUSH HOSTS on the server; the VPN pool shares one counter"
            )
        if errno in _DOWN_ERRNOS or errno is None or errno >= 2000:
            return 1, f"down {host}:{port} — {type(exc).__name__} {errno}: {exc.msg}"
        # Any other server-side error (1045 access denied, 1044/1049 database
        # problems, ...) came from a MySQL server that completed the handshake.
        return 0, f"ok auth-refused (MySQL {errno}), {who} — a server answered"
    except OSError as exc:
        return 1, f"down {host}:{port} — {type(exc).__name__}: {exc}"
    try:
        conn.close()  # sends COM_QUIT: a clean close, not an aborted connect
    except Exception:  # noqa: BLE001 — the login already proved reachability
        pass
    return 0, f"ok login, {who}"


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (``None`` reads ``sys.argv``).

    Returns
    -------
    int
        Process exit status per the module docstring's table.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="seconds (default 2)"
    )
    args = parser.parse_args(argv)
    code, line = probe(args.host, args.port, args.timeout)
    print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
