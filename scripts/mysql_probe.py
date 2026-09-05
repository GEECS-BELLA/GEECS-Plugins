#!/usr/bin/env python3
"""Bounded MySQL reachability probe that completes the protocol handshake (#790).

Never probe the DB port with a bare TCP connect: MySQL counts a connect that
drops before the handshake toward its host block (error 1129) — the full
story, incident and remedy are in ``docs/platform/fleet_map.md`` (the MySQL
admonition). This script is the one DB probe: a real, bounded login through
the same pure-Python connector ``geecs_core.db.geecs_db`` uses, closed
cleanly (``COM_QUIT``).

Target and credentials come from GEECS's ``Configurations.INI`` (via
``geecs_core.db.geecs_db._find_credentials``) when it resolves within
``--creds-timeout`` — the same server the real clients connect to.
Otherwise the ``--host`` / ``--port`` fallback (the caller's guess) is
probed with a throwaway probe user; the server's ``Access denied`` still
proves the port answered and costs the host nothing. The lookup is bounded
on its own because ``Configurations.INI`` lives on the data share: a
stalled SMB mount must not turn into a "DB down" verdict.

Exit codes (one status word on stdout, then ``host:port`` where known,
then detail)::

    0  ok           reachable — ``login`` succeeded or ``auth-refused`` (1045-class)
    1  down         no MySQL answered within the timeout (refused / timed out / reset)
    3  blocked      reachable, but THIS host is blocked (1129) — needs FLUSH HOSTS
    4  no-connector mysql-connector-python is not importable here
    5  unverified   a server answered but the *connector* could not complete the
                    handshake (e.g. an auth plugin it lacks) — from the server's
                    side that is an aborted connect; do not retry in a loop

Only an error the **server** sent (an ERR packet, which always carries a
``sqlstate``, or a server errno below 2000) proves a completed handshake;
connector-raised errors (errno −1, no sqlstate) never do, and the
connector's own 2xxx range means nothing answered.
"""

from __future__ import annotations

import argparse
import threading

#: Offered when no GEECS credentials resolve. A refused login is a complete
#: handshake, so it is *not* counted toward the host block.
PROBE_USER = "geecs-lab-status-probe"

#: Client-side connector errors that mean "nothing answered": can't connect
#: (2003), unknown host (2005), lost connection (2013), timeout (2055 is not
#: used by the pure connector; timeouts surface as 2003 with a reason).
_DOWN_ERRNOS = frozenset({2003, 2005, 2013})

#: The connector's own errno range (client errors). Server errnos sit below
#: it (1xxx) and, on MySQL 8, above it (3xxx) — those come in ERR packets.
_CLIENT_ERRNO_RANGE = range(2000, 3000)

#: Server refused *this host*: the max_connect_errors block.
_BLOCKED_ERRNO = 1129


def _lookup_credentials(timeout: float) -> dict[str, object] | None:
    """Run ``_find_credentials`` on a daemon thread, bounded by ``timeout``.

    Parameters
    ----------
    timeout : float
        Seconds to wait for the lookup (the INI lives on the data share).

    Returns
    -------
    dict[str, object] | None
        The resolved credentials, or ``None`` when the lookup failed or did
        not finish in time (the thread is abandoned; the process exits soon).
    """
    box: dict[str, object] = {}

    def run() -> None:
        try:
            from geecs_core.db.geecs_db import _find_credentials

            box["creds"] = dict(_find_credentials())
        except Exception:  # noqa: BLE001 — no config here is fine; the probe user suffices
            pass

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    found = box.get("creds")
    return found if isinstance(found, dict) else None


def _credentials(
    host: str, port: int, creds_timeout: float = 2.0
) -> tuple[dict[str, object], str]:
    """Return connector kwargs and a label for where the target/credentials came from.

    Parameters
    ----------
    host : str
        Fallback host — probed only when ``Configurations.INI`` does not
        resolve (its ``[Database] ipaddress`` wins: that is the server the
        real clients connect to).
    port : int
        Fallback TCP port, same rule.
    creds_timeout : float
        Seconds allowed for the credential lookup before falling back.

    Returns
    -------
    tuple[dict[str, object], str]
        ``(kwargs for mysql.connector.connect, who)`` — ``who`` is
        ``"geecs credentials"`` or ``"probe user"``.
    """
    found = _lookup_credentials(creds_timeout)
    if found:
        return (
            {
                "host": str(found.get("host") or host),
                "port": int(found.get("port") or port),
                "user": found["user"],
                "password": found["password"],
                "database": found["database"],
            },
            "geecs credentials",
        )
    return (
        {"host": host, "port": port, "user": PROBE_USER, "password": ""},
        "probe user",
    )


def classify_error(exc: Exception, target: str, who: str) -> tuple[int, str]:
    """Reduce a connector exception to the exit table's verdict.

    Parameters
    ----------
    exc : Exception
        The ``mysql.connector.Error`` raised by ``connect``.
    target : str
        ``host:port`` for the status line.
    who : str
        Credential label for the status line.

    Returns
    -------
    tuple[int, str]
        ``(exit_code, status_line)`` per the module docstring's table.
    """
    errno = getattr(exc, "errno", None)
    sqlstate = getattr(exc, "sqlstate", None)
    msg = getattr(exc, "msg", str(exc))
    name = type(exc).__name__
    if errno == _BLOCKED_ERRNO:
        return 3, (
            f"blocked {target} answered but refuses this host (MySQL 1129): "
            "too many aborted connects from this address — a DB admin must run "
            "FLUSH HOSTS on the server; the VPN pool shares one counter"
        )
    if errno in _DOWN_ERRNOS or errno in _CLIENT_ERRNO_RANGE:
        return 1, f"down {target} — {name} {errno}: {msg}"
    # Only the server proves a completed handshake: its ERR packets carry a
    # sqlstate (protocol 4.1+), and its errnos sit below 2000 (1045 access
    # denied, 1044/1049 database problems, ...) or, on MySQL 8, above 3000
    # (sqlstate present). Connector-raised errors (errno -1, no sqlstate —
    # an auth plugin it lacks, say) are a connect-and-drop from the server's
    # point of view.
    if sqlstate or (errno is not None and 0 < errno < 2000):
        return 0, f"ok {target} auth-refused (MySQL {errno}), {who} — a server answered"
    return 5, (
        f"unverified {target} answered but the connector could not complete the "
        f"handshake ({name} {errno}: {msg}) — the server counts that as an "
        "aborted connect; do not loop on this probe"
    )


def probe(
    host: str, port: int, timeout: float, creds_timeout: float = 2.0
) -> tuple[int, str]:
    """Attempt one handshake-completing connection.

    Parameters
    ----------
    host : str
        Fallback host (see ``_credentials``).
    port : int
        Fallback TCP port.
    timeout : float
        Connect timeout in seconds (the connector's ``connection_timeout``).
    creds_timeout : float
        Seconds allowed for the ``Configurations.INI`` lookup.

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

    kwargs, who = _credentials(host, port, creds_timeout)
    target = f"{kwargs['host']}:{kwargs['port']}"
    try:
        conn = mysql.connector.connect(
            **kwargs,
            use_pure=True,  # the same implementation geecs_core.db.geecs_db uses
            connection_timeout=timeout,
        )
    except mysql_errors.Error as exc:
        return classify_error(exc, target, who)
    except OSError as exc:
        return 1, f"down {target} — {type(exc).__name__}: {exc}"
    try:
        conn.close()  # sends COM_QUIT: a clean close, not an aborted connect
    except Exception:  # noqa: BLE001 — the login already proved reachability
        pass
    return 0, f"ok {target} login, {who}"


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
    parser.add_argument(
        "--host",
        required=True,
        help="fallback host when Configurations.INI does not resolve",
    )
    parser.add_argument("--port", type=int, default=3306, help="fallback port")
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="connect seconds (default 2)"
    )
    parser.add_argument(
        "--creds-timeout",
        type=float,
        default=2.0,
        help="seconds for the Configurations.INI lookup on the data share (default 2)",
    )
    args = parser.parse_args(argv)
    code, line = probe(args.host, args.port, args.timeout, args.creds_timeout)
    print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
