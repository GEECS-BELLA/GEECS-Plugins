"""Tiled integration for the RunEngine: the engine-side spool and the shared checks.

The engine never talks to Tiled.  :func:`subscribe_tiled_spool` subscribes
the per-run document spool (:mod:`geecs_bluesky.tiled_spool`) — the
catalog location from the standard ``~/.config/geecs_python_api/config.ini``
decides only whether spooling is on at all — and the separate
``geecs-tiled-writer`` service (:mod:`geecs_bluesky.tiled_writer`)
registers each run from its spool file.  Failures degrade to a warning:
scans run fine without Tiled.

Shared here: :func:`tiled_server_reachable` (the bounded TCP pre-check
the writer runs every sweep) and :class:`SafeDocumentCallback` (the
run-scoped failure guard any RE callback here wears).
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

# The one config reader (issue #527): re-exported here so existing callers
# (and test monkeypatches of this module's attribute) keep working.
from geecs_data_utils.tiled_catalog import read_tiled_config

logger = logging.getLogger(__name__)

#: TCP connect budget for the pre-check in :func:`tiled_server_reachable`.
#: ``tiled.client.from_uri`` issues synchronous HTTP requests whose connect
#: timeout is far longer — off the lab network that stalls session/scanner
#: construction for the full HTTP timeout just to discover the catalog is
#: unreachable.  2 s comfortably covers a live LAN/VPN handshake.
TILED_REACHABILITY_TIMEOUT_S = 2.0


def tiled_server_reachable(
    tiled_uri: str, timeout: float = TILED_REACHABILITY_TIMEOUT_S
) -> bool:
    """Cheap TCP reachability pre-check for the Tiled server at *tiled_uri*.

    Attempts one ``socket.create_connection`` to the URI's host/port (default
    port from the scheme: 443 for https, else 80).  Returns ``True`` on
    connect, ``False`` on any socket error — including a *timeout*, which is
    bounded at *timeout* seconds instead of the Tiled client's HTTP connect
    timeout.  An unparseable URI returns ``True`` so ``from_uri`` reports the
    real error.
    """
    parsed = urlparse(tiled_uri)
    host = parsed.hostname
    if host is None:
        return True  # let from_uri produce the actionable parse error
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class SafeDocumentCallback:
    """Document callback wrapper that logs and disables itself on failure.

    A failure only disables forwarding for the remainder of the *current*
    run: the next ``start`` document re-enables the callback (and is itself
    forwarded), so one transient storage error cannot silently disable
    persistence for every subsequent scan on a long-lived RunEngine.
    """

    def __init__(self, callback: Callable[[str, dict], None], label: str) -> None:
        self._callback = callback
        self._label = label
        self._enabled = True
        self._run_uid: str | None = None

    def __call__(self, name: str, doc: dict) -> None:
        """Forward one document unless the wrapped callback failed this run."""
        if name == "start":
            if not self._enabled:
                logger.error(
                    "%s re-enabled at start of run %s — it was disabled by a "
                    "failure during run %s, whose remaining documents were "
                    "NOT persisted",
                    self._label,
                    doc.get("uid"),
                    self._run_uid,
                )
                self._enabled = True
            self._run_uid = doc.get("uid")
        if not self._enabled:
            return
        try:
            self._callback(name, doc)
        except Exception:
            self._enabled = False
            logger.error(
                "%s failed while handling %s document during run %s; "
                "disabling callback for the remainder of this run",
                self._label,
                name,
                self._run_uid,
                exc_info=True,
            )


def subscribe_tiled_spool(run_engine, state_dir: Path | None = None) -> int | None:
    """Subscribe the Tiled document spool to *run_engine*; return the token or ``None``.

    Spooling is on when ``config.ini`` names a catalog (``[tiled] uri``,
    :func:`read_tiled_config`) — the same switch that turned the in-process
    writer on before — and off, with a warning, otherwise: a box with no
    Tiled has no writer to drain the spool, and the files would only pile
    up.  The state directory is *state_dir*, else
    :func:`~geecs_bluesky.tiled_spool.default_state_dir` (the
    ``GEECS_TILED_WRITER_STATE`` variable the units set).

    Nothing here reaches the network: whether the catalog is *reachable*
    is the writer's concern, sweep by sweep.  A missing or stale writer
    heartbeat under the state directory is a WARNING here, not a refusal:
    the runs still spool, and reach Tiled once a writer runs.
    """
    from geecs_bluesky.tiled_spool import (
        SpoolCallback,
        SpoolLayout,
        default_state_dir,
        read_heartbeat,
    )

    tiled_uri, _api_key = read_tiled_config()
    if not tiled_uri:
        logger.warning("No Tiled URI configured — Tiled storage disabled")
        return None
    layout = SpoolLayout(state_dir if state_dir is not None else default_state_dir())
    try:
        layout.ensure()
    except OSError:
        logger.warning(
            "Tiled spool directory %s cannot be created — Tiled storage disabled",
            layout.spool_dir,
            exc_info=True,
        )
        return None
    heartbeat = read_heartbeat(layout.heartbeat_path)
    if heartbeat is None or heartbeat.is_stale():
        logger.warning(
            "no fresh geecs-tiled-writer heartbeat under %s — runs will spool "
            "there and reach Tiled only once the writer runs",
            layout.state_dir,
        )
    token = run_engine.subscribe(
        SafeDocumentCallback(SpoolCallback(layout), label="TiledSpool")
    )
    logger.info(
        "Tiled spool subscribed — documents to %s for geecs-tiled-writer (catalog %s)",
        layout.spool_dir,
        tiled_uri,
    )
    return token


__all__ = [
    "TILED_REACHABILITY_TIMEOUT_S",
    "SafeDocumentCallback",
    "read_tiled_config",
    "subscribe_tiled_spool",
    "tiled_server_reachable",
]
