"""Tiled catalog integration shared by BlueskyScanner and GeecsSession.

One call — :func:`subscribe_tiled` — reads the catalog location from the
standard ``~/.config/geecs_python_api/config.ini`` (unless given explicitly),
connects a ``TiledWriter``, and subscribes it to a RunEngine.  Failures degrade
to a warning: scans run fine without Tiled.
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def read_tiled_config() -> tuple[str | None, str | None]:
    """Read Tiled URI and API key from ``~/.config/geecs_python_api/config.ini``.

    Returns ``(uri, api_key)``, either of which may be ``None`` if absent.
    """
    config_path = Path.home() / ".config" / "geecs_python_api" / "config.ini"
    if not config_path.exists():
        return None, None

    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    if "tiled" not in cfg:
        return None, None

    uri = cfg["tiled"].get("uri") or None
    api_key = cfg["tiled"].get("api_key") or None
    logger.debug("Tiled config loaded from %s — uri=%s", config_path, uri)
    return uri, api_key


def prepare_descriptor_for_tiled(doc: dict) -> dict:
    """Store GEECS external asset datum IDs as internal Tiled metadata.

    The RunEngine document stream still emits formal Resource/Datum docs for
    GEECS native files.  The current lab Tiled server does not yet have readers
    for those custom asset specs, so letting TiledWriter register them as
    external data sources aborts the scan on ``stop``.  Until the server has
    GEECS-aware adapters, Tiled stores the datum-id strings in its event table.
    """
    for data_key in doc.get("data_keys", {}).values():
        if str(data_key.get("source", "")).startswith("geecs://"):
            data_key.pop("external", None)
            data_key["geecs_external_asset"] = True
    return doc


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


def subscribe_tiled(
    run_engine,
    tiled_uri: str | None = None,
    api_key: str | None = None,
) -> int | None:
    """Subscribe a TiledWriter to *run_engine*; return the token or ``None``.

    With no explicit ``tiled_uri``, the location comes from
    :func:`read_tiled_config`.  Silently skips (warning-level log) if
    ``tiled[client]`` is not installed, no URI is configured, or the server is
    unreachable — the caller remains functional without Tiled.
    """
    if tiled_uri is None:
        tiled_uri, api_key = read_tiled_config()
    if not tiled_uri:
        logger.warning("No Tiled URI configured — Tiled storage disabled")
        return None

    try:
        from bluesky.callbacks.tiled_writer import TiledWriter
        from tiled.client import from_uri
    except ImportError:
        logger.warning(
            "tiled not installed — Tiled storage disabled. "
            "Enable with: pip install 'tiled[client]'"
        )
        return None

    try:
        client = from_uri(tiled_uri, api_key=api_key)
        writer = SafeDocumentCallback(
            TiledWriter(client, patches={"descriptor": prepare_descriptor_for_tiled}),
            label="TiledWriter",
        )
        token = run_engine.subscribe(writer)
        logger.info("TiledWriter subscribed — catalog at %s", tiled_uri)
        return token
    except Exception:
        logger.warning(
            "Could not connect TiledWriter to %s — Tiled storage disabled",
            tiled_uri,
            exc_info=True,
        )
        return None
