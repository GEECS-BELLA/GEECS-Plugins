"""Startup import warm-up: load the cycle-bearing packages on one thread (#778).

Several console daemon threads lazily import packages whose module graphs
contain import cycles, and they all spawn within the first second of a
launch:

- ``qs-doc-stream`` (``app/scan_monitor.py``): ``bluesky.callbacks.zmq`` →
  ``bluesky`` → ``bluesky._vendor.super_state_machine``.
- ``qs-console-stream`` and the manager status poll
  (``ZmqQueueClient._manager``): ``bluesky_queueserver_api`` →
  ``bluesky_queueserver`` → ``bluesky``.
- ``console-health-poll`` (``services/health.py``):
  ``geecs_bluesky.tiled_integration`` → ``geecs_data_utils.tiled_catalog``
  → the ``geecs_data_utils`` package ``__init__`` → ``tiled_catalog``.
- ``console-idle-scan-probe`` (``services/ops_paths.py``):
  ``geecs_data_utils.scan_paths`` → the same package ``__init__``.

A cycle imports fine on ONE thread — importlib hands back the partially
initialised module — but when two threads first-import the same cycle
concurrently each holds a per-module lock the other needs.  importlib
detects that and raises ``_DeadlockError`` in one thread while the other
sees a "partially initialized module (most likely due to a circular
import)".  Which thread loses is timing dependent, so the symptom moved
around between launches: a dead document stream one time, a blank idle
scan-number display the next.

:func:`warm_imports` imports each such module once, synchronously, on the
caller's thread; :class:`~geecs_console.app.main_window.MainWindow` calls
it first thing, before any controller spawns a thread, so every later
lazy import finds a fully initialised module in ``sys.modules``.  The lazy
imports themselves stay where they are — this is a load-*order* fix, not a
dependency change: each module's offline import-safety is unchanged, and a
missing dependency still surfaces at the call site that needs it.  The
warm-up is best-effort: an import failure is logged and skipped, never
raised.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Sequence

logger = logging.getLogger(__name__)

#: The modules the console's daemon threads lazily import that carry (or
#: pull in) an import cycle.  Add to this tuple when a new daemon thread's
#: lazy import reaches ``bluesky``, ``bluesky_queueserver_api`` or
#: ``geecs_data_utils`` through a new module — the entries name the exact
#: modules the threads import so a warm-up miss is greppable.
WARM_MODULES: tuple[str, ...] = (
    # DocumentStreamWorker._run (qs-doc-stream)
    "bluesky.callbacks.zmq",
    # ZmqQueueClient._manager (status poll) and ConsoleStreamWorker._run
    "bluesky_queueserver_api.zmq",
    "bluesky_queueserver_api.console_monitor",
    # ops_paths.todays_scan_folder (idle scan-number probe)
    "geecs_data_utils.scan_paths",
    # GatewayTiledDbHealth._tiled_uri (health poll)
    "geecs_bluesky.tiled_integration",
)


def warm_imports(modules: Sequence[str] = WARM_MODULES) -> list[str]:
    """Import *modules* on the calling thread, tolerating failures.

    Call this on the GUI thread **before** any daemon thread spawns.
    Importing is idempotent (``sys.modules`` caches), so calling it again
    costs nothing.

    Parameters
    ----------
    modules : sequence of str
        Fully qualified module names to import; defaults to
        :data:`WARM_MODULES`.

    Returns
    -------
    list of str
        The names that failed to import (logged at WARNING; the lazy
        import site that actually needs the module reports the real
        error at use time).  Empty when every import succeeded.
    """
    started = time.perf_counter()
    failed: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 — best-effort; the real user of the module reports
            logger.warning("startup import warm-up: %s failed: %s", name, exc)
            failed.append(name)
    logger.debug(
        "startup import warm-up: %d module(s) in %.2f s (%d failed)",
        len(modules),
        time.perf_counter() - started,
        len(failed),
    )
    return failed
