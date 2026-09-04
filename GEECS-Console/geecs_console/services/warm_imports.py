"""Startup import warm-up: load the cycle-bearing packages on one thread (#778).

Python's per-module import lock serialises two threads importing the
*same* module.  It does NOT protect a package whose ``__init__`` imports
its own submodules when two threads enter that package through
*different* submodules: thread A holds submodule X's lock and, inside the
package ``__init__``, waits for Y; thread B holds Y's lock and waits for
X.  importlib detects the cycle and raises ``_DeadlockError`` in one
thread while the other sees a "partially initialized module (most likely
due to a circular import)".  Which thread loses is timing dependent.

The console's daemon threads spawn within the first second of a launch
and hit exactly this shape (the racing pairs were reproduced cold, and
verified quiet after the warm-up, in the #784 review):

- ``bluesky``: the document stream (``qs-doc-stream``,
  ``DocumentStreamWorker._run`` → ``bluesky.callbacks.zmq``) vs the health
  poll (``console-health-poll``, ``GatewayTiledDbHealth._check_gateway`` →
  ``geecs_bluesky.devices.ca._pv`` → ophyd_async → ``bluesky.protocols``).
  The observed ``_DeadlockError`` is on
  ``bluesky._vendor.super_state_machine.errors``.
- ``geecs_data_utils``: the idle scan-number probe
  (``console-idle-scan-probe``, ``ops_paths.todays_scan_folder`` →
  ``geecs_data_utils.scan_paths``) vs the health poll
  (``GatewayTiledDbHealth._tiled_uri`` → ``geecs_bluesky.tiled_integration``
  → ``geecs_data_utils.tiled_catalog``).  The observed error is
  ``read_tiled_config`` "partially initialized".

:func:`warm_imports` imports one entry per cycle-bearing package once,
synchronously, on the caller's thread;
:class:`~geecs_console.app.main_window.MainWindow` calls it first thing,
before any controller spawns a thread, so every later lazy import finds
the package fully initialised in ``sys.modules``.  The lazy imports
themselves stay where they are — this is a load-*order* fix, not a
dependency change: each module's offline import-safety is unchanged, and a
missing dependency still surfaces at the call site that needs it.  The
warm-up is best-effort: an import failure is logged and skipped, never
raised.

Deliberately NOT warmed (heavy — aioca/ophyd — and safe once ``bluesky``
is loaded, because they reach it through an already-initialised package):
``geecs_bluesky.devices`` (health poll, device panel) and
``geecs_bluesky.plans`` (the console stream's ``_failed_move_prefix``).
``bluesky_queueserver_api`` is not warmed either: it does not import
``bluesky`` and has no cycle of its own.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Sequence

logger = logging.getLogger(__name__)

#: The modules the console's daemon threads lazily import that enter a
#: package with import cycles (``bluesky``, ``geecs_data_utils``).  One
#: entry per racing import site, named exactly as the thread imports it so
#: a miss is greppable.  A new daemon-thread lazy import that reaches one
#: of those packages goes here; ``tests/test_warm_imports.py`` greps the
#: thread-body modules for the ``bluesky`` / ``geecs_data_utils`` sites.
WARM_MODULES: tuple[str, ...] = (
    # DocumentStreamWorker._run (qs-doc-stream) — loads all of bluesky
    "bluesky.callbacks.zmq",
    # ops_paths.todays_scan_folder (console-idle-scan-probe)
    "geecs_data_utils.scan_paths",
    # GatewayTiledDbHealth._tiled_uri (console-health-poll)
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
