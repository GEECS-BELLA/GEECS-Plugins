"""The one sync/async bridge in geecs-core (DESIGN.md rule 2).

A single asyncio event loop runs on a lazily started daemon thread, shared by
every :class:`~geecs_core.client.geecs_device.GeecsDevice` in the process.
Synchronous callers hand coroutines to it via :func:`run_sync`; the transport
objects those coroutines create are therefore all bound to this one loop.

Services with their own event loop (the gateways) must never import this
module — they consume ``geecs_core.transport`` natively.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine

_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()


def get_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background event loop, starting it on first use.

    Returns
    -------
    asyncio.AbstractEventLoop
        A running loop owned by a daemon thread named
        ``geecs-core-client-loop``. The thread dies with the process; sockets
        held by live devices are cleaned up by their ``close()`` calls, not by
        loop shutdown.
    """
    global _loop
    with _lock:
        if _loop is None or _loop.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name="geecs-core-client-loop",
                daemon=True,
            )
            thread.start()
            _loop = loop
        return _loop


def run_sync(coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
    """Run *coro* on the shared loop and block until it finishes.

    Parameters
    ----------
    coro : Coroutine
        The coroutine to execute.
    timeout : float, optional
        Seconds to wait for the result. ``None`` (default) waits forever —
        callers normally rely on the transport's own timeouts instead, so a
        hung wait here indicates a missing transport timeout, not a slow
        device.

    Returns
    -------
    Any
        The coroutine's return value. Exceptions raised inside the coroutine
        (``GeecsCommandFailedError`` etc.) propagate to the caller unchanged.
    """
    future = asyncio.run_coroutine_threadsafe(coro, get_loop())
    return future.result(timeout)
