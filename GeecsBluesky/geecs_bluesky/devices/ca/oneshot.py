"""One-shot blocking CA reads on a shared persistent event loop.

aioca caches its CA channels per asyncio event loop, so a per-call
``asyncio.run()`` (a fresh loop every read) re-creates the channel on
every call and strands the previous loop's cache entries for the process
lifetime — the aioca loop-cache leak.  Sync callers that need an
occasional blocking read (the console health probe, the pre-submit
preflight's liveness/staleness samples) go through this module instead:
it owns **one** persistent asyncio loop in one daemon thread and serves
every read through it, so each PV's channel is created once and reused.

This is the one blessed one-shot blocking read; long-lived subscriptions
belong on a caller-owned loop (the console device-panel pattern), and
in-plan reads belong to ophyd-async signals — never this module.  The
one sanctioned in-plan exception is the pre-claim liveness probe
(:func:`~geecs_bluesky.devices.ca.liveness.probe_disconnected`, used by
the queue-plan preamble): it runs before the devices exist, so there is
no signal to read yet — and it must gather its reads concurrently
(:func:`try_caget_many`) so the worst case blocks the RE loop for one
timeout budget, never one per device.

``aioca`` is imported lazily on first use (the ``ca`` extra), so the
module itself imports anywhere.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional

#: Grace added to the CA timeout for the cross-thread ``future.result``
#: backstop, so a wedged loop surfaces as a timeout instead of a hang.
_RESULT_GRACE_S = 2.0

_loop_lock = threading.Lock()
_loop: Optional[asyncio.AbstractEventLoop] = None


def _shared_loop() -> asyncio.AbstractEventLoop:
    """Return the process-wide reader loop, starting its thread on first use."""
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="geecs-ca-oneshot", daemon=True
            )
            thread.start()
            _loop = loop
        return _loop


def caget_once(pv: str, *, timeout: float, datatype: Any = None) -> Any:
    """One blocking CA read of *pv* on the shared loop.

    Parameters
    ----------
    pv : str
        Bare PV name (no ``ca://`` prefix).
    timeout : float
        CA read budget in seconds.
    datatype :
        Passed to ``caget``.  **Must be ``str`` for enum PVs whose choice
        string is compared** (e.g. the gateway's ``CONNECTED`` is a
        DBR_ENUM — a native read returns the integer index, and
        ``str(1)`` never equals ``"Disconnected"``).  ``None`` reads the
        channel's native type.

    Returns
    -------
    Any
        The channel value.

    Raises
    ------
    Exception
        Whatever ``aioca`` raises on failure or timeout (``CANothing``),
        or ``concurrent.futures.TimeoutError`` if the shared loop itself
        is wedged past the grace budget.
    """
    from aioca import caget

    async def _get() -> Any:
        if datatype is None:
            return await caget(pv, timeout=timeout)
        return await caget(pv, datatype=datatype, timeout=timeout)

    future = asyncio.run_coroutine_threadsafe(_get(), _shared_loop())
    try:
        return future.result(timeout=timeout + _RESULT_GRACE_S)
    except BaseException:
        future.cancel()
        raise


def try_caget_once(pv: str, *, timeout: float, datatype: Any = None) -> Any:
    """:func:`caget_once`, with every failure read as ``None``.

    For fail-open probes where an unreadable PV is not a verdict (the
    liveness doctrine) — callers that must distinguish failure use
    :func:`caget_once` directly.
    """
    try:
        return caget_once(pv, timeout=timeout, datatype=datatype)
    except Exception:
        return None


def try_caget_many(
    pvs: list[str], *, timeout: float, datatype: Any = None
) -> list[Any]:
    """Concurrent fail-open reads of *pvs* on the shared loop.

    One gather, one ``timeout`` budget for the whole batch — N dead PVs
    cost the same wall time as one (the sequential per-PV alternative
    would block the caller — possibly the RE loop — for ``N × timeout``).
    Each failed read is ``None`` in the result, positionally matching
    *pvs* — including the whole batch when ``aioca`` itself is missing
    (the ``ca`` extra): fail-open covers the import too, matching
    :func:`try_caget_once`.
    """
    try:
        from aioca import caget
    except Exception:
        return [None] * len(pvs)

    async def _one(pv: str) -> Any:
        try:
            if datatype is None:
                return await caget(pv, timeout=timeout)
            return await caget(pv, datatype=datatype, timeout=timeout)
        except Exception:
            return None

    async def _all() -> list[Any]:
        return list(await asyncio.gather(*(_one(pv) for pv in pvs)))

    future = asyncio.run_coroutine_threadsafe(_all(), _shared_loop())
    try:
        return future.result(timeout=timeout + _RESULT_GRACE_S)
    except BaseException:
        future.cancel()
        return [None] * len(pvs)
