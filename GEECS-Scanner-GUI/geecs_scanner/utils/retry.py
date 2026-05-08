"""Centralized retry-with-backoff utility.

Usage
-----
::

    from geecs_scanner.utils.retry import retry
    from geecs_scanner.engine.dialog_request import DEVICE_COMMAND_ERRORS

    result = retry(
        lambda: device.set(var, value),
        attempts=3,
        delay=0.5,
        catch=DEVICE_COMMAND_ERRORS,
        on_retry=lambda exc, n: logger.debug("[%s] retry %d: %s", device_name, n, exc),
    )

If all attempts are exhausted the last caught exception is re-raised.  Call
sites that want to surface a higher-level :class:`~geecs_scanner.utils.exceptions.DeviceCommandError`
should wrap the call::

    try:
        result = retry(lambda: device.set(var, value), ...)
    except DEVICE_COMMAND_ERRORS as exc:
        raise DeviceCommandError(device_name, f"set {var}") from exc
"""

from __future__ import annotations

import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 1.0,
    catch: tuple[type[BaseException], ...],
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> T:
    """Call *fn*, retrying on any exception in *catch*.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to invoke.  Must be safe to call multiple times.
    attempts : int
        Maximum number of calls including the first.  Must be >= 1.
    delay : float
        Seconds to sleep before the *second* attempt.
    backoff : float
        Multiplier applied to *delay* after each failure.  ``1.0`` gives a
        constant interval; ``2.0`` gives exponential backoff.
    catch : tuple of exception types
        Exception types that trigger a retry.  Any other exception propagates
        immediately without sleeping or incrementing the attempt counter.
    on_retry : callable, optional
        Called as ``on_retry(exc, attempt_number)`` after each failed attempt
        and before sleeping.  ``attempt_number`` is 1-based.  Intended for
        per-attempt ``DEBUG`` logging.

    Returns
    -------
    T
        Return value of *fn* on success.

    Raises
    ------
    Exception
        The last caught exception when all *attempts* are exhausted.
    ValueError
        If *attempts* < 1.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    last_exc: Exception | None = None
    wait = delay

    for attempt in range(attempts):
        try:
            return fn()
        except catch as exc:
            last_exc = exc
            if on_retry is not None:
                on_retry(exc, attempt + 1)
            if attempt < attempts - 1:
                time.sleep(wait)
                wait *= backoff

    raise last_exc  # type: ignore[misc]  # always set when attempts >= 1
