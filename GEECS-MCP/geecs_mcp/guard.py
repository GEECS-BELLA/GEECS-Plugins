"""The tools-never-raise backstop, in one place.

Every tool's wrapper routes through here, so a bug anywhere in an impl
reaches the agent as an ``internal_error`` envelope instead of a protocol
exception, and a service's verdict reaches it as that service's own kind
and words (:mod:`geecs_mcp.clients`).
"""

from __future__ import annotations

import logging

import anyio

from geecs_mcp import errors
from geecs_mcp.clients import ServiceRefusal, ServiceUnavailable

logger = logging.getLogger("geecs_mcp.guard")


async def run_async(impl, *args, **kwargs) -> str:
    """Await one async impl; refusals, outages and bugs become envelopes."""
    try:
        return await impl(*args, **kwargs)
    except ServiceRefusal as exc:
        return errors.make_error(exc.kind, exc.message, **exc.extra)
    except ServiceUnavailable as exc:
        return errors.make_error(
            "service_unreachable", exc.message, service=exc.service, url=exc.url
        )
    except ValueError as exc:  # an argument check (identity_for, finiteness)
        return errors.make_error("invalid_request", str(exc))
    except Exception as exc:
        logger.exception("tool %s failed", getattr(impl, "__name__", impl))
        return errors.make_error("internal_error", str(exc))


async def run_guarded(impl, *args) -> str:
    """Dispatch one SYNC impl to a worker thread; a raise becomes an envelope.

    The analysis domain's wrapper until PR E moves it onto the portal's
    routes (its impls read the share synchronously).
    """
    try:
        return await anyio.to_thread.run_sync(impl, *args)
    except Exception as exc:
        logger.exception("tool %s failed", getattr(impl, "__name__", impl))
        return errors.make_error("internal_error", str(exc))
