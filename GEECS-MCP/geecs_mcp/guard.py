"""The tools-never-raise backstop, in one place.

Every tool's wrapper routes through here, so a bug anywhere in an impl
reaches the agent as an ``internal_error`` envelope instead of a protocol
exception, and a service's verdict reaches it as that service's own kind
and words (:mod:`geecs_mcp.clients`).
"""

from __future__ import annotations

import logging

from geecs_mcp import errors
from geecs_mcp.clients import ServiceRefusal, ServiceUnavailable
from geecs_mcp.identity import InvalidArgument

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
    except InvalidArgument as exc:  # an argument this server checks itself
        return errors.make_error("invalid_request", str(exc))
    except Exception as exc:
        logger.exception("tool %s failed", getattr(impl, "__name__", impl))
        return errors.make_error("internal_error", str(exc))
