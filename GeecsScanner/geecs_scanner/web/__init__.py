"""The FastAPI layer over :mod:`geecs_scanner.service`.

Three lines per route: take the body, call the service, return the model.
:func:`create_app` is the process (its own port and unit); :func:`create_scanner_router`
is the same surface as a router, so a host could mount it under a prefix
the way the portal mounts the logbook — kept at zero cost even though the
scanner ships as its own process.
"""

from geecs_scanner.web.app import create_app, create_scanner_router

__all__ = ["create_app", "create_scanner_router"]
