"""GEECS Scanner — the web scanner console's service layer and HTTP API.

The third web surface on the GEECS surface kit, and the replacement for
the PySide6 GEECS-Console (#869).
Two layers, one seam:

- :mod:`geecs_scanner.service` — pure Python over
  :mod:`geecs_bluesky.qs_client` and the configs resolver.  Every function
  takes the injected client and returns a Pydantic model; no FastAPI, no
  JSON strings.  This is the tested surface, and the layer any later
  client (an OSPREY panel, an MCP) imports.
- :mod:`geecs_scanner.web` — the FastAPI app and router over that
  service: the JSON API, the Server-Sent-Events stream that carries what
  the manager and the document stream say, and (from 0.2.0) the page.

Invariants the API keeps (the whole point of the rebuild's client-side
expansion): the client expands a
:class:`geecs_schemas.Preset`; nothing worker-side re-derives detectors or
points.  The scanner never binds a variable to a device itself — it always
goes through :func:`geecs_bluesky.qs_client.expand_preset` and the
resolver.  ``md["geecs"]`` is provenance only.  The scan number is
claimed worker-side; the page reads it from the start document.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("geecs-scanner")
except PackageNotFoundError:  # pragma: no cover — source checkout, not installed
    __version__ = "0.0.0+source"

__all__ = ["__version__"]
