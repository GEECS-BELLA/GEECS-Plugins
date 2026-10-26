"""Canopy with the GEECS module mounted in-process at ``/geecs``.

Run in place of ``canopy.main:app``, in Canopy's own image or venv, with no
patch to Canopy::

    uvicorn geecs_canopy_module.app:app --host 0.0.0.0 --port 8000

Canopy's lifespan (which binds Hare's broker) runs unchanged; the module
reaches Hare through ``canopy.hare.gateway`` and gates its routes with
Canopy's own key dependencies. Its module tag is ``geecs``, which Canopy's
fixed ``MODULES`` set does not know: only ``@*`` keys reach it.
Environment: ``GEECS_CANOPY_JOURNAL`` (SQLite path), ``GEECS_CANOPY_WINDOW_S``
(freshness window), ``GEECS_CANOPY_PUT_TIMEOUT_S`` (put-callback timeout).
"""

from __future__ import annotations

import os

from canopy.core.errors import CanopyError, register_canopy_error_handler
from canopy.core.modules import tag_module
from canopy.core.security import require_read_access, require_write_access
from canopy.hare import gateway
from canopy.main import app

from geecs_canopy_module.family import load_devices
from geecs_canopy_module.journal import Journal
from geecs_canopy_module.routes import HareRefused, WriteValue, build_app


class GatewayHare:
    """The Hare port over Hare's in-process gateway (the broker Canopy's lifespan bound)."""

    async def get(self, pv: str) -> dict[str, object]:
        """Read through ``gateway.get_value``: the cache when monitored, else a one-shot read."""
        try:
            return (await gateway.get_value(pv)).model_dump(by_alias=True, mode="json")
        except CanopyError as exc:
            error = f"{exc.error_code.value}: {exc.detail}"
            return {"pv": pv, "value": None, "connected": False, "error": error}

    async def put(
        self, pv: str, value: WriteValue, *, wait: bool, timeout: float | None
    ) -> None:
        """Put through the broker, as Hare's own PUT route does; refusals keep Hare's code."""
        try:
            await gateway.get_broker().put(pv, value, wait=wait, timeout=timeout)
        except CanopyError as exc:
            raise HareRefused(
                exc.status_code, exc.error_code.value, exc.detail
            ) from exc


geecs_app = build_app(
    GatewayHare(),
    Journal(os.environ.get("GEECS_CANOPY_JOURNAL", "geecs_journal.sqlite3")),
    load_devices(),
    read_access=require_read_access,
    write_access=require_write_access,
    window_s=float(os.environ.get("GEECS_CANOPY_WINDOW_S", "1.0")),
    put_timeout_s=float(os.environ.get("GEECS_CANOPY_PUT_TIMEOUT_S", "30")),
)
register_canopy_error_handler(geecs_app)
tag_module(geecs_app, "geecs")
app.mount("/geecs", geecs_app)
