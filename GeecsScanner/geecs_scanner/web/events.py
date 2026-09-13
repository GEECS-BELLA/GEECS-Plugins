"""``GET /api/events`` — the Server-Sent-Events stream the page lives on.

One stream, three event types, each a JSON object:

- ``status``   — the manager poll (:class:`~geecs_scanner.service.models.StatusOut`),
  sent when it changes;
- ``progress`` — the latest-run picture from the document stream, sent
  when it changes;
- ``console``  — one manager console-output line each, with a ``seq`` the
  page keeps so a reconnect resumes where it left off
  (``?since=<seq>``).

SSE, not WebSocket: one direction, plain HTTP, survives every reverse
proxy with one "no buffering" line, and the browser reconnects on its own.
The generator polls the service (each poll a bounded manager round trip on
the threadpool) and sleeps between rounds; a comment line every fifteen
seconds keeps idle connections open.  ``?once=1`` sends one round and
closes — for tests and for ``curl``.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import anyio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from geecs_scanner.service.scanner import ScannerService

#: Seconds between polls of the manager while a page is connected.
POLL_S = 1.0
#: Seconds between keepalive comments on an otherwise quiet stream.
KEEPALIVE_S = 15.0

_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # nginx: never buffer this response
    "Connection": "keep-alive",
}


def _frame(event: str, payload: object) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def register(router: APIRouter, service: ScannerService) -> None:
    """Attach the events route to *router*."""

    @router.get("/api/events")
    async def events(
        request: Request, since: int = 0, once: int = 0
    ) -> StreamingResponse:
        """Status, progress and console lines as Server-Sent Events."""

        async def gen() -> AsyncIterator[str]:
            last_status: str | None = None
            last_progress: str | None = None
            cursor = since
            last_sent = time.monotonic()
            while True:
                status = await anyio.to_thread.run_sync(service.status)
                s = status.model_dump_json()
                if s != last_status:
                    last_status = s
                    last_sent = time.monotonic()
                    yield _frame("status", status.model_dump())
                progress = service.progress()
                p = progress.model_dump_json()
                if p != last_progress:
                    last_progress = p
                    last_sent = time.monotonic()
                    yield _frame("progress", progress.model_dump())
                for line in service.console_since(cursor):
                    cursor = line.seq
                    last_sent = time.monotonic()
                    yield _frame("console", line.model_dump())
                if once:
                    return
                if time.monotonic() - last_sent > KEEPALIVE_S:
                    last_sent = time.monotonic()
                    yield ": keepalive\n\n"
                if await request.is_disconnected():
                    return
                await anyio.sleep(POLL_S)

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers=_HEADERS
        )
