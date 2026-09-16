"""``GET /api/events`` — the Server-Sent-Events stream the page lives on.

One stream, three event types, each a JSON object:

- ``status``   — the manager poll (:class:`~geecs_scanner.service.models.StatusOut`),
  sent every round so the page can show how long ago the manager answered;
- ``progress`` — the latest-run picture from the document stream, sent
  when it changes;
- ``log``      — the run's ``scan.log`` lines (:mod:`geecs_scanner.service.scanlog`),
  one frame per chunk, from the folder the start document names; a new
  run replays its file from the top, and a folder this host cannot read
  is said once (``available: false``);
- ``console``  — one manager console-output line each, carrying
  ``id: <epoch>:<seq>`` so the browser's own reconnect resumes where it
  left off (``Last-Event-ID``); ``?since=<seq>`` is the manual form. A new
  epoch (the scanner restarted) replays from the start and the page clears
  its tail.

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


def _frame(event: str, payload: object, event_id: str | None = None) -> str:
    head = f"id: {event_id}\n" if event_id else ""
    return f"{head}event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def _resume_from(request: Request, since: int, epoch: str) -> int:
    """Where to resume the console cursor after a browser reconnect.

    ``EventSource`` reconnects by itself and sends ``Last-Event-ID`` — the
    ``<epoch>:<seq>`` the console frames carry — so a blip resumes where it
    left off. A different epoch means this process restarted and its ``seq``
    started over: replay from the start (the page clears its tail on the
    epoch change). The ``?since=`` query is the manual form of the same.
    """
    last = request.headers.get("last-event-id", "")
    if last:
        got_epoch, _, seq = last.rpartition(":")
        if got_epoch == epoch and seq.isdigit():
            return int(seq)
        return 0
    return since


def register(router: APIRouter, service: ScannerService) -> None:
    """Attach the events route to *router*."""

    @router.get("/api/events")
    async def events(
        request: Request, since: int = 0, once: int = 0
    ) -> StreamingResponse:
        """Status, progress and console lines as Server-Sent Events."""

        async def gen() -> AsyncIterator[str]:
            last_progress: str | None = None
            last_optimization: str | None = None
            epoch = service.streams.epoch
            cursor = _resume_from(request, since, epoch)
            log_folder: str | None = None
            log_offset = 0
            log_said_missing = False
            last_sent = time.monotonic()
            while True:
                # Every round, changed or not: the page shows how long ago the
                # manager last answered, and a quiet manager must read as
                # "answered, unchanged", never as "unheard-from".
                status = await anyio.to_thread.run_sync(service.status)
                last_sent = time.monotonic()
                yield _frame("status", status.model_dump())
                progress = service.progress()
                p = progress.model_dump_json()
                if p != last_progress:
                    last_progress = p
                    last_sent = time.monotonic()
                    yield _frame("progress", progress.model_dump())
                optimization = service.optimization()
                encoded = optimization.model_dump_json()
                if encoded != last_optimization:
                    last_optimization = encoded
                    yield _frame("optimization", optimization.model_dump())
                if progress.scan_folder and progress.scan_folder != log_folder:
                    log_folder, log_offset, log_said_missing = (
                        progress.scan_folder,
                        0,
                        False,
                    )
                if log_folder:
                    chunk = await anyio.to_thread.run_sync(
                        service.scan_log, log_offset, log_folder
                    )
                    if chunk.available:
                        log_offset = chunk.offset
                        if chunk.lines:
                            last_sent = time.monotonic()
                            yield _frame("log", chunk.model_dump())
                    elif not log_said_missing:
                        log_said_missing = True
                        yield _frame("log", chunk.model_dump())
                for line in service.console_since(cursor):
                    cursor = line.seq
                    last_sent = time.monotonic()
                    payload = line.model_dump()
                    payload["epoch"] = epoch
                    yield _frame("console", payload, f"{epoch}:{line.seq}")
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
