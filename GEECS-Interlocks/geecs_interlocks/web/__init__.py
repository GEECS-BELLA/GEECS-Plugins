"""The web client: a page on the kit, served by the interlock server itself.

Routes (JSON for the browser only — every file this package writes is YAML):

``GET /``
    The page.
``GET /api/status``
    The current snapshot.
``GET /api/stream``
    Server-sent events: one ``data:`` line per new snapshot, at most
    :data:`STREAM_HZ` a second. The page treats a silence past its own
    threshold as stale; it never keeps showing a verdict it has stopped
    hearing.
``GET /api/trend?name=``
    One series per check of an interlock, over the monitor's trend window.
``GET /api/events?limit=``
    The latest state changes, newest first.

``POST /api/arm`` ``{"operator", "interlock"?}``
    Arm the settings checks (of one interlock, when named): ``Monitor.arm``.
``POST /api/reload`` ``{"operator"}``
    Re-read the running configuration file: ``Engine.reload``.

The two actions act on the server, never on the experiment. Each needs the
operator's name (400 without it), is logged with it, and answers the
engine's own message: 200 when done, 409 when refused (nothing to arm, a
setting with no value yet, a file that does not validate). The page asks
for a confirmation first. Configurations are edited in the configurator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from geecs_web_theme.web import (
    ForwardedPrefixMiddleware,
    RevalidatingStaticFiles,
    make_templates,
    mount_theme,
)

from geecs_interlocks.engine import Engine
from geecs_interlocks.monitor import Event, InterlockStatus, Snapshot
from pydantic import BaseModel, Field

from geecs_interlocks.present import (
    KIT_STATE,
    is_age,
    number_text,
    stall_age,
    status_is_age,
    verdict_text,
)

logger = logging.getLogger(__name__)

_PKG = Path(__file__).parent
#: Snapshots sent per second at most, per browser.
STREAM_HZ = 4.0
#: Events ``/api/events`` returns when the page asks for no number.
EVENTS_DEFAULT = 100


class ActionIn(BaseModel):
    """An operator's request to arm or reload."""

    operator: str | None = Field(None, description="Who pressed the button")
    interlock: str | None = Field(
        None, description="Arm this interlock only (arm; ignored by reload)"
    )


class ActionOut(BaseModel):
    """The engine's answer to an action."""

    ok: bool
    message: str
    armed: list[str] = Field(default_factory=list)


def _operator(body: ActionIn) -> str:
    operator = (body.operator or "").strip()
    if not operator:
        raise HTTPException(400, "enter your name (operator) before acting")
    return operator


def _answer(out: ActionOut) -> JSONResponse:
    return JSONResponse(out.model_dump(), status_code=200 if out.ok else 409)


def status_json(status: InterlockStatus) -> dict[str, Any]:
    """One interlock as the page reads it."""
    return {
        "name": status.name,
        "description": status.description,
        "condition": status.condition,
        "state": status.state,
        "kit_state": KIT_STATE[status.state],
        "diagnostic": status.diagnostic,
        "value": number_text(status.value, status_is_age(status)),
        "since": status.since,
        "armable": status.armable,
        "checks": [
            {
                "condition": c.condition,
                "unsafe": c.unsafe,
                "diagnostic": c.diagnostic,
                "value": number_text(c.value, is_age(c.kind)),
            }
            for c in status.checks
        ],
    }


def snapshot_json(snapshot: Snapshot, engine: Engine) -> dict[str, Any]:
    """The whole snapshot as the page reads it: the verdict, every interlock, the stream."""
    headline, detail = verdict_text(snapshot)
    safe = snapshot.safe
    stalled = stall_age(snapshot)
    if stalled is not None:
        # Every reader treats an old snapshot as no snapshot, as the stream does.
        safe, headline = False, "Not updating"
        detail = f"evaluation stopped {stalled:.0f} s ago"
    order = ("tripped", "waiting", "settling", "ok", "off")
    statuses = sorted(snapshot.statuses, key=lambda s: order.index(s.state))
    return {
        "taken": snapshot.taken,
        "sent": time.time(),
        "safe": safe,
        "stalled": stalled is not None,
        "headline": headline,
        "detail": detail,
        "kit_state": "ok" if safe else "failed",
        "source": snapshot.source,
        "configuration": engine.path.stem if engine.path is not None else None,
        "clients": len(engine.stream.clients()),
        "statuses": [status_json(s) for s in statuses],
    }


def event_json(event: Event) -> dict[str, Any]:
    """One state change as the page reads it."""
    return {
        "at": event.at,
        "name": event.name,
        "old": event.old,
        "new": event.new,
        "kit_state": KIT_STATE[event.new],
        "diagnostic": event.diagnostic,
    }


def create_app(engine: Engine, *, experiment: str | None = None) -> FastAPI:
    """The web client's app over *engine*."""
    app = FastAPI(title="GEECS interlocks", docs_url=None, redoc_url=None)
    app.add_middleware(ForwardedPrefixMiddleware)
    mount_theme(app)
    app.mount(
        "/static",
        RevalidatingStaticFiles(directory=str(_PKG / "static")),
        name="static",
    )
    _add_page(app, engine, experiment)
    _add_api(app, engine)
    _add_actions(app, engine)
    return app


def _add_page(app: FastAPI, engine: Engine, experiment: str | None) -> None:
    templates = make_templates(_PKG / "templates")

    @app.get("/", response_class=HTMLResponse)
    def page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "experiment": experiment or "",
                "configuration": engine.path.stem if engine.path else "",
                "stream_hz": STREAM_HZ,
            },
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}


def _add_api(app: FastAPI, engine: Engine) -> None:
    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return snapshot_json(engine.snapshot(), engine)

    @app.get("/api/trend")
    def trend(name: str) -> dict[str, Any]:
        series = engine.monitor.trend(name)
        if not series:
            raise HTTPException(404, f"no interlock named {name!r}")
        return {
            "name": name,
            "series": [
                {
                    "condition": s.condition,
                    "limit": s.limit,
                    "t": [p[0] for p in s.points],
                    "v": [p[1] for p in s.points],
                }
                for s in series
            ],
        }

    @app.get("/api/events")
    def events(
        limit: int = EVENTS_DEFAULT, name: str | None = None
    ) -> list[dict[str, Any]]:
        kept = [e for e in engine.events() if name is None or e.name == name]
        return [event_json(e) for e in reversed(kept[-max(1, limit) :])]

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        async def frames() -> AsyncIterator[str]:
            last: Snapshot | None = None
            while not await request.is_disconnected():
                snapshot = engine.snapshot()
                if snapshot is not last:
                    last = snapshot
                    yield f"data: {json.dumps(snapshot_json(snapshot, engine))}\n\n"
                await asyncio.sleep(1.0 / STREAM_HZ)

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def _add_actions(app: FastAPI, engine: Engine) -> None:
    @app.post("/api/arm", response_model=ActionOut)
    def arm(body: ActionIn) -> JSONResponse:
        operator = _operator(body)
        name = (body.interlock or "").strip() or None
        try:
            armed = engine.arm(name)
        except ValueError as exc:  # NothingToArm too
            logger.warning("arm %s by %s refused: %s", name or "(all)", operator, exc)
            return _answer(ActionOut(ok=False, message=str(exc)))
        message = f"armed {', '.join(armed)}"
        logger.info("%s by %s", message, operator)
        return _answer(ActionOut(ok=True, message=message, armed=armed))

    @app.post("/api/reload", response_model=ActionOut)
    def reload(body: ActionIn) -> JSONResponse:
        operator = _operator(body)
        logger.info("reload asked by %s", operator)
        result = engine.reload()
        log = logger.info if result.ok else logger.warning
        log("%s (by %s)", result.message, operator)
        return _answer(ActionOut(ok=result.ok, message=result.message))


__all__ = [
    "ActionIn",
    "ActionOut",
    "EVENTS_DEFAULT",
    "STREAM_HZ",
    "create_app",
    "event_json",
    "snapshot_json",
    "status_json",
]
