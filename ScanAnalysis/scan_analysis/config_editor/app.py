"""FastAPI surface of the config editor: JSON API + the editor page + its two static assets.

Everything is relative to the router's mount, so it works under
``/configs`` behind the portal's reverse-proxy prefix (its one host) and
on a bare app in tests alike: the page computes its API base from its own
URL, and no absolute portal URL is written here.

API (under the mount)::

    GET  /                          the editor page (the portal's full-page form)
    GET  /static/editor.js|.css     the editor assets (the portal includes them too)
    GET  /api/list                  analyzers + groups (validity, summary), namespaces, git pending
    GET  /api/schema/{kind}         JSON Schema for kind = analyzer (the recipe, its
                                    steps/measure bound to the core's registry) | group
    GET  /api/{kind}s/{id}          one document: raw, etag, yaml, validity
    POST /api/validate/{kind}       {document} -> {ok, errors, canonical, yaml}
    PUT  /api/{kind}s/{ns}/{id}     {document, etag|null} -> saved {etag, yaml, created}; 409 / 422
    DELETE /api/{kind}s/{id}?etag=  -> 204; 409
    POST /api/preview               {document, params} -> image/png (host-provided; 404 without)
    POST /api/preview/summary       {document, params, index} -> image/png: the document's
                                    index-th summary over a few shots (host-provided; 404 without)

Errors are JSON ``{"detail": ...}`` with honest statuses: 404 unknown,
409 conflict (stale etag / duplicate id), 422 invalid document (with the
pydantic locations), 400 for a preview the analyzer refused — never a 500
for a config problem.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from scan_analysis.config_store import (
    ConfigStore,
    ConflictError,
    DocumentInvalid,
    NotFound,
)

logger = logging.getLogger(__name__)

__all__ = ["PreviewFn", "SummaryPreviewFn", "create_editor_router"]

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"
_TEMPLATES = Jinja2Templates(directory=str(_HERE / "templates"))

#: ``(document, params) -> PNG bytes``: render the (validated) document under
#: edit on pixels the host owns.  Raise ``ValueError`` for a refusal the
#: user can act on (400) and ``LookupError`` for a missing shot (404).
PreviewFn = Callable[[Mapping[str, Any], Mapping[str, Any]], bytes]
#: ``(document, params, index) -> PNG bytes``: the document's ``index``-th
#: summary drawn over a few of the host's shots (``params.shots``), the way
#: a run draws it.  Same error ladder; ``LookupError`` also for no summary
#: at ``index``.
SummaryPreviewFn = Callable[[Mapping[str, Any], Mapping[str, Any], int], bytes]

_KINDS = {"analyzer", "group"}


def _kind(value: str) -> str:
    if value not in _KINDS:
        raise HTTPException(status_code=404, detail=f"unknown document kind {value!r}")
    return value


def _plural(kind: str) -> str:
    return kind + "s"


def create_editor_router(
    store: ConfigStore,
    *,
    preview: Optional[PreviewFn] = None,
    summary_preview: Optional[SummaryPreviewFn] = None,
    read_only: bool = False,
    theme_url: str = "/theme",
) -> APIRouter:
    """Build the editor router over *store*.

    Parameters
    ----------
    store : ConfigStore
        The configs tree.
    preview : callable, optional
        Host-provided live preview (the portal renders the current shot);
        without it ``POST /api/preview`` is 404 and the page hides the pane.
    summary_preview : callable, optional
        Host-provided summary preview (the document's summaries over a few
        shots of the current scan); without it ``POST /api/preview/summary``
        is 404 and the page hides that block.
    read_only : bool, default False
        Serve the browser and validation but refuse writes (405).
    theme_url : str, default "/theme"
        Where the host serves ``geecs_web_theme`` — the Data Portal mounts
        it at ``/theme``. The template prefixes it with the request's
        ``root_path`` so a reverse-proxy mount still resolves. There is no
        fallback palette: every host serves the theme, and a copied palette
        is exactly the drift the shared package exists to remove.
    """
    router = APIRouter()

    def guard_write() -> None:
        if read_only:
            raise HTTPException(
                status_code=405, detail="config editor is read-only here"
            )

    @router.get("/", response_class=HTMLResponse)
    def page(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request,
            "editor.html",
            {
                "preview": preview is not None,
                "read_only": read_only,
                "theme_url": theme_url.rstrip("/"),
            },
        )

    @router.get("/static/{name}")
    def static(name: str) -> FileResponse:
        if name not in ("editor.js", "editor.css"):
            raise HTTPException(status_code=404)
        # the asset ships with ScanAnalysis, not the host page that cache-busts
        # it by its own version: revalidate every time (the files are small)
        return FileResponse(_STATIC / name, headers={"Cache-Control": "no-cache"})

    @router.get("/api/list")
    def listing() -> JSONResponse:
        return JSONResponse(
            {
                "analyzers": [e.to_json() for e in store.list("analyzer")],
                "groups": [e.to_json() for e in store.list("group")],
                "namespaces": {
                    "analyzer": store.namespaces("analyzer"),
                    "group": store.namespaces("group"),
                },
                "known_ids": store.known_ids(),
                "pending": store.pending_changes(),
                "preview": preview is not None,
                "summary_preview": summary_preview is not None,
                "read_only": read_only,
            },
            headers={"Cache-Control": "no-cache"},
        )

    @router.get("/api/schema/{kind}")
    def schema(kind: str) -> JSONResponse:
        return JSONResponse(store.schema(_kind(kind)))

    @router.get("/api/{plural}/{doc_id}")
    def read(plural: str, doc_id: str) -> JSONResponse:
        kind = _kind(plural[:-1] if plural.endswith("s") else plural)
        try:
            loaded = store.read(kind, doc_id)
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(loaded.to_json(), headers={"Cache-Control": "no-cache"})

    @router.post("/api/validate/{kind}")
    def validate(kind: str, body: dict = Body(...)) -> JSONResponse:
        document = body.get("document")
        if not isinstance(document, dict):
            raise HTTPException(
                status_code=422, detail="body.document must be a mapping"
            )
        return JSONResponse(store.validate(_kind(kind), document).to_json())

    @router.put("/api/{plural}/{namespace}/{doc_id}")
    def save(
        plural: str, namespace: str, doc_id: str, body: dict = Body(...)
    ) -> JSONResponse:
        guard_write()
        kind = _kind(plural[:-1] if plural.endswith("s") else plural)
        document = body.get("document")
        if not isinstance(document, dict):
            raise HTTPException(
                status_code=422, detail="body.document must be a mapping"
            )
        try:
            saved = store.save(kind, namespace, doc_id, document, etag=body.get("etag"))
        except DocumentInvalid as exc:
            return JSONResponse(
                {"detail": "invalid document", "errors": exc.errors}, status_code=422
            )
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(saved.to_json(), status_code=201 if saved.created else 200)

    @router.delete("/api/{plural}/{doc_id}", status_code=204)
    def delete(plural: str, doc_id: str, etag: str = Query(...)) -> Response:
        guard_write()
        kind = _kind(plural[:-1] if plural.endswith("s") else plural)
        try:
            store.delete(kind, doc_id, etag=etag)
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(status_code=204)

    def _render_preview(body: dict, draw: Callable[[dict, dict], bytes]) -> Response:
        """Validate the body's document, hand it to the host, map its errors."""
        document = body.get("document")
        params = body.get("params") or {}
        if not isinstance(document, dict) or not isinstance(params, dict):
            raise HTTPException(
                status_code=422, detail="body needs document + params mappings"
            )
        report = store.validate("analyzer", document)
        if not report.ok:
            return JSONResponse(
                {"detail": "invalid document", "errors": report.errors}, status_code=422
            )
        try:
            png = draw(report.canonical or document, params)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — an analyzer failure is a 400, never a 500
            raise HTTPException(
                status_code=400, detail=f"preview failed: {exc}"
            ) from exc
        return Response(
            content=png, media_type="image/png", headers={"Cache-Control": "no-store"}
        )

    @router.post("/api/preview")
    def preview_endpoint(body: dict = Body(...)) -> Response:
        if preview is None:
            raise HTTPException(status_code=404, detail="no live preview on this host")
        return _render_preview(body, preview)

    @router.post("/api/preview/summary")
    def summary_preview_endpoint(body: dict = Body(...)) -> Response:
        if summary_preview is None:
            raise HTTPException(
                status_code=404, detail="no summary preview on this host"
            )
        index = body.get("index", 0)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise HTTPException(
                status_code=422, detail="body.index must be a non-negative integer"
            )
        return _render_preview(
            body, lambda document, params: summary_preview(document, params, index)
        )

    return router
