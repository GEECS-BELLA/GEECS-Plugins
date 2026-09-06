"""FastAPI surface of the config editor: JSON API + the editor page + its two static assets.

Everything is relative to the router's mount, so it works at ``/`` (the
standalone host) and under ``/configs`` behind the portal's reverse-proxy
prefix alike: the page computes its API base from its own URL, and no
absolute portal URL is written here.

API (under the mount)::

    GET  /                          the editor page (standalone chrome)
    GET  /static/editor.js|.css     the editor assets (the portal includes them too)
    GET  /api/list                  analyzers + groups (validity, summary), namespaces, git pending
    GET  /api/schema/{kind}         JSON Schema for kind = analyzer | group
    GET  /api/{kind}s/{id}          one document: raw, etag, yaml, validity
    POST /api/validate/{kind}       {document} -> {ok, errors, canonical, yaml}
    PUT  /api/{kind}s/{ns}/{id}     {document, etag|null} -> saved {etag, yaml, created}; 409 / 422
    DELETE /api/{kind}s/{id}?etag=  -> 204; 409
    POST /api/preview               {document, params} -> image/png (host-provided; 404 without)

Errors are JSON ``{"detail": ...}`` with honest statuses: 404 unknown,
409 conflict (stale etag / duplicate id), 422 invalid document (with the
pydantic locations), 400 for a preview the analyzer refused — never a 500
for a config problem.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from fastapi import APIRouter, Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from scan_analysis.config_store import (
    ConfigStore,
    ConflictError,
    DocumentInvalid,
    NotFound,
)

logger = logging.getLogger(__name__)

__all__ = ["PreviewFn", "create_editor_app", "create_editor_router", "main"]

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"
_TEMPLATES = Jinja2Templates(directory=str(_HERE / "templates"))

#: ``(document, params) -> PNG bytes``: render the (validated) document under
#: edit on pixels the host owns.  Raise ``ValueError`` for a refusal the
#: user can act on (400) and ``LookupError`` for a missing shot (404).
PreviewFn = Callable[[Mapping[str, Any], Mapping[str, Any]], bytes]

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
    read_only: bool = False,
) -> APIRouter:
    """Build the editor router over *store*.

    Parameters
    ----------
    store : ConfigStore
        The configs tree.
    preview : callable, optional
        Host-provided live preview (the portal renders the current shot);
        without it ``POST /api/preview`` is 404 and the page hides the pane.
    read_only : bool, default False
        Serve the browser and validation but refuse writes (405).
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
            {"preview": preview is not None, "read_only": read_only},
        )

    @router.get("/static/{name}")
    def static(name: str) -> FileResponse:
        if name not in ("editor.js", "editor.css"):
            raise HTTPException(status_code=404)
        return FileResponse(_STATIC / name)

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

    @router.post("/api/preview")
    def preview_endpoint(body: dict = Body(...)) -> Response:
        if preview is None:
            raise HTTPException(status_code=404, detail="no live preview on this host")
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
            png = preview(report.canonical or document, params)
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

    return router


def create_editor_app(root: Path, *, read_only: bool = False) -> FastAPI:
    """The standalone host: the editor at ``/`` over one configs tree, no preview."""
    app = FastAPI(title="GEECS analysis config editor", docs_url=None, redoc_url=None)
    app.include_router(create_editor_router(ConfigStore(root), read_only=read_only))
    return app


def _default_root() -> Optional[Path]:
    try:
        from geecs_data_utils import ScanPaths

        root = ScanPaths.paths_config.scan_analysis_configs_path
    except Exception:  # noqa: BLE001 — no config.ini is a normal standalone case
        return None
    return Path(root) if root else None


def main(argv: Optional[list[str]] = None) -> int:
    """``scan-config-editor``: serve the editor standalone."""
    parser = argparse.ArgumentParser(
        description="Serve the GEECS analysis config editor over a scan_analysis_configs tree."
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=None,
        help="the scan_analysis_configs root (default: config.ini scan_analysis_configs_path)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8210)
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.configs or _default_root()
    if root is None or not (root / "analyzers").is_dir():
        parser.error(
            "--configs must name a tree with an analyzers/ folder (or set config.ini)"
        )
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    logger.info("config editor over %s at http://%s:%d/", root, args.host, args.port)
    uvicorn.run(
        create_editor_app(root, read_only=args.read_only),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
