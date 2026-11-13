"""The app factory: the read-only API over the store, the home page, the kit at ``/theme``.

Read only in this slice: no route writes, and none exists for saving. The
API answers CORS for the surfaces that embed the editor (a site value),
because the embedded editor runs in the host page's origin.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from fastapi import APIRouter, Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from geecs_schemas import KIND_STORAGE, SCHEMA_REGISTRY, NotAFile
from geecs_web_theme.web import ForwardedPrefixMiddleware, RevalidatingStaticFiles, make_templates, mount_theme  # fmt: skip

from geecs_configurator.kinds import KindInfo, kinds
from geecs_configurator.settings import Settings
from geecs_configurator.store import Diff, DocumentStore, Entry, Loaded, NotFound, Report, dump_yaml  # fmt: skip

_PKG = Path(__file__).resolve().parent
_TEMPLATES = make_templates(_PKG / "templates")
#: A posted document: any JSON mapping the form hands back.
Draft = dict[str, Any]  # Any: the draft is validated by its kind's model, not here
T = TypeVar("T")
api = APIRouter(prefix="/api")


def create_app(settings: Settings) -> FastAPI:
    """Build the app over the settings' configs root."""
    app = FastAPI(title=settings.title)
    app.state.settings, app.state.store = settings, DocumentStore(settings.configs_root)
    app.add_middleware(ForwardedPrefixMiddleware)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"], allow_headers=["Content-Type"])  # fmt: skip
    mount_theme(app)
    app.mount("/static", RevalidatingStaticFiles(directory=str(_PKG / "static")), name="static")  # fmt: skip
    app.include_router(api)
    app.add_api_route("/", home, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/healthz", healthz, methods=["GET"])
    return app


def _store(request: Request) -> DocumentStore:
    return request.app.state.store


def _file_kind(kind: str) -> str:
    """A kind with documents to list: a file kind, or the family files (package data)."""
    storage = KIND_STORAGE.get(kind)
    if storage is None:
        raise HTTPException(404, f"no kind {kind!r}")
    if isinstance(storage, NotAFile) and kind != "family_file":
        raise HTTPException(404, f"{kind} is not a file: {storage.reason}")
    return kind


def _answer(call: Callable[[], T]) -> T:
    try:
        return call()
    except NotFound as exc:
        raise HTTPException(404, str(exc)) from exc


def healthz(request: Request) -> dict[str, str]:
    """Up, and whether the configs root is there."""
    found = request.app.state.settings.configs_root.is_dir()
    return {"status": "ok", "configs_root": "found" if found else "missing"}


@api.get("/kinds")
def list_kinds() -> list[KindInfo]:
    """Every registered kind: storage, tier, path or reason, readers, dry run."""
    return kinds()


@api.get("/experiments")
def experiments(request: Request) -> list[str]:
    """The experiments in the configs root."""
    return _store(request).experiments()


@api.get("/schema/{kind}")
def schema(kind: str) -> dict[str, Any]:  # Any: a JSON Schema
    """The kind's JSON Schema, the document's and the stricter fields' tiers in it."""
    if kind not in SCHEMA_REGISTRY:
        raise HTTPException(404, f"no kind {kind!r}")
    return SCHEMA_REGISTRY[kind].model_json_schema()


@api.get("/documents/{kind}")
def documents(
    request: Request, kind: str, experiment: str | None = None
) -> list[Entry]:
    """The documents of a kind, valid or not."""
    return _answer(lambda: _store(request).list(_file_kind(kind), experiment))


@api.get("/documents/{kind}/{doc_id:path}")
def document(request: Request, kind: str, doc_id: str, experiment: str | None = None) -> Loaded:  # fmt: skip
    """One document as it is on disk, with its etag, errors and tier."""
    return _answer(lambda: _store(request).read(_file_kind(kind), experiment, doc_id))


@api.post("/validate/{kind}")
def validate(request: Request, kind: str, document: Draft = Body(embed=True), id: str | None = Body(None, embed=True)) -> Report:  # fmt: skip
    """Validate a draft without writing it; ``id`` adds the setup's name-is-its-stem rule."""
    return _answer(lambda: _store(request).validate(kind, document, stem=Path(id).stem if id else None))  # fmt: skip


@api.post("/diff/{kind}/{doc_id:path}")
def diff(request: Request, kind: str, doc_id: str, experiment: str | None = None, document: Draft = Body(embed=True)) -> Diff:  # fmt: skip
    """What a draft changes in the stored document, each change's tier, and the strictest."""
    return _answer(lambda: _store(request).diff(_file_kind(kind), experiment, doc_id, document))  # fmt: skip


@api.post("/yaml", response_class=PlainTextResponse)
def render_yaml(document: Draft = Body(embed=True)) -> str:
    """A draft as YAML (field order kept)."""
    return dump_yaml(document)


def home(request: Request, experiment: str = "", kind: str = "", id: str = "") -> HTMLResponse:  # fmt: skip
    """The catalogue, a kind's documents, and the open document (selection in the URL)."""
    settings, store = request.app.state.settings, _store(request)
    known = store.experiments()
    if experiment not in known:
        experiment = settings.experiment if settings.experiment in known else (known[0] if known else "")  # fmt: skip
    catalogue = kinds()
    browsable = {k.kind for k in catalogue if k.storage != "not-a-file"}
    kind = kind if kind in browsable else ""
    try:
        docs = store.list(kind, experiment) if kind else []
    except NotFound:
        docs = []
    return _TEMPLATES.TemplateResponse(request, "home.html", {
        "settings": settings, "experiments": known, "experiment": experiment, "catalogue": catalogue,
        "kind": kind, "docs": docs, "doc_id": id if id in {d.id for d in docs} else "",
    })  # fmt: skip
