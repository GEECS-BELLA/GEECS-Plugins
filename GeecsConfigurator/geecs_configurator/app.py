"""The app factory: the API over the store, the save route, the home page, the kit at ``/theme``.

With the commit path on (``Settings.commits``) the configs root is the
repository's live checkout: a document is served after a fetch and
fast-forward under the repository's lock (the etag the editor holds is of
the current base; an unreachable origin serves the last accepted main and
says so in ``X-Configs-Stale``), ``PUT /api/documents/...`` saves
(``save.py``), and a timer reconciles the inbox. Off, the service is
read-only and the save route answers 405. The API answers CORS for the
surfaces that embed the editor (a site value), because the embedded editor
runs in the host page's origin.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar

from fastapi import APIRouter, Body, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from geecs_schemas import KIND_STORAGE, SCHEMA_REGISTRY, NotAFile
from pydantic import BaseModel
from geecs_web_theme.web import ForwardedPrefixMiddleware, RevalidatingStaticFiles, make_templates, mount_theme  # fmt: skip

from geecs_configurator import inbox
from geecs_configurator.kinds import KindInfo, kinds
from geecs_configurator.repository import ConfigsRepository, GitError, Identity
from geecs_configurator.save import (
    CompareLinkOnly,
    PullRequests,
    Refused,
    Saved,
    SaveRequest,
    save,
)
from geecs_configurator.settings import Settings
from geecs_configurator.store import Diff, DocumentStore, Entry, Loaded, NotFound, Report, dump_yaml  # fmt: skip

_PKG = Path(__file__).resolve().parent
_TEMPLATES = make_templates(_PKG / "templates")
#: A posted document: any JSON mapping the form hands back.
Draft = dict[str, Any]  # Any: the draft is validated by its kind's model, not here
T = TypeVar("T")
api = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


def create_app(settings: Settings, pull_requests: PullRequests | None = None) -> FastAPI:  # fmt: skip
    """Build the app over the settings' configs root (the live checkout when the commit path is on)."""
    app = FastAPI(title=settings.title, lifespan=_lifespan)
    app.state.settings, app.state.store = settings, DocumentStore(settings.configs_root)
    app.state.repo = (
        ConfigsRepository(
            settings.state_dir,
            settings.origin,
            settings.branch,
            Identity(settings.committer_name, settings.committer_email),
        )  # fmt: skip
        if settings.commits and settings.state_dir is not None
        else None
    )
    app.state.pull_requests = pull_requests or CompareLinkOnly(settings.github_repo, settings.branch)  # fmt: skip
    app.add_middleware(ForwardedPrefixMiddleware)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST", "PUT"], allow_headers=["Content-Type"])  # fmt: skip
    mount_theme(app)
    app.mount("/static", RevalidatingStaticFiles(directory=str(_PKG / "static")), name="static")  # fmt: skip
    app.include_router(api)
    app.add_api_route("/", home, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/healthz", healthz, methods=["GET"])
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """With the commit path on: set the repository up and run the inbox timer until shutdown."""
    repo: ConfigsRepository | None = app.state.repo
    stop = threading.Event()
    if repo is not None:
        repo.ensure()
        inbox.prepare(repo, app.state.store.experiments())
        every = app.state.settings.reconcile_seconds
        if every > 0:
            threading.Thread(target=_timer, args=(app, stop, every), name="configurator-inbox", daemon=True).start()  # fmt: skip
    try:
        yield
    finally:
        stop.set()


def _timer(app: FastAPI, stop: threading.Event, every: float) -> None:
    while not stop.wait(every):
        try:
            reconcile(app)
        except Exception:  # the timer outlives any one failure; the next tick retries
            logger.exception("the inbox reconcile failed")


def reconcile(app: FastAPI) -> list[inbox.Published]:
    """Publish what waits in the inbox (nothing to do, and no git, when it is empty)."""
    repo: ConfigsRepository = app.state.repo
    experiments = app.state.store.experiments()
    if not inbox.pending(repo, experiments):
        return []
    author = Identity(
        app.state.settings.inbox_author, app.state.settings.committer_email
    )
    return inbox.reconcile(repo, experiments, author)


#: The serve path's bounds: how long a read waits for the lock, how long its fetch may take, and how
#: recent a fetch makes another unnecessary (seconds). A read never waits on the network for long:
#: past these it serves the last accepted main and says so.
SERVE_WAIT_S, SERVE_FETCH_S, FRESH_S = 5.0, 10.0, 2.0


def _refresh(app: FastAPI, repo: ConfigsRepository) -> str | None:
    """Under the lock: reconcile the inbox, then fetch and fast-forward the live checkout; why it is stale, or ``None``."""
    try:
        reconcile(app)
        if time.monotonic() - repo.fetched_at > FRESH_S:
            repo.sync(timeout=SERVE_FETCH_S)
    except (GitError, OSError) as exc:
        logger.warning("serving the last accepted main: %s", exc)
        return " ".join(str(exc).split())[:200]
    return None


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
    return {"status": "ok", "configs_root": "found" if found else "missing", "commits": "on" if request.app.state.repo else "off"}  # fmt: skip


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


def _served(model: BaseModel, status: int = 200, headers: dict[str, str] | None = None) -> Response:  # fmt: skip
    """A model as JSON with ``.inf``/``.nan`` as strings (FastAPI's own encoding would serve ``null``)."""
    return Response(model.model_dump_json(), status_code=status, media_type="application/json", headers=headers)  # fmt: skip


@api.get("/documents/{kind}/{doc_id:path}", response_model=Loaded)
def document(request: Request, kind: str, doc_id: str, experiment: str | None = None) -> Response:  # fmt: skip
    """One document as it is on main (fetched first when the commit path is on), with its etag, errors and tier."""
    repo: ConfigsRepository | None = request.app.state.repo
    stale, locked = None, repo is not None and repo.lock.acquire(timeout=SERVE_WAIT_S)
    try:
        if repo is not None:
            # busy (a long push or fetch): read the live checkout as it stands, never wait on the network
            stale = _refresh(request.app, repo) if locked else "the repository is busy (a push or a fetch); serving the last accepted main"  # fmt: skip
        loaded = _answer(lambda: _store(request).read(_file_kind(kind), experiment, doc_id))  # fmt: skip
    finally:
        if locked and repo is not None:
            repo.lock.release()
    return _served(loaded, headers={"X-Configs-Stale": stale} if stale else None)


@api.put("/documents/{kind}/{doc_id:path}", response_model=Saved)
def save_document(request: Request, kind: str, doc_id: str, body: SaveRequest, experiment: str | None = None) -> Response:  # fmt: skip
    """Save a whole document over the version it was served (its etag), as its tier says."""
    repo = request.app.state.repo
    if repo is None:
        raise HTTPException(405, "read-only: the commit path is not configured")
    _reconcile_quietly(request.app)
    try:
        saved = _answer(lambda: save(_store(request), repo, request.app.state.pull_requests, _file_kind(kind), experiment, doc_id, body, email=request.app.state.settings.committer_email))  # fmt: skip
    except Refused as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)
    return _served(saved)


def _reconcile_quietly(app: FastAPI) -> None:
    """The inbox goes first, before the save's fetch; its failure is the inbox's, not the save's."""
    try:
        reconcile(app)
    except (GitError, OSError) as exc:
        logger.warning("the inbox reconcile before a save failed: %s", exc)


@api.post("/validate/{kind}")
def validate(request: Request, kind: str, document: Draft = Body(embed=True), id: str | None = Body(None, embed=True)) -> Report:  # fmt: skip
    """Validate a draft without writing it; ``id`` adds the setup's name-is-its-stem rule."""
    return _answer(lambda: _store(request).validate(kind, document, stem=Path(id).stem if id else None))  # fmt: skip


@api.post("/diff/{kind}/{doc_id:path}", response_model=Diff)
def diff(request: Request, kind: str, doc_id: str, experiment: str | None = None, document: Draft = Body(embed=True)) -> Response:  # fmt: skip
    """What a draft changes in the stored document, each change's tier, and the strictest."""
    return _served(_answer(lambda: _store(request).diff(_file_kind(kind), experiment, doc_id, document)))  # fmt: skip


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
