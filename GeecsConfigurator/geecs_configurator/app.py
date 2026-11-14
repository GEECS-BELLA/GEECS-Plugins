"""The app factory: the API over the store, the save route, the home page, the kit at ``/theme``.

With the commit path on (``Settings.commits``) the configs root is the
repository's live checkout: a document is served after a fetch and
fast-forward under the repository's lock (the etag the editor holds is of
the current base; an unreachable origin or a busy lock serves the last
accepted main within seconds and says so in ``X-Configs-Stale``), ``PUT /api/documents/...`` saves
(``save.py``), and a timer reconciles the inbox. Every experiment stays
browsable; a document at a path this box does not own (``ownership.py``) is
served ``read_only`` and its save is the 403. Off, the service is
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
from geecs_schemas.storage import STORAGE_ROOT_PATHS
from pydantic import BaseModel
from geecs_web_theme.web import ForwardedPrefixMiddleware, RevalidatingStaticFiles, make_templates, mount_theme  # fmt: skip

import httpx

from geecs_configurator import inbox, setups
from geecs_configurator.apply import (
    Archiver,
    DryRun,
    Run,
    ANALYSIS_KINDS,
    PortalClient,
    ScannerClient,
    ServiceApplier,
)
from geecs_configurator.kinds import DRY_RUNS, KindInfo, kinds
from geecs_configurator.log import CHIP, STATES, Applier
from geecs_configurator.repository import ConfigsRepository, GitError
from geecs_configurator.save import CompareLinkOnly, DeleteRequest, check_owned, delete, PullRequests, Refused, Saved, SaveRequest, save  # fmt: skip
from geecs_configurator.validation import deletable, referrers, still_named
from geecs_configurator.service import CommitPath
from geecs_configurator.settings import Settings
from geecs_configurator.store import Diff, DocumentStore, Entry, Loaded, NotFound, Report, dump_yaml  # fmt: skip

_PKG = Path(__file__).resolve().parent
_TEMPLATES = make_templates(_PKG / "templates")
#: A posted document: any JSON mapping the form hands back.
Draft = dict[str, Any]  # Any: the draft is validated by its kind's model, not here
T = TypeVar("T")
api = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


def create_app(settings: Settings, pull_requests: PullRequests | None = None, *, http: httpx.Client | None = None, applier: Applier | None = None, consumer_http: httpx.Client | None = None, archiver_run: Run | None = None, portal_http: httpx.Client | None = None) -> FastAPI:  # fmt: skip
    """Build the app over the settings' configs root (the live checkout when the commit path is on).

    ``pull_requests``, ``http`` (the client to GitHub), ``applier``,
    ``consumer_http`` (the client to the scanner), ``portal_http`` (to the
    portal) and ``archiver_run`` (the
    archiver's subprocess) are the seams tests fill with fakes.
    """
    app = FastAPI(title=settings.title, lifespan=_lifespan)
    app.state.settings, app.state.store = settings, DocumentStore(settings.configs_root)
    #: The consumers: the dry runs (always) and, with the commit path, the applies.
    consumers = ServiceApplier(
        settings.configs_root,
        scanners={e: ScannerClient(url, consumer_http) for e, url in settings.scanner_urls().items()},
        archiver=Archiver(settings.archiver_command, run=archiver_run) if settings.archiver_command else None,
        portal=PortalClient(settings.portal_url, portal_http) if settings.portal_url else None,
    )  # fmt: skip
    app.state.consumers = consumers
    commit = CommitPath(settings, pull_requests=pull_requests, http=http, applier=applier or consumers) if settings.commits else None  # fmt: skip
    app.state.commit, app.state.repo = commit, commit.repo if commit else None
    app.state.pull_requests = commit.pull_requests if commit else CompareLinkOnly(settings.github_repo, settings.branch)  # fmt: skip
    app.add_middleware(ForwardedPrefixMiddleware)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Content-Type"])  # fmt: skip
    mount_theme(app)
    app.mount("/static", RevalidatingStaticFiles(directory=str(_PKG / "static")), name="static")  # fmt: skip
    app.include_router(api)
    app.include_router(setups.router)
    app.add_api_route("/", home, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/log", log_page, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/healthz", healthz, methods=["GET"])
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """With the commit path on: set the repository up and run the timer (inbox, fetch) until shutdown."""
    commit: CommitPath | None = app.state.commit
    stop = threading.Event()
    if commit is not None:
        commit.start()  # the repository, through the gate to main; the share caught up
        inbox.prepare(commit.repo, commit.inbox_experiments())
        if (
            app.state.settings.reconcile_seconds > 0
            or app.state.settings.fetch_seconds > 0
            or app.state.settings.apply_seconds > 0
        ):
            threading.Thread(target=commit.run, args=(stop,), name="configurator-timer", daemon=True).start()  # fmt: skip
    try:
        yield
    finally:
        stop.set()
        if commit is not None:
            # inside the unit's TimeoutStopSec: a share left mid-merge (index.lock) blocks every later fast-forward
            commit.wait_share(timeout=10)


def reconcile(app: FastAPI) -> list[inbox.Published]:
    """Publish what waits in the inbox (nothing to do, and no git, when it is empty)."""
    return app.state.commit.reconcile()


#: The serve path's bounds: how long a read waits for the lock, how long its fetch may take, and how
#: recent a fetch makes another unnecessary (seconds). A read never waits on the network for long:
#: past these it serves the last accepted main and says so.
SERVE_WAIT_S, SERVE_FETCH_S, FRESH_S = 5.0, 10.0, 2.0


def _refresh(repo: ConfigsRepository) -> str | None:
    """Under the lock: fetch (bounded, skipped when recent) and fast-forward the live checkout; why it is stale, or ``None``.

    The inbox is not published here: its push takes the full git timeout,
    and a read must not; the timer and the save publish it. The live
    checkout never holds the inbox's files, so a fetch first is harmless.
    """
    try:
        if time.monotonic() - repo.fetched_at > FRESH_S:
            repo.sync(timeout=SERVE_FETCH_S)
        else:  # a recent fetch: no network, but main may be ahead of live (a fetch then a failed push)
            repo.advance()
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
    """Up; whether the configs root is there; the commit path's state (main applied, the share current)."""
    found = request.app.state.settings.configs_root.is_dir()
    out = {"status": "ok", "configs_root": "found" if found else "missing", "commits": "on" if request.app.state.repo else "off"}  # fmt: skip
    commit: CommitPath | None = request.app.state.commit
    if commit is not None:
        out["main"] = commit.main_state()
        out["share"] = "off" if commit.settings.share_clone is None else commit.share_error or "current"  # fmt: skip
        out["owns"] = commit.ownership.describe()
        missing = sorted(set(commit.ownership.experiments) - set(commit.store.experiments()))  # fmt: skip
        if missing:  # a typo, or a case mismatch: owned names match folders exactly
            out["owns"] += f" (no folder in the configs repo: {', '.join(missing)})"
        out["credential"] = (
            "GitHub App" if commit.credential else "git's own (deploy key)"
        )
    return out


@api.get("/log")
def log_rows(
    request: Request, limit: int = 200
) -> list[dict[str, Any]]:  # Any: the rows as JSON
    """The commit log, newest first (empty when the commit path is off)."""
    commit: CommitPath | None = request.app.state.commit
    return [r.model_dump() for r in commit.log.rows(limit)] if commit else []


def log_page(request: Request) -> HTMLResponse:
    """The commit log: what was committed, by whom, and whether it took effect (the four states)."""
    commit: CommitPath | None = request.app.state.commit
    return _TEMPLATES.TemplateResponse(request, "log.html", {
        "settings": request.app.state.settings, "rows": commit.log.rows() if commit else [], "commit": commit,
        "main": commit.main_state() if commit else "applied",
        "states": STATES, "chip": CHIP,
    })  # fmt: skip


@api.get("/kinds")
def list_kinds() -> list[KindInfo]:
    """Every registered kind: storage, tier, path or reason, readers, dry run."""
    return kinds()


@api.get("/experiments")
def experiments(request: Request) -> list[str]:
    """The experiments in the configs root."""
    return _store(request).experiments()


@api.get("/schema/{kind}")
def schema(request: Request, kind: str) -> JSONResponse:
    """The kind's JSON Schema, the document's and the stricter fields' tiers in it.

    An analysis kind's form is its consumer's: the portal's schema, where a
    recipe's steps and measure are the analysis core's registry (the step
    list, each step's parameters typed). Without a portal, or when it does
    not answer, the document's own schema is served and
    ``X-Schema-Source`` says why (a step is then typed as a name and its
    parameters as written).
    """
    if kind not in SCHEMA_REGISTRY:
        raise HTTPException(404, f"no kind {kind!r}")
    portal: PortalClient | None = request.app.state.consumers.portal
    if kind in ANALYSIS_KINDS:
        if portal is None:
            source = (
                "geecs-schemas (no portal configured: GEECS_CONFIGURATOR_PORTAL_URL)"
            )
        else:
            try:
                return JSONResponse(portal.schema(kind), headers={"X-Schema-Source": "portal"})  # fmt: skip
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("the portal's %s schema did not answer: %s", kind, exc)
                source = f"geecs-schemas (the portal did not answer: {' '.join(str(exc).split())[:200]})"  # fmt: skip
        return JSONResponse(SCHEMA_REGISTRY[kind].model_json_schema(), headers={"X-Schema-Source": source.encode("ascii", "replace").decode()})  # fmt: skip
    return JSONResponse(SCHEMA_REGISTRY[kind].model_json_schema())


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
            stale = _refresh(repo) if locked else "the repository is busy (a push or a fetch); serving the last accepted main"  # fmt: skip
        loaded = _answer(lambda: _store(request).read(_file_kind(kind), experiment, doc_id))  # fmt: skip
        commit: CommitPath | None = request.app.state.commit
        if commit is not None and not commit.ownership.owns_file(_store(request).root, loaded.path):  # fmt: skip
            loaded = loaded.model_copy(
                update={"read_only": True}
            )  # another box's: browsable, never saved here
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
    if request.app.state.commit.settling.is_set():
        # an apply pass holds the repository (a reopen takes a minute or more): say so now, never keep the save waiting
        return _busy()
    try:  # another box's document: refused before anything runs, the dry run included
        _answer(lambda: check_owned(_store(request), request.app.state.commit.ownership, _file_kind(kind), experiment, doc_id, create=body.create))  # fmt: skip
    except Refused as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)
    # The consumer's own check of the draft first (kinds.DRY_RUNS): a refusal is the save's refusal.
    check: DryRun = request.app.state.consumers.dry_run(
        kind, experiment, body.document, doc_id
    )
    if check.refusal is not None:
        return JSONResponse({"detail": {"reason": f"the {check.by}'s dry run refused the draft: {check.refusal}", "dry_run": check.model_dump()}}, status_code=422)  # fmt: skip
    if request.app.state.commit.settling.is_set():  # a pass began during the dry run
        return _busy()
    _reconcile_quietly(request.app)
    try:
        saved = _answer(lambda: save(_store(request), repo, request.app.state.pull_requests, _file_kind(kind), experiment, doc_id, body, email=request.app.state.settings.committer_email, ownership=request.app.state.commit.ownership))  # fmt: skip
    except Refused as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)
    if (
        kind in DRY_RUNS and not check.ran and check.by != "none"
    ):  # asked, and it could not answer
        saved.notes.append(f"the dry run did not run: {check.detail}")
    elif check.questions:
        saved.notes.append("the dry run asked: " + "; ".join(q.get("title", "") for q in check.questions))  # fmt: skip
    request.app.state.commit.saved(kind, saved)
    if saved.outcome != "nothing to do":
        inbox.reset_backoff(
            repo
        )  # origin answered: a waiting measurement need not wait out its backoff
    return _served(saved)


class DeleteCheck(BaseModel):
    """What a delete would do, asked before it: the file, whether its kind is deleted from here, and who names it."""

    path: str
    deletable: bool
    referrers: list[str]
    reason: str = ""


@api.post("/delete-check/{kind}/{doc_id:path}", response_model=DeleteCheck)
def delete_check(request: Request, kind: str, doc_id: str, experiment: str | None = None) -> DeleteCheck:  # fmt: skip
    """The delete's dry run: the file it removes and every document that names it (any one refuses the delete)."""
    store, repo = _store(request), request.app.state.repo
    if repo is not None and repo.lock.acquire(
        timeout=SERVE_WAIT_S
    ):  # the referrers as main has them now
        try:
            _refresh(repo)
        finally:
            repo.lock.release()
    loaded = _answer(lambda: store.read(_file_kind(kind), experiment, doc_id))
    if (
        not loaded.etag
    ):  # the save's own rule: what did not read or parse is not saved, or deleted, over
        return DeleteCheck(path=loaded.path, deletable=False, referrers=[], reason=f"{loaded.path} did not read or parse, so the editor never showed it; fix or remove it by hand (a pull request)")  # fmt: skip
    if not deletable(kind):
        return DeleteCheck(path=loaded.path, deletable=False, referrers=[], reason=f"a {kind} is one file its readers expect; it is not deleted from the editor")  # fmt: skip
    names = referrers(store, kind, experiment, Path(doc_id).stem, besides=loaded.path)  # fmt: skip
    reason = still_named(loaded.path, names) if names else ""
    return DeleteCheck(path=loaded.path, deletable=not names, referrers=names, reason=reason)  # fmt: skip


@api.delete("/documents/{kind}/{doc_id:path}", response_model=Saved)
def delete_document(request: Request, kind: str, doc_id: str, body: DeleteRequest, experiment: str | None = None) -> Response:  # fmt: skip
    """Delete a document as it was served (its etag): a commit or a pull request as its tier says, the operator the git author."""
    repo = request.app.state.repo
    if repo is None:
        raise HTTPException(405, "read-only: the commit path is not configured")
    if request.app.state.commit.settling.is_set():
        return _busy()
    try:
        _answer(lambda: check_owned(_store(request), request.app.state.commit.ownership, _file_kind(kind), experiment, doc_id, create=False))  # fmt: skip
        saved = _answer(lambda: delete(_store(request), repo, request.app.state.pull_requests, kind, experiment, doc_id, body, email=request.app.state.settings.committer_email, ownership=request.app.state.commit.ownership))  # fmt: skip
    except Refused as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)
    request.app.state.commit.saved(kind, saved)
    return _served(saved)


def _busy() -> JSONResponse:
    """A save while an apply pass holds the repository: answered at once (nothing saved), never kept waiting."""
    return JSONResponse({"detail": "the configurator is applying a committed change (the worker reopens); nothing was saved: save again in a minute"}, status_code=503, headers={"Retry-After": "60"})  # fmt: skip


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


@api.post("/dry-run/{kind}", response_model=DryRun)
def dry_run(request: Request, kind: str, experiment: str | None = None, id: str | None = None, document: Draft = Body(embed=True)) -> DryRun:  # fmt: skip
    """The kind's consumer check of an unsaved draft (``ran: false`` with the reason when there is none here); ``id`` names the document."""
    if kind not in SCHEMA_REGISTRY:
        raise HTTPException(404, f"no kind {kind!r}")
    return request.app.state.consumers.dry_run(kind, experiment, document, id)


@api.get("/readback")
def readback(
    request: Request, variable: str, experiment: str | None = None
) -> JSONResponse:
    """One live readback of ``Device:Variable``, through the experiment's scanner (the form's current value)."""
    settings: Settings = request.app.state.settings
    scanner: ScannerClient | None = request.app.state.consumers.scanners.get(
        experiment or settings.experiment
    )
    if scanner is None:
        raise HTTPException(
            503, "no scanner configured (GEECS_CONFIGURATOR_SCANNER_URL)"
        )
    try:
        status, body = scanner.readback(variable)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, f"the scanner did not answer: {exc}") from exc
    return JSONResponse(body, status_code=status)


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
        "commits": request.app.state.repo is not None, "scanner": experiment in request.app.state.consumers.scanners, "applies": setups.applies_for(settings, experiment),
        "not_owned": _not_owned(request.app.state.commit, catalogue, kind, experiment, store.folder(kind, experiment) / id if id in {d.id for d in docs} else None),
    })  # fmt: skip


def _not_owned(commit: CommitPath | None, catalogue: list[KindInfo], kind: str, experiment: str, opened: Path | None = None) -> str:  # fmt: skip
    """The page's word when what it shows is not this box's to save (empty when it is, or the commit path is off).

    In the shared analysis tree ownership is per facility folder: the word
    stands over the listing, and over an open document only when that
    document is another facility's.
    """
    if commit is None:
        return ""
    own = commit.ownership
    path = next((k.path for k in catalogue if k.kind == kind), None) or ""
    if (
        kind and "<experiment>" not in path
    ):  # the shared analysis tree: owned per facility folder
        if not path.startswith(STORAGE_ROOT_PATHS["analysis"]):
            return ""
        if opened is not None and own.owns_file(commit.store.root, opened.relative_to(commit.store.root).as_posix()):  # fmt: skip
            return ""
        return f"This box saves only in its own folders of the shared analysis tree ({', '.join(own.analysis_folders) or 'none'}); other documents here are read-only."  # fmt: skip
    if experiment and experiment not in own.experiments:
        return f"{experiment} is not this services box's: browsable here, saved from the box that owns it or by pull request. This box owns {own.describe()}."  # fmt: skip
    return ""
