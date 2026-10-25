"""The FastAPI glue every GEECS web surface needs, written once.

Three surfaces (the Data Portal, the logbook, the scanner) each carried a
copy of the same three things: the ``X-Forwarded-Prefix`` middleware that
lets a page live behind a reverse proxy, the ``/theme`` mount that serves
this package's stylesheets, and a template environment that puts ``root``
in every context. The copies were verbatim by intent and drifted anyway
(the portal computed ``root`` in Python, the logbook in Jinja, the scanner
in a context processor). This module is the one copy.

It needs FastAPI and Jinja2, so it lives behind the ``web`` extra::

    geecs-web-theme = { path = "../GeecsWebTheme", extras = ["web"] }

Importing :mod:`geecs_web_theme` itself still needs nothing — the theme is a
static directory first, and this module is opt-in glue for hosts that are
FastAPI apps.

Usage::

    from geecs_web_theme.web import ForwardedPrefixMiddleware, make_templates, mount_theme

    app = FastAPI(root_path=static_prefix)
    app.add_middleware(ForwardedPrefixMiddleware)
    mount_theme(app)                                   # /theme/…
    templates = make_templates(TEMPLATES_DIR)          # {{ root }} in every context

and in a template, ``{{ root }}/theme/theme-boot.js`` — every link, form,
fetch base and asset URL goes through ``root`` so a mount prefix carries.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Union

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

from geecs_web_theme import static_dir

__all__ = [
    "FORWARDED_PREFIX_HEADER",
    "ForwardedPrefixMiddleware",
    "RevalidatingStaticFiles",
    "clean_prefix",
    "make_templates",
    "mount_theme",
    "operation_id",
    "root_of",
]

#: Requests carrying a proxy mount prefix — the Grafana/JupyterHub
#: convention every reverse proxy speaks.
FORWARDED_PREFIX_HEADER = b"x-forwarded-prefix"

#: A valid mount prefix: non-empty ``/segment`` parts of RFC-3986-ish path
#: characters — no ``//``, no query/fragment/quote characters, no
#: whitespace or backslashes.
_PREFIX_RE = re.compile(r"(?:/[A-Za-z0-9._~%@+-]+)+")


def clean_prefix(raw: str) -> str:
    """Normalize a mount prefix: ``/portal/`` → ``/portal``; anything bad → ``""``.

    Accepts only what :data:`_PREFIX_RE` matches — anything else is
    treated as no prefix rather than propagated into every link on the
    page. A bare ``/`` means "mounted at root", i.e. no prefix.

    Parameters
    ----------
    raw : str
        The header value as the proxy sent it.

    Returns
    -------
    str
        The prefix without a trailing slash, or ``""``.
    """
    prefix = raw.strip().rstrip("/")
    if not prefix or _PREFIX_RE.fullmatch(prefix) is None:
        return ""
    return prefix


class ForwardedPrefixMiddleware:
    """Adopt the proxy's ``X-Forwarded-Prefix`` as the ASGI ``root_path``.

    Behind ``proxy /portal → portal:8200`` the app itself never sees the
    mount point; the proxy names it in this header. Setting
    ``scope["root_path"]`` makes every link, form, redirect and JS fetch
    (all built through the one ``root`` context value) carry the prefix,
    so the surface works at root and under any mount point. The header,
    when present, wins over a static ``--root-path`` (the proxy is
    authoritative for where it mounted us); a client faking it only
    rewrites its own page's links.

    **Two proxy shapes, one setting each — not interchangeable.** A
    *prefix-stripping* proxy (nginx ``proxy_pass …/``, Caddy
    ``handle_path``) forwards the bare path and must send this header. A
    *prefix-preserving* proxy (forwards ``/portal/run/…`` as is, sends no
    header) is what ``FastAPI(root_path=…)`` / ``--root-path`` is for:
    the app then expects the prefixed upstream path — Starlette's plain
    routes still answer unprefixed, but a ``Mount`` (``/static``,
    ``/theme``) resolves files against its own prefixed ``root_path`` and
    answers the prefixed path **only**. So ``--root-path`` is not a
    fallback for a stripping proxy that omits the header: that pairing
    serves the HTML and loses every stylesheet and script (a styleless
    page, not a 404). Verified on the logbook and the portal, 2026-09-13.

    The path is re-prefixed too (the ASGI-canonical shape: ``path``
    includes ``root_path``). Starlette's router strips ``root_path`` from
    the FRONT of ``path`` wherever it happens to match, so the
    proxy-stripped path alone would double-strip under a mount named like
    a route head (``/run``, ``/api``, …), 404ing that whole route family —
    and its trailing-slash redirects build the Location from ``path``,
    which would drop the prefix. Re-prefixing makes the strip exact and
    the redirects complete.
    """

    def __init__(self, app: Callable[..., Any]) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        """Rewrite ``root_path`` and ``path`` from the header, then pass through."""
        if scope["type"] == "http":
            for name, value in scope.get("headers", []):
                if name == FORWARDED_PREFIX_HEADER:
                    prefix = clean_prefix(value.decode("latin-1"))
                    if prefix:
                        scope["root_path"] = prefix
                        scope["path"] = prefix + scope["path"]
                    break
        await self.app(scope, receive, send)


def operation_id(route: APIRoute) -> str:
    """A route's OpenAPI ``operationId``, from its method and path alone.

    FastAPI's default names an operation after the route *function*
    (``status_api_status_get``), so renaming a Python function renames the
    operation in every client generated from the document. This one is the
    method and the path's words — ``GET /api/run/{uid}/analysis`` is
    ``get_api_run_uid_analysis`` — which changes only when the route does.
    A multi-method route gets one id for all its operations (FastAPI's
    default does the same, from whichever method a set yields first; this
    takes the alphabetically first, so the id is the same in every
    process) — register one route per method instead. Pass it as
    ``FastAPI(generate_unique_id_function=operation_id)``; it also names
    FastAPI's generated request-body schemas (``Body_<id>``).

    Parameters
    ----------
    route : APIRoute
        The route being registered.

    Returns
    -------
    str
        ``<method>_<path words>``, or ``<method>_root`` for ``/``.
    """
    method = sorted(route.methods)[0].lower()
    words = re.findall(r"[A-Za-z0-9]+", route.path_format)
    return "_".join([method, *(words or ["root"])])


def root_of(request: Request) -> str:
    """Return the request's URL prefix (``""`` at root) — prepend it to every path."""
    return request.scope.get("root_path", "").rstrip("/")


class RevalidatingStaticFiles(StaticFiles):
    """``StaticFiles`` whose answers say ``Cache-Control: no-cache``.

    Without a freshness header a browser guesses one from ``Last-Modified``
    and keeps serving its old copy without asking — so after a deploy a page
    ran the new script against the old stylesheet (the logbook's lookup
    panel, unstyled, until a hard reload). ``no-cache`` means *revalidate
    every time*, not *never store*: the ETag starlette already sends makes
    an unchanged file a 304 with no body. Every surface's ``/static`` and
    the ``/theme`` mount use this.
    """

    # Any: passed through untouched to starlette's own file_response.
    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        """Starlette's file response, with the revalidate header."""
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def mount_theme(app: FastAPI, path: str = "/theme", *, name: str = "theme") -> None:
    """Serve this package's static directory from *app* at *path*.

    Every surface mounts the theme itself, so its pages reach
    ``{{ root }}/theme/…`` on their own origin — no cross-origin fetch, and
    the kit's reference page appears at ``<path>/kit.html`` for free.

    Parameters
    ----------
    app : FastAPI
        The host application.
    path : str, default "/theme"
        Where to mount. Templates assume ``/theme``; change both or neither.
    name : str, default "theme"
        The mount's route name, for ``url_for``.
    """
    app.mount(path, RevalidatingStaticFiles(directory=str(static_dir())), name=name)


def make_templates(
    directory: Union[Path, str],
    *,
    globals: Optional[Mapping[str, Any]] = None,
    filters: Optional[Mapping[str, Callable[..., Any]]] = None,
    context_processors: Iterable[Callable[[Request], dict[str, Any]]] = (),
) -> Jinja2Templates:
    """Build a template environment with ``root`` in every context.

    ``root`` is :func:`root_of` for the rendering request, so a template
    writes ``{{ root }}/theme/kit.css`` and ``{{ root }}/api/…`` and the
    page works at root and under any proxy mount. Pass a surface's own
    globals, filters and further context processors through; they are
    installed on the environment in that order.

    Parameters
    ----------
    directory : Path or str
        The template directory.
    globals : mapping, optional
        Names installed as Jinja globals (a vocabulary such as ``STATES``).
    filters : mapping, optional
        Jinja filters to install.
    context_processors : iterable of callables, optional
        Further ``request -> dict`` processors, run after the ``root`` one.

    Returns
    -------
    Jinja2Templates
        Ready for ``templates.TemplateResponse(request, name, context)``.
    """
    templates = Jinja2Templates(
        directory=str(directory),
        context_processors=[
            lambda request: {"root": root_of(request)},
            *context_processors,
        ],
    )
    if globals:
        templates.env.globals.update(globals)
    if filters:
        templates.env.filters.update(filters)
    return templates
