"""The FastAPI glue behaves like the portal's copy it replaces.

The cases are the portal's own prefix tests, re-homed: they were written
against a shipped incident each (a mount named like a route head 404ing a
whole route family; a trailing-slash redirect dropping the prefix; a
malformed header propagated into every link).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import HTMLResponse, PlainTextResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from geecs_web_theme import STATES  # noqa: E402
from geecs_web_theme.web import (  # noqa: E402
    ForwardedPrefixMiddleware,
    clean_prefix,
    make_templates,
    mount_theme,
    root_of,
)


def _app(tmp_path: Path) -> FastAPI:
    (tmp_path / "page.html").write_text(
        '<a href="{{ root }}/run/1">{{ root }}|{{ word }}|{{ kit_state["ok"] }}</a>'
    )
    app = FastAPI()
    app.add_middleware(ForwardedPrefixMiddleware)
    mount_theme(app)
    templates = make_templates(
        tmp_path,
        globals={"kit_state": STATES},
        filters={"shout": str.upper},
        context_processors=[lambda request: {"word": "extra"}],
    )

    @app.get("/")
    def index(request: Request) -> PlainTextResponse:
        return PlainTextResponse(root_of(request))

    @app.get("/run/{n}")
    def run(request: Request, n: int) -> HTMLResponse:
        return templates.TemplateResponse(request, "page.html", {})

    @app.get("/run/{n}/detail/")
    def detail(n: int) -> PlainTextResponse:
        return PlainTextResponse("detail")

    return app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(_app(tmp_path))


PREFIX = {"X-Forwarded-Prefix": "/portal"}


def test_root_is_empty_at_root(client: TestClient) -> None:
    assert client.get("/").text == ""


def test_header_becomes_root(client: TestClient) -> None:
    assert client.get("/", headers=PREFIX).text == "/portal"


def test_trailing_slash_prefix_is_normalized(client: TestClient) -> None:
    assert client.get("/", headers={"X-Forwarded-Prefix": "/portal/"}).text == "/portal"


@pytest.mark.parametrize(
    "bad",
    [
        "portal",
        "//evil.example",
        "/a b",
        "/",
        "/x\\y",
        "/p?x=1",
        "/p#frag",
        '/p"x',
        "/p<q>",
        "",
    ],
)
def test_malformed_prefixes_are_ignored(client: TestClient, bad: str) -> None:
    assert clean_prefix(bad) == ""
    assert client.get("/", headers={"X-Forwarded-Prefix": bad}).text == ""


def test_mount_named_like_a_route_head_still_routes(client: TestClient) -> None:
    # Without the path re-prefix Starlette double-strips "/run" and 404s.
    response = client.get("/run/1", headers={"X-Forwarded-Prefix": "/run"})
    assert response.status_code == 200
    assert 'href="/run/run/1"' in response.text


def test_trailing_slash_redirect_keeps_the_prefix(client: TestClient) -> None:
    response = client.get("/run/1/detail", headers=PREFIX, follow_redirects=False)
    assert response.status_code in (301, 307)
    assert response.headers["location"].endswith("/portal/run/1/detail/")


def test_templates_carry_root_globals_filters_and_processors(
    client: TestClient,
) -> None:
    text = client.get("/run/1", headers=PREFIX).text
    assert text == '<a href="/portal/run/1">/portal|extra|Finished as intended</a>'


def test_theme_mount_serves_the_kit_and_its_reference_page(client: TestClient) -> None:
    for name in ("theme-boot.js", "theme.css", "kit.css", "kit.js", "kit.html"):
        response = client.get(f"/theme/{name}", headers=PREFIX)
        assert response.status_code == 200, name
    assert ".kit .chip" in client.get("/theme/kit.css").text


def test_theme_mount_is_named_for_url_for(tmp_path: Path) -> None:
    app = _app(tmp_path)
    assert app.url_path_for("theme", path="kit.css") == "/theme/kit.css"
