"""References between notes: the permalink, the stored form, and arriving.

A logbook whose notes cannot cite each other makes the reader carry the
connection in their head — "the clock change is written up somewhere in
last Saturday". Three separate things have to hold for a citation to work,
and each is pinned here:

1. a note has **one name** that does not depend on which book it is in
   (``/entry/{id}``, which redirects to whichever page draws it);
2. what a body **stores** is relative (``entry/<id>``) — never the mount
   prefix or the host it was written on, for the same reason an attachment
   link is relative;
3. the target is **reachable** when you get there — which is a page
   concern, tested at the markup level here and exercised in a browser.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_logbook.app import create_app
from geecs_logbook.render import render_markdown

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples" / "logbook_templates"


@pytest.fixture
def app(share: Path, tmp_path: Path) -> TestClient:
    """A writable logbook, as the service runs it."""
    return TestClient(
        create_app(
            "Undulator",
            base_directory=share,
            notes_db=tmp_path / "notes.db",
            templates_dir=EXAMPLES,
        )
    )


@pytest.fixture
def readonly(share: Path) -> TestClient:
    """No store: no permalink route, and no tools that would offer one."""
    return TestClient(create_app("Undulator", base_directory=share))


def _entry(client: TestClient, **extra: object) -> dict:
    payload: dict = {"day": "2026-09-11", "author": "S. Barber", "body_md": "a note"}
    payload.update(extra)
    r = client.post("/api/entries", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


class TestStoredFormIsRelative:
    """What a body holds, and what the renderer does with it."""

    def test_a_reference_is_rewritten_onto_the_serving_route(self) -> None:
        """``entry/<id>`` becomes the permalink route, and says it is one."""
        out = render_markdown(
            "see [the jet note](entry/9f3a2b1c4d5e)", entry_base="/log/entry"
        )
        assert 'href="/log/entry/9f3a2b1c4d5e"' in out
        assert 'class="entryref"' in out

    def test_without_a_base_it_is_left_exactly_as_written(self) -> None:
        """An offline export keeps the relative link rather than a wrong one."""
        out = render_markdown("see [x](entry/9f3a2b1c4d5e)")
        assert 'href="entry/9f3a2b1c4d5e"' in out
        assert "entryref" not in out

    def test_only_an_id_shaped_target_is_taken_as_a_reference(self) -> None:
        """A relative link that merely starts with ``entry/`` is left alone.

        The rewrite is keyed on the id's alphabet, not on the word, so an
        ordinary link to some other page under that prefix keeps its href
        and does not get dressed up as a cross-reference.
        """
        out = render_markdown("[docs](entry/how-to-write-one)", entry_base="/log/entry")
        assert 'href="entry/how-to-write-one"' in out
        assert "entryref" not in out

    def test_an_absolute_link_is_never_rewritten(self) -> None:
        """Only the relative form is ours; a full URL is the author's."""
        out = render_markdown(
            "[x](https://elsewhere.test/entry/9f3a2b1c4d5e)", entry_base="/log/entry"
        )
        assert "https://elsewhere.test/entry/9f3a2b1c4d5e" in out
        assert "entryref" not in out

    def test_the_preview_renders_a_reference_like_the_page(
        self, app: TestClient
    ) -> None:
        """Preview is the page's own renderer, so it must pass the base too."""
        r = app.post("/api/preview", json={"body_md": "[x](entry/9f3a2b1c4d5e)"})
        assert r.status_code == 200, r.text
        assert 'class="entryref"' in r.json()["html"]


class TestPermalink:
    """One name for a note, whichever book and whichever page shape."""

    def test_a_scans_entry_lands_on_its_day_page(self, app: TestClient) -> None:
        entry = _entry(app, book="scans", scan=1)
        r = app.get(f"/entry/{entry['entry_id']}", follow_redirects=False)
        assert r.status_code in (302, 303, 307)
        assert r.headers["location"] == (f"/day/2026-09-11#entry-{entry['entry_id']}")

    def test_an_ops_entry_lands_on_its_month_page(self, app: TestClient) -> None:
        """The other book renders a month, and the permalink knows that."""
        entry = _entry(app, book="ops")
        r = app.get(f"/entry/{entry['entry_id']}", follow_redirects=False)
        assert r.headers["location"] == (f"/month/2026-09#entry-{entry['entry_id']}")

    def test_the_anchor_is_the_id_the_page_actually_uses(self, app: TestClient) -> None:
        """The fragment has to match the element, or the link lands nowhere."""
        entry = _entry(app, book="scans", scan=1)
        fragment = (
            app.get(f"/entry/{entry['entry_id']}", follow_redirects=False)
            .headers["location"]
            .split("#")[1]
        )
        assert f'id="{fragment}"' in app.get("/day/2026-09-11").text

    def test_an_unknown_id_is_a_404(self, app: TestClient) -> None:
        assert app.get("/entry/deadbeef0000").status_code == 404

    def test_a_deleted_entry_is_a_404(self, app: TestClient) -> None:
        """A tombstone is gone from every listing but the change feed."""
        entry = _entry(app, book="ops")
        assert app.delete(f"/api/entries/{entry['entry_id']}").status_code == 204
        assert app.get(f"/entry/{entry['entry_id']}").status_code == 404

    def test_a_read_only_logbook_has_no_permalink_route(
        self, readonly: TestClient
    ) -> None:
        """It is registered with the write verbs; without a store there is
        nothing to name, and the page offers no tools that would copy one."""
        assert readonly.get("/entry/deadbeef0000").status_code == 404
        assert "data-permalink" not in readonly.get("/day/2026-09-11").text


class TestPagesOfferTheLink:
    """Every drawn entry carries a copyable name, in both books."""

    def test_the_day_page_offers_one_per_entry(self, app: TestClient) -> None:
        entry = _entry(app, book="scans", scan=1)
        body = app.get("/day/2026-09-11").text
        assert f'href="/entry/{entry["entry_id"]}"' in body
        assert "data-permalink" in body

    def test_the_month_page_offers_one_per_entry(self, app: TestClient) -> None:
        entry = _entry(app, book="ops")
        body = app.get("/month/2026-09").text
        assert f'href="/entry/{entry["entry_id"]}"' in body

    def test_the_pages_tell_the_editor_where_permalinks_live(
        self, app: TestClient
    ) -> None:
        """editor.js recognises a pasted link by this prefix; without it a
        pasted permalink is pasted as a bare URL instead of a reference."""
        for page in ("/day/2026-09-11", "/month/2026-09"):
            assert 'data-entry-base="/entry"' in app.get(page).text
