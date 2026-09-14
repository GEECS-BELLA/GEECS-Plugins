"""Both books fold their composers behind an affordance, on one contract.

The scan log has done this since composers were per-scan: an abandoned
note folds away with Close and nothing is discarded. The ops book kept a
single composer wedged open at the top of the month, so it had no Close
button at all — the asymmetry this pins shut.

What is actually load-bearing is the CONTRACT, because ``editor.js``
implements it once for every composer on either page:

  ``data-open-composer="X"``  a button that opens composer X
  ``data-compose-host="X"``   the hidden element holding it
  ``data-insert="X"``         an affordance row that folds away while X is
                              open — the day page's positional one, optional

A page that grows a composer without these is a text box with no way back,
which is the state the ops book was in.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_logbook.app import create_app

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples" / "logbook_templates"

#: Every composer on a page, by the anchor its host names.
_HOSTS = re.compile(r'data-compose-host="([^"]+)"')
_OPENERS = re.compile(r'data-open-composer="([^"]+)"')
_CLOSERS = re.compile(r'data-close-composer="([^"]+)"')


@pytest.fixture
def app(share: Path, tmp_path: Path) -> TestClient:
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
    return TestClient(create_app("Undulator", base_directory=share))


PAGES = ("/day/2026-09-11", "/month/2026-09")


@pytest.mark.parametrize("page", PAGES)
class TestEveryComposerFolds:
    """The rules that hold on both pages, stated once."""

    def test_every_composer_starts_folded(self, app: TestClient, page: str) -> None:
        """A page opens as a document, not as a wall of empty text boxes."""
        body = app.get(page).text
        hosts = _HOSTS.findall(body)
        assert hosts, f"{page} draws no composer at all"
        for anchor in hosts:
            host = re.search(
                r"<div([^>]*)data-compose-host=\"" + re.escape(anchor) + r"\"", body
            )
            assert host and "hidden" in host.group(1), (
                f"{page}: composer {anchor!r} is on screen before it is asked for"
            )

    def test_every_composer_has_something_that_opens_it(
        self, app: TestClient, page: str
    ) -> None:
        """A folded composer with no opener is a composer nobody can reach."""
        body = app.get(page).text
        assert set(_HOSTS.findall(body)) <= set(_OPENERS.findall(body))

    def test_every_composer_has_a_way_back(self, app: TestClient, page: str) -> None:
        """Close is the control this change exists to give the ops book.

        It folds without discarding — the form keeps its text — so it is
        safe to reach for, and Esc is wired to the same thing.
        """
        body = app.get(page).text
        assert set(_HOSTS.findall(body)) == set(_CLOSERS.findall(body)), (
            f"{page}: a composer here cannot be closed again"
        )

    def test_nothing_opens_a_composer_that_is_not_there(
        self, app: TestClient, page: str
    ) -> None:
        """A dead opener is a button that appears to do nothing."""
        body = app.get(page).text
        assert set(_OPENERS.findall(body)) <= set(_HOSTS.findall(body))

    def test_a_read_only_page_has_no_composer_and_no_opener(
        self, readonly: TestClient, page: str
    ) -> None:
        body = readonly.get(page).text
        assert not _HOSTS.findall(body)
        assert not _OPENERS.findall(body)


class TestTheOpsBookKeepsOneDatedComposer:
    """Why the ops book does not grow a composer per day group."""

    def test_one_composer_for_the_whole_month(self, app: TestClient) -> None:
        """It takes a date instead — which is what makes one enough."""
        body = app.get("/month/2026-09").text
        assert _HOSTS.findall(body) == ["ops"]
        assert 'class="when"' in body

    def test_each_day_heading_points_that_composer_at_its_day(
        self, app: TestClient
    ) -> None:
        """ "+ note" on a day group carries the day it means.

        editor.js opens the composer and sets the date from this one
        attribute, so the reveal and the date cannot come apart — they used
        to be two listeners on the same button, and the scroll ran while the
        composer was still hidden.
        """
        r = app.post(
            "/api/entries",
            json={
                "day": "2026-09-11",
                "book": "ops",
                "author": "S. Barber",
                "body_md": "jet line back up",
            },
        )
        assert r.status_code == 201, r.text
        body = app.get("/month/2026-09").text
        opener = re.search(
            r'<button[^>]*data-open-composer="ops"[^>]*'
            r'data-compose-day="2026-09-11"[^>]*>',
            body,
        )
        assert opener, "the day heading's + note does not name its day"


class TestTheDayPageKeepsItsPositionalAffordance:
    """The rule row IS the position, which the ops book's opener is not."""

    def test_each_anchor_has_a_rule_row_naming_it(self, app: TestClient) -> None:
        body = app.get("/day/2026-09-11").text
        rows = set(re.findall(r'data-insert="([^"]+)"', body))
        assert rows == set(_HOSTS.findall(body))

    def test_the_ops_book_has_no_rule_row(self, app: TestClient) -> None:
        """Its composer has no place on the page; a rule row would imply one."""
        assert "data-insert=" not in app.get("/month/2026-09").text
