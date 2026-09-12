"""The ops book: the month page, its filters, and the links between the books."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from geecs_logbook.router import create_log_router

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples" / "logbook_templates"


@pytest.fixture
def app(share: Path, tmp_path: Path) -> TestClient:
    """A writable logbook with the example templates, mounted as the portal does."""
    app = FastAPI()
    app.include_router(
        create_log_router(
            "Undulator",
            base_directory=share,
            notes_db=tmp_path / "notes.db",
            templates_dir=EXAMPLES,
        ),
        prefix="/log",
    )
    return TestClient(app)


@pytest.fixture
def readonly(share: Path) -> TestClient:
    """No store: the month page exists but holds nothing."""
    app = FastAPI()
    app.include_router(
        create_log_router("Undulator", base_directory=share), prefix="/log"
    )
    return TestClient(app)


def _ops(client: TestClient, day: str, body: str, **extra: object) -> dict:
    payload = {"day": day, "book": "ops", "author": "S. Barber", "body_md": body}
    payload.update(extra)
    r = client.post("/log/api/entries", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


class TestMonthPage:
    """One month of the ops book, newest day first, from the store alone."""

    def test_groups_by_day_newest_first(self, app: TestClient) -> None:
        """Days descend; within a day, entries read in the order written."""
        _ops(app, "2026-09-03", "first on the 3rd")
        _ops(app, "2026-09-11", "on the 11th")
        _ops(app, "2026-09-03", "second on the 3rd")
        _ops(app, "2026-08-31", "last month")
        _ops(app, "2026-10-01", "next month")
        html = app.get("/log/month/2026-09").text
        assert html.index('id="day-2026-09-11"') < html.index('id="day-2026-09-03"')
        assert html.index("first on the 3rd") < html.index("second on the 3rd")
        assert "last month" not in html and "next month" not in html
        assert "Days<b>2</b>" in html and "Notes<b>3</b>" in html

    def test_scans_book_stays_out(self, app: TestClient) -> None:
        """A note on the day page is not an ops note, anchored or not."""
        r = app.post(
            "/log/api/entries",
            json={
                "day": "2026-09-11",
                "author": "a",
                "body_md": "scan note",
                "scan": 1,
            },
        )
        assert r.status_code == 201
        app.post(
            "/log/api/entries",
            json={"day": "2026-09-11", "author": "a", "body_md": "day intro"},
        )
        html = app.get("/log/month/2026-09").text
        assert "scan note" not in html and "day intro" not in html
        assert "No operations notes in September 2026 yet" in html

    def test_never_reads_the_share(
        self, app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The month page is the store's alone: a dead share changes nothing."""
        from geecs_logbook import scan_reader
        from geecs_logbook.routes import _common

        def no_share(*a: object, **k: object) -> None:
            raise AssertionError("month page touched the share")

        monkeypatch.setattr(scan_reader, "read_day", no_share)
        monkeypatch.setattr(_common, "read_day", no_share)
        _ops(app, "2026-09-11", "still here")
        assert "still here" in app.get("/log/month/2026-09").text

    def test_tag_chips_filter_through_the_url(self, app: TestClient) -> None:
        """Chips count the whole month; the list shows the filtered part."""
        _ops(app, "2026-09-11", "chiller #maintenance")
        _ops(app, "2026-09-10", "energy up #laser")
        _ops(app, "2026-09-09", "amplifier #laser #maintenance")
        html = app.get("/log/month/2026-09?tag=laser").text
        assert "energy up" in html and "amplifier" in html and "chiller" not in html
        assert "Notes<b>2 / 3</b>" in html
        assert 'href="/log/month/2026-09?tag=laser"' in html
        assert 'href="/log/month/2026-09?tag=maintenance"' in html
        assert "clear filter" in html
        assert 'class="tagchip is-active"' in html
        empty = app.get("/log/month/2026-09?tag=nothing").text
        assert "No notes tagged <code>#nothing</code>" in empty

    def test_prev_next_and_day_links(self, app: TestClient) -> None:
        """Month stepping across a year boundary; each day links to its day page."""
        _ops(app, "2026-01-05", "january")
        html = app.get("/log/month/2026-01").text
        assert 'href="/log/month/2025-12"' in html
        assert 'href="/log/month/2026-02"' in html
        assert 'href="/log/day/2026-01-05"' in html
        assert 'data-compose-day="2026-01-05"' in html

    def test_malformed_month_is_a_400(self, app: TestClient) -> None:
        """Not YYYY-MM is the caller's error."""
        assert app.get("/log/month/2026-9").status_code == 400
        assert app.get("/log/month/september").status_code == 400
        assert app.get("/log/api/month/2026-13/entries").status_code == 400

    def test_composer_defaults_to_today_when_in_the_month(
        self, app: TestClient
    ) -> None:
        """Today in the shown month; the month's edge otherwise."""
        today = date.today()
        html = app.get(f"/log/month/{today.strftime('%Y-%m')}").text
        assert f'class="when" value="{today.isoformat()}"' in html
        assert (
            'data-book="ops"' in html
            and "data-day=" not in html.split("<main")[1].split(">")[0]
        )
        past = app.get("/log/month/2020-02").text
        assert 'class="when" value="2020-02-29"' in past
        future = app.get("/log/month/2099-03").text
        assert 'class="when" value="2099-03-01"' in future

    def test_read_only_mount_has_no_composer(self, readonly: TestClient) -> None:
        """Without a store: the page exists, says so, and offers nothing to write."""
        html = readonly.get("/log/month/2026-09").text
        assert "Read-only" in html and 'class="composer"' not in html
        assert readonly.get("/log/api/month/2026-09/entries").json() == []


class TestMonthJson:
    """The month's entries as JSON, either book, optionally one tag."""

    def test_book_and_tag_filters(self, app: TestClient) -> None:
        """Default is the ops book; ?book= and ?tag= narrow the same query."""
        _ops(app, "2026-09-11", "#laser one")
        _ops(app, "2026-09-12", "#jet two")
        app.post(
            "/log/api/entries",
            json={"day": "2026-09-11", "author": "a", "body_md": "scan", "scan": 1},
        )
        ops = app.get("/log/api/month/2026-09/entries").json()
        assert [e["body_md"] for e in ops] == ["#laser one", "#jet two"]
        laser = app.get("/log/api/month/2026-09/entries?tag=laser").json()
        assert [e["body_md"] for e in laser] == ["#laser one"]
        scans = app.get("/log/api/month/2026-09/entries?book=scans").json()
        assert [e["body_md"] for e in scans] == ["scan"]
        assert app.get("/log/api/month/2026-09/entries?book=nope").status_code == 422


class TestTypeButtons:
    """Templates as buttons on both pages, and the chip on a stored entry."""

    def test_each_page_offers_its_books_templates(self, app: TestClient) -> None:
        """Ops buttons on the month page, scans buttons on the day page, both-book ones on each."""
        month = app.get("/log/month/2026-09").text
        day = app.get("/log/day/2026-09-11").text
        assert 'data-type="laser"' in month and 'data-type="laser"' not in day
        assert 'data-type="result"' in day and 'data-type="result"' not in month
        assert 'data-type="fault"' in month and 'data-type="fault"' in day
        # Row order follows the files' order key.
        assert month.index('data-type="laser"') < month.index('data-type="handover"')
        assert month.index('data-type="handover"') < month.index('data-type="fault"')

    def test_buttons_carry_a_tone_class_never_a_colour(self, app: TestClient) -> None:
        """The file said ``colour: crit``; the page says ``tone-crit``."""
        html = app.get("/log/month/2026-09").text
        assert 'class="typebtn tone-crit" type="button" data-type="fault"' in html
        assert 'class="typebtn tone-ok" type="button" data-type="laser"' in html

    def test_prefills_ship_once_as_json(self, app: TestClient) -> None:
        """One JSON block carries every prefill; the editor reads it by name."""
        html = app.get("/log/month/2026-09").text
        assert html.count('id="logbook-seeds"') == 1
        block = html.split('id="logbook-seeds">')[1].split("</script>")[0]
        import json

        seeds = json.loads(block)
        assert seeds["laser"].startswith("#laser")
        assert "result" in seeds  # every template, not only this book's

    def test_stored_template_renders_as_its_chip(self, app: TestClient) -> None:
        """An entry started from a template shows the label in the file's tone."""
        _ops(app, "2026-09-11", "### Laser\n#laser\nenergy up", template="laser")
        _ops(app, "2026-09-11", "plain", template="retired_type")
        html = app.get("/log/month/2026-09").text
        assert '<span class="chip chip-type tone-ok">Laser</span>' in html
        assert '<span class="chip chip-quiet">retired_type</span>' in html
        assert '<span class="chip chip-tag">#laser</span>' in html

    def test_without_a_directory_composers_are_plain(
        self, share: Path, tmp_path: Path
    ) -> None:
        """No templates configured: no button row, an empty prefill block."""
        app = FastAPI()
        app.include_router(
            create_log_router(
                "Undulator", base_directory=share, notes_db=tmp_path / "n.db"
            ),
            prefix="/log",
        )
        html = TestClient(app).get("/log/month/2026-09").text
        assert 'class="types"' not in html and "typebtn" not in html
        assert 'id="logbook-seeds">{}</script>' in html


class TestCrossLinks:
    """The two books point at each other."""

    def test_day_page_counts_ops_notes_and_links_to_the_month(
        self, app: TestClient
    ) -> None:
        """The strip appears only when there is something behind it."""
        before = app.get("/log/day/2026-09-11").text
        assert "opsstrip" not in before
        assert 'href="/log/month/2026-09"' in before  # the topbar link, always
        _ops(app, "2026-09-11", "a")
        _ops(app, "2026-09-11", "b")
        _ops(app, "2026-09-12", "other day")
        after = app.get("/log/day/2026-09-11").text
        assert "2 ops notes today" in after
        assert 'href="/log/month/2026-09#day-2026-09-11"' in after

    def test_month_page_links_back_to_the_scan_log(self, app: TestClient) -> None:
        """Topbar: today's day page."""
        html = app.get("/log/month/2026-09").text
        assert f'href="/log/day/{date.today().isoformat()}"' in html
