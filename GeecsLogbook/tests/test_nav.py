"""Navigation: the rail calendar's marks, the today names, keyboard wiring."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from geecs_logbook.router import create_log_router
from geecs_logbook.scan_reader import days_with_folders


@pytest.fixture
def app(share: Path, tmp_path: Path) -> TestClient:
    """A writable logbook over the synthetic share."""
    app = FastAPI()
    app.include_router(
        create_log_router(
            "Undulator", base_directory=share, notes_db=tmp_path / "notes.db"
        ),
        prefix="/log",
    )
    return TestClient(app)


def _note(client: TestClient, day: str, body: str, **extra: object) -> None:
    payload = {"day": day, "author": "S. Barber", "body_md": body, **extra}
    assert client.post("/log/api/entries", json=payload).status_code == 201


class TestDaysWithFolders:
    """One listing of the month folder, never a walk of the days."""

    def test_lists_the_month_folder_once(
        self, share: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fixture's one day folder is found with a single scandir."""
        import os

        from geecs_logbook import scan_reader

        calls: list[str] = []
        real = os.scandir

        def counted(path: object) -> object:
            calls.append(str(path))
            return real(path)

        monkeypatch.setattr(scan_reader.os, "scandir", counted)
        found = days_with_folders(date(2026, 9, 1), "Undulator", base_directory=share)
        assert found == {date(2026, 9, 11)}
        assert len(calls) == 1 and calls[0].endswith("09-Sep")

    def test_ignores_strays_and_other_months(self, share: Path) -> None:
        """Only ``YY_MMDD`` folders of this month are days; files and strays are not."""
        month = share / "Undulator" / "Y2026" / "09-Sep"
        (month / "26_1001").mkdir()  # October's shape, filed in the wrong month
        (month / "26_0932").mkdir()  # not a date
        (month / "notes").mkdir()
        (month / "26_0912").write_text("a file, not a day")
        assert days_with_folders(
            date(2026, 9, 1), "Undulator", base_directory=share
        ) == {date(2026, 9, 11)}

    def test_missing_month_is_empty_missing_share_is_none(
        self, share: Path, tmp_path: Path
    ) -> None:
        """No month folder means nothing happened; no experiment dir means no share."""
        assert (
            days_with_folders(date(2027, 1, 1), "Undulator", base_directory=share)
            == set()
        )
        assert (
            days_with_folders(
                date(2026, 9, 1), "Undulator", base_directory=tmp_path / "unmounted"
            )
            is None
        )

    def test_never_creates_anything(
        self, share: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scan-folder invariant: a listing makes no directories."""

        def explode(*a: object, **k: object) -> None:
            raise AssertionError("days_with_folders called mkdir")

        monkeypatch.setattr(Path, "mkdir", explode)
        days_with_folders(date(2027, 1, 1), "Undulator", base_directory=share)


class TestMonthMarks:
    """``/api/month/{m}/days`` — what the calendar draws."""

    def test_marks_folders_and_notes_per_book(self, app: TestClient) -> None:
        """Folder days from the share, note counts from the store, merged."""
        _note(app, "2026-09-11", "on a scan", scan=1)
        _note(app, "2026-09-11", "ops that day", book="ops")
        _note(app, "2026-09-03", "quiet day #laser", book="ops")
        body = app.get("/log/api/month/2026-09/days").json()
        assert body["month"] == "2026-09" and body["share"] is True
        assert body["days"] == {
            "2026-09-03": {"folder": False, "notes": 0, "ops": 1},
            "2026-09-11": {"folder": True, "notes": 1, "ops": 1},
        }

    def test_share_down_is_said_not_hidden(self, tmp_path: Path) -> None:
        """No experiment directory: the store's marks stand, ``share`` is false."""
        app = FastAPI()
        app.include_router(
            create_log_router(
                "Undulator",
                base_directory=tmp_path / "unmounted",
                notes_db=tmp_path / "notes.db",
            ),
            prefix="/log",
        )
        client = TestClient(app)
        _note(client, "2026-09-11", "still counted", book="ops")
        body = client.get("/log/api/month/2026-09/days").json()
        assert body["share"] is False
        assert body["days"] == {"2026-09-11": {"folder": False, "notes": 0, "ops": 1}}

    def test_readonly_logbook_marks_folders_only(self, share: Path) -> None:
        """Without a store there are no notes to count, and no error."""
        app = FastAPI()
        app.include_router(
            create_log_router("Undulator", base_directory=share), prefix="/log"
        )
        body = TestClient(app).get("/log/api/month/2026-09/days").json()
        assert body["days"] == {"2026-09-11": {"folder": True, "notes": 0, "ops": 0}}

    def test_bad_month_is_400(self, app: TestClient) -> None:
        """The same month grammar as the page."""
        assert app.get("/log/api/month/2026-9/days").status_code == 400

    def test_month_page_never_calls_it(
        self, app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The marks are the calendar's lazy fetch, not the month page's path."""
        from geecs_logbook.routes import month

        def explode(*a: object, **k: object) -> None:
            raise AssertionError("month page listed the share")

        monkeypatch.setattr(month, "days_with_folders", explode)
        assert app.get("/log/month/2026-09").status_code == 200


class TestTodayNames:
    """Bookmarkable names for today, in both books."""

    def test_log_today_redirects_to_the_day(self, app: TestClient) -> None:
        res = app.get("/log/today", follow_redirects=False)
        assert res.status_code == 307
        assert res.headers["location"] == f"day/{date.today().isoformat()}"

    def test_month_today_lands_on_todays_group(self, app: TestClient) -> None:
        res = app.get("/log/month/today", follow_redirects=False)
        today = date.today()
        assert res.status_code == 307
        assert res.headers["location"] == (
            f"{today.strftime('%Y-%m')}#day-{today.isoformat()}"
        )


class TestRailWiring:
    """What the pages hand nav.js, and the controls they draw."""

    def test_day_page_carries_step_targets_and_the_calendar(
        self, app: TestClient
    ) -> None:
        html = app.get("/log/day/2026-09-11").text
        assert 'data-prev="/log/day/2026-09-10"' in html
        assert 'data-next="/log/day/2026-09-12"' in html
        assert f'data-today="/log/day/{date.today().isoformat()}"' in html
        assert '<details class="cal">' in html
        assert 'src="/log/static/nav.js"' in html

    def test_month_page_always_offers_today(self, app: TestClient) -> None:
        """The month rail's Today control is present in every month."""
        today = date.today()
        target = f"/log/month/{today.strftime('%Y-%m')}#day-{today.isoformat()}"
        this_month = app.get(f"/log/month/{today.strftime('%Y-%m')}").text
        other = app.get("/log/month/2019-01").text
        assert f'class="todaylink" href="{target}"' in this_month
        assert f'class="todaylink" href="{target}"' in other
        assert ">\n        Today</a>" in this_month
        assert "Back to this month</a>" in other
        assert 'data-month="2019-01"' in other and 'data-today="' + target in other
        assert 'data-prev="/log/month/2018-12"' in other
        assert '<details class="cal">' in other

    def test_nav_script_is_served(self, app: TestClient) -> None:
        res = app.get("/log/static/nav.js")
        assert res.status_code == 200 and "prefetch" in res.text


class TestChangeFeed:
    """``GET /api/entries?since=`` — the synchroniser's listing."""

    def test_requires_since_or_cursor_and_a_timezone(self, app: TestClient) -> None:
        assert app.get("/log/api/entries").status_code == 422
        naive = app.get("/log/api/entries", params={"since": "2026-09-11T08:00:00"})
        assert naive.status_code == 422 and "timezone" in naive.text
        assert app.get("/log/api/entries", params={"cursor": "junk"}).status_code == 422

    def test_lists_changes_with_tombstones_and_pages(self, app: TestClient) -> None:
        """Create, delete, and read the feed back: both rows, delete last."""
        _note(app, "2026-09-11", "keep me", book="ops")
        gone = app.post(
            "/log/api/entries",
            json={"day": "2026-09-11", "author": "a", "body_md": "drop me"},
        ).json()
        assert app.delete(f"/log/api/entries/{gone['entry_id']}").status_code == 204
        since = "2000-01-01T00:00:00+00:00"
        body = app.get("/log/api/entries", params={"since": since}).json()
        assert [e["body_md"] for e in body["entries"]] == ["keep me", "drop me"]
        assert body["entries"][1]["deleted_at"] is not None
        assert body["next_cursor"] is None
        # Hidden everywhere else.
        listed = app.get("/log/api/day/2026-09-11/entries").json()
        assert [e["body_md"] for e in listed] == ["keep me"]
        # Paged: one per page, the cursor carries on.
        first = app.get("/log/api/entries", params={"since": since, "limit": 1}).json()
        assert [e["body_md"] for e in first["entries"]] == ["keep me"]
        second = app.get(
            "/log/api/entries", params={"cursor": first["next_cursor"], "limit": 1}
        ).json()
        assert [e["body_md"] for e in second["entries"]] == ["drop me"]
        assert second["next_cursor"] is None
        # Narrowed to a book, and to live rows.
        ops = app.get("/log/api/entries", params={"since": since, "book": "ops"}).json()
        assert [e["body_md"] for e in ops["entries"]] == ["keep me"]
        live = app.get(
            "/log/api/entries", params={"since": since, "include_deleted": "false"}
        ).json()
        assert [e["body_md"] for e in live["entries"]] == ["keep me"]

    def test_not_served_without_a_store(self, share: Path) -> None:
        app = FastAPI()
        app.include_router(
            create_log_router("Undulator", base_directory=share), prefix="/log"
        )
        res = TestClient(app).get(
            "/log/api/entries", params={"since": "2000-01-01T00:00:00+00:00"}
        )
        assert res.status_code == 404
