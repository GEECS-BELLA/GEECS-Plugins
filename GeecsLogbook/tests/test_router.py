"""The mounted logbook router: routes, status codes and rendering."""

from __future__ import annotations

from pathlib import Path

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from geecs_logbook.router import create_log_router


@pytest.fixture
def client(share: Path) -> TestClient:
    """Mount the router the way the Data Portal does."""
    app = FastAPI()
    app.include_router(
        create_log_router("Undulator", base_directory=share), prefix="/log"
    )
    return TestClient(app)


class TestDayJson:
    """The JSON peer of the day page."""

    def test_returns_the_days_scans(self, client: TestClient) -> None:
        """Every scan folder present appears, in number order."""
        body = client.get("/log/api/day/2026-09-11").json()
        assert [s["number"] for s in body["scans"]] == [1, 6, 31, 40]
        assert body["exists"] is True

    def test_carries_status_and_failure_reason(self, client: TestClient) -> None:
        """A failed scan exposes why it failed."""
        scans = client.get("/log/api/day/2026-09-11").json()["scans"]
        failed = next(s for s in scans if s["number"] == 6)
        assert failed["status"] == "failed"
        assert "uc_amp4_ir_input-hdf-capture" in failed["failure_reason"]

    def test_empty_day_is_not_an_error(self, client: TestClient) -> None:
        """A date with no scans returns 200 and an empty day."""
        res = client.get("/log/api/day/2019-01-01")
        assert res.status_code == 200
        assert res.json()["exists"] is False

    def test_malformed_date_is_a_400(self, client: TestClient) -> None:
        """A path segment that is not YYYY-MM-DD is the caller's error."""
        assert client.get("/log/api/day/yesterday").status_code == 400


class TestDayPage:
    """The rendered day document."""

    def test_renders_each_scan(self, client: TestClient) -> None:
        """Every scan gets a collapsible block anchored by its label."""
        html = client.get("/log/day/2026-09-11").text
        assert 'id="Scan001"' in html
        assert 'id="Scan006"' in html
        assert 'id="Scan031"' in html

    def test_shows_the_purpose_from_scan_start_info(self, client: TestClient) -> None:
        """ScanStartInfo is displayed rather than asked for again."""
        html = client.get("/log/day/2026-09-11").text
        assert "807 phase 1 acceptance" in html

    def test_shows_the_failure_reason(self, client: TestClient) -> None:
        """The failure strip carries the ScanEndInfo text."""
        html = client.get("/log/day/2026-09-11").text
        assert "Scan ended with an error" in html
        assert "uc_amp4_ir_input-hdf-capture" in html

    def test_serves_its_stylesheet(self, client: TestClient) -> None:
        """The mounted static files resolve under the router prefix."""
        res = client.get("/log/static/scanlog.css")
        assert res.status_code == 200
        assert "--accent" in res.text

    def test_small_day_opens_expanded(self, client: TestClient) -> None:
        """Under the threshold every scan block starts open."""
        html = client.get("/log/day/2026-09-11").text
        assert html.count('<details class="panel scan"') == 4
        assert "Collapse all" in html


class TestHonestChips:
    """The status chip must not contradict the card under it."""

    def test_unfinalised_scan_does_not_claim_to_lack_scan_info(
        self, client: TestClient
    ) -> None:
        """Scan040 has a full ScanInfo, just no ScanEndInfo yet.

        Both it and a folder with no ScanInfo at all are `incomplete`, but
        rendering "no scan info" over a card listing the scan variable and
        shot count parsed *from* ScanInfo is a plain contradiction — and it
        hit the most common state on the real share.
        """
        html = client.get("/log/day/2026-09-11").text
        assert "not finalised" in html
        assert "no scan info" in html  # Scan031, which really has none
        assert html.count("not finalised") == 1
        assert html.count("no scan info") == 1


@pytest.fixture
def writable(share: Path, tmp_path: Path) -> TestClient:
    """The router with a notes store, as the portal mounts it with --notes-db."""
    app = FastAPI()
    app.include_router(
        create_log_router(
            "Undulator", base_directory=share, notes_db=tmp_path / "notes.db"
        ),
        prefix="/log",
    )
    return TestClient(app)


def _post(client: TestClient, **body: object) -> dict:
    payload = {"day": "2026-09-11", "author": "S. Barber", "body_md": "hello"}
    payload.update(body)
    r = client.post("/log/api/entries", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


class TestWriteApi:
    """The entry endpoints, as the composer and an agent use them."""

    def test_read_only_mount_has_no_write_routes(self, client: TestClient) -> None:
        """Without a store the write routes do not exist at all."""
        r = client.post(
            "/log/api/entries",
            json={"day": "2026-09-11", "author": "a", "body_md": "x", "scan": 1},
        )
        assert r.status_code in (404, 405)

    def test_day_level_entry_needs_no_scan(self, writable: TestClient) -> None:
        """Neither scan nor after: the entry is about the day, and lands at the root."""
        e = _post(writable)
        assert e["scan"] is None and e["after"] is None
        assert e["updated_at"] == e["created_at"]
        listed = writable.get("/log/api/day/2026-09-11/entries").json()
        assert [x["entry_id"] for x in listed] == [e["entry_id"]]
        page = writable.get("/log/day/2026-09-11").text
        assert "hello" in page

    def test_scan_zero_is_not_an_anchor(self, writable: TestClient) -> None:
        """Scans start at 1; 0 is refused rather than silently meaning the day."""
        r = writable.post(
            "/log/api/entries",
            json={"day": "2026-09-11", "author": "a", "body_md": "x", "scan": 0},
        )
        assert r.status_code == 422

    def test_both_anchors_is_a_422(self, writable: TestClient) -> None:
        """On a scan and after one at once is refused."""
        r = writable.post(
            "/log/api/entries",
            json={
                "day": "2026-09-11",
                "author": "a",
                "body_md": "x",
                "scan": 1,
                "after": 1,
            },
        )
        assert r.status_code == 422

    def test_an_agent_cannot_be_born_kept(self, writable: TestClient) -> None:
        """The draft rule holds at the API, not only in the docs."""
        r = writable.post(
            "/log/api/entries",
            json={
                "day": "2026-09-11",
                "author": "osprey",
                "body_md": "x",
                "scan": 1,
                "kind": "agent_analysis",
                "status": "kept",
            },
        )
        assert r.status_code == 422 and "draft" in r.text
        e = _post(
            writable, author="osprey", scan=1, kind="agent_analysis", status="draft"
        )
        assert e["status"] == "draft"
        kept = writable.post(
            f"/log/api/entries/{e['entry_id']}/status", json={"status": "kept"}
        ).json()
        assert kept["status"] == "kept" and kept["updated_at"] > e["updated_at"]

    def test_edit_conflict_carries_the_current_entry(
        self, writable: TestClient
    ) -> None:
        """A stale version is a 409 with what is there now."""
        e = _post(writable, scan=1)
        first = writable.patch(
            f"/log/api/entries/{e['entry_id']}",
            json={"body_md": "one", "editor": "a", "expected_version": 1},
        )
        assert first.status_code == 200 and first.json()["version"] == 2
        stale = writable.patch(
            f"/log/api/entries/{e['entry_id']}",
            json={"body_md": "two", "editor": "b", "expected_version": 1},
        )
        assert stale.status_code == 409
        assert stale.json()["detail"]["current"]["body_md"] == "one"

    def test_delete_hides_the_entry_everywhere(self, writable: TestClient) -> None:
        """After a delete: 204, then gone from the listing and from every route."""
        e = _post(writable, scan=1)
        assert writable.delete(f"/log/api/entries/{e['entry_id']}").status_code == 204
        assert writable.get("/log/api/day/2026-09-11/entries").json() == []
        assert writable.delete(f"/log/api/entries/{e['entry_id']}").status_code == 404
        again = writable.patch(
            f"/log/api/entries/{e['entry_id']}",
            json={"body_md": "x", "editor": "a", "expected_version": 2},
        )
        assert again.status_code == 404
        assert "hello" not in writable.get("/log/day/2026-09-11").text


_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class TestAttachments:
    """Upload lands on the host beside the database; serving is contained."""

    def _upload(
        self,
        client: TestClient,
        entry_id: str,
        name: str = "shot.png",
        data: bytes = _PNG,
        ctype: str = "image/png",
    ):
        return client.post(
            f"/log/api/entries/{entry_id}/attachments",
            files={"file": (name, data, ctype)},
        )

    def test_upload_then_serve(self, writable: TestClient, share: Path) -> None:
        """201 with a relative link; the served route returns the bytes."""
        e = _post(writable, scan=1)
        r = self._upload(writable, e["entry_id"])
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["link"] == f"attachments/{e['entry_id']}/shot.png"
        assert body["attachment"]["id"] != e["entry_id"]  # its own id
        got = writable.get(f"/log/attachments/{e['entry_id']}/shot.png")
        assert got.status_code == 200 and got.content == _PNG
        listed = writable.get("/log/api/day/2026-09-11/entries").json()
        assert [a["filename"] for a in listed[0]["attachments"]] == ["shot.png"]
        assert listed[0]["version"] == 2

    def test_same_name_twice_is_numbered_not_overwritten(
        self, writable: TestClient
    ) -> None:
        """Every clipboard paste is image.png; the second must not eat the first."""
        e = _post(writable, scan=1)
        first = self._upload(writable, e["entry_id"], "image.png", _PNG)
        second = self._upload(writable, e["entry_id"], "image.png", _PNG + b"2")
        assert first.json()["link"].endswith("/image.png")
        assert second.json()["link"].endswith("/image-2.png")
        base = f"/log/attachments/{e['entry_id']}"
        assert writable.get(f"{base}/image.png").content == _PNG
        assert writable.get(f"{base}/image-2.png").content == _PNG + b"2"
        ids = {
            a["id"]
            for a in writable.get("/log/api/day/2026-09-11/entries").json()[0][
                "attachments"
            ]
        }
        assert len(ids) == 2

    def test_type_and_size_limits(self, writable: TestClient) -> None:
        """415 for a type we do not take; 413 over the cap; 422 for nothing."""
        e = _post(writable, scan=1)
        assert (
            self._upload(
                writable, e["entry_id"], "x.exe", b"MZ", "application/octet-stream"
            ).status_code
            == 415
        )
        big = b"\x00" * (20 * 1024 * 1024 + 1)
        assert self._upload(writable, e["entry_id"], "big.png", big).status_code == 413
        assert self._upload(writable, e["entry_id"], "e.png", b"").status_code == 422
        assert self._upload(writable, "nope").status_code == 404

    def test_serving_is_contained(self, writable: TestClient) -> None:
        """No path escapes the entry's attachment directory."""
        e = _post(writable, scan=1)
        self._upload(writable, e["entry_id"])
        for bad in (
            "../../Scan001/shot.png",
            "..%2F..%2Fshot.png",
            "../../../scans/Scan001/ScanInfoScan001.ini",
        ):
            r = writable.get(f"/log/attachments/{e['entry_id']}/{bad}")
            assert r.status_code == 404, bad

    def test_upload_and_serving_work_without_the_share(
        self, writable: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bytes are store-first: an unmounted share costs nothing but the mirror."""
        from geecs_logbook import mirror

        e = _post(writable, scan=1)

        def no_share(*a: object, **k: object) -> Path:
            raise mirror.MirrorUnavailable("gone")

        monkeypatch.setattr(mirror, "logbook_root", no_share)
        r = self._upload(writable, e["entry_id"])
        assert r.status_code == 201, r.text
        assert (
            writable.get(f"/log/attachments/{e['entry_id']}/shot.png").content == _PNG
        )

    def test_rendered_page_links_to_the_serving_route(
        self, writable: TestClient
    ) -> None:
        """A body's relative link is rewritten onto /log/attachments/<id>/<file>."""
        e = _post(writable, scan=1)
        self._upload(writable, e["entry_id"])
        writable.patch(
            f"/log/api/entries/{e['entry_id']}",
            json={
                "body_md": f"![s](attachments/{e['entry_id']}/shot.png)",
                "editor": "a",
                "expected_version": 2,
            },
        )
        page = writable.get("/log/day/2026-09-11").text
        assert f'src="/log/attachments/{e["entry_id"]}/shot.png"' in page


class TestBooksTagsHistory:
    """The foundation for the ops book: books, tags and an entry's past."""

    def test_ops_entry_is_day_level_only(self, writable: TestClient) -> None:
        """An ops entry with a scan anchor is refused; without one it lands."""
        r = writable.post(
            "/log/api/entries",
            json={
                "day": "2026-09-11",
                "author": "a",
                "body_md": "x",
                "book": "ops",
                "scan": 1,
            },
        )
        assert r.status_code == 422
        e = _post(writable, book="ops", body_md="chiller filter swapped #maintenance")
        assert e["book"] == "ops" and e["tags"] == ["maintenance"]

    def test_day_page_shows_scans_book_and_counts_ops(
        self, writable: TestClient
    ) -> None:
        """Ops entries do not render on the day document; the API can filter either way."""
        _post(writable, scan=1, body_md="on the scan")
        _post(writable, book="ops", body_md="in the ops book")
        page = writable.get("/log/day/2026-09-11").text
        assert "on the scan" in page and "in the ops book" not in page
        both = writable.get("/log/api/day/2026-09-11/entries").json()
        ops = writable.get("/log/api/day/2026-09-11/entries?book=ops").json()
        assert len(both) == 2 and [x["body_md"] for x in ops] == ["in the ops book"]
        assert (
            writable.get("/log/api/day/2026-09-11/entries?book=nope").status_code == 422
        )

    def test_a_mirror_crash_never_fails_the_save(
        self, writable: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anything the mirror raises after the row landed is a deferral, not a 500."""
        from geecs_logbook import mirror

        def boom(*a: object, **k: object) -> None:
            raise RuntimeError("unexpected")

        monkeypatch.setattr(mirror, "mirror_one", boom)
        e = _post(writable, scan=1, body_md="still saved")
        assert (
            writable.get(f"/log/api/entries/{e['entry_id']}").json()["body_md"]
            == "still saved"
        )

    def test_tags_follow_the_body(self, writable: TestClient) -> None:
        """Tags are parsed at save and re-parsed on edit; the page shows chips."""
        e = _post(writable, scan=1, body_md="#laser tuned; see #jet")
        assert e["tags"] == ["laser", "jet"]
        page = writable.get("/log/day/2026-09-11").text
        assert "#laser</span>" in page and "#jet</span>" in page
        edited = writable.patch(
            f"/log/api/entries/{e['entry_id']}",
            json={"body_md": "just #jet now", "editor": "a", "expected_version": 1},
        ).json()
        assert edited["tags"] == ["jet"]

    def test_history_keeps_every_earlier_state(self, writable: TestClient) -> None:
        """Edit, keep, delete: each leaves the state it replaced, oldest first."""
        e = _post(writable, scan=1, body_md="v1")
        writable.patch(
            f"/log/api/entries/{e['entry_id']}",
            json={"body_md": "v2", "editor": "b", "expected_version": 1},
        )
        writable.post(
            f"/log/api/entries/{e['entry_id']}/status", json={"status": "draft"}
        )
        writable.delete(f"/log/api/entries/{e['entry_id']}")
        hist = writable.get(f"/log/api/entries/{e['entry_id']}/history").json()
        assert [(h["reason"], h["version"], h["entry"]["body_md"]) for h in hist] == [
            ("edit", 1, "v1"),
            ("status", 2, "v2"),
            ("delete", 3, "v2"),
        ]
        assert hist[2]["entry"]["status"] == "draft"
        assert writable.get("/log/api/entries/nope/history").status_code == 404


class TestEditorHooks:
    """What the page hands the editor script, and the preview it calls."""

    def test_page_carries_the_editors_facts_and_the_script(
        self, writable: TestClient
    ) -> None:
        """The editor reads api/day/book off <main id=logbook>; the script is served."""
        page = writable.get("/log/day/2026-09-11").text
        assert 'id="logbook"' in page and 'data-api="/log/api"' in page
        assert 'data-day="2026-09-11"' in page and 'data-book="scans"' in page
        assert (
            'data-accept="application/pdf,image/gif,image/jpeg,image/png,image/webp"'
            in page
        )
        assert "editor.js" in page
        js = writable.get("/log/static/editor.js")
        assert js.status_code == 200 and "uploadFiles" in js.text

    def test_preview_renders_like_the_page(self, writable: TestClient) -> None:
        """Preview is the same renderer with the same attachment base."""
        r = writable.post(
            "/log/api/preview",
            json={"body_md": "> [!TIP]\n> ok\n\n![p](attachments/e/a.png)"},
        )
        assert r.status_code == 200
        html = r.json()["html"]
        assert 'class="callout callout-tip"' in html
        assert 'src="/log/attachments/e/a.png"' in html

    def test_preview_needs_a_store(self, client: TestClient) -> None:
        """A read-only logbook has no composer, so no preview."""
        assert client.post("/log/api/preview", json={"body_md": "x"}).status_code in (
            404,
            405,
        )


def _scan_block_parents(html: str) -> set[tuple[str, ...]]:
    """Return the ancestor chains of every scan block, below ``<main>``.

    A flat day gives exactly ``{("main",)}``. Anything else means something
    was introduced around the loop — which is what a grouping is, whatever
    element or class name it wears.
    """
    from html.parser import HTMLParser

    class Walk(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.stack: list[str] = []
            self.found: set[tuple[str, ...]] = set()

        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            if tag == "details" and d.get("class") == "panel scan":
                if "main" in self.stack:
                    below = self.stack[self.stack.index("main") + 1 :]
                    self.found.add(("main", *below))
                else:
                    self.found.add(tuple(self.stack))
            if tag not in {"br", "img", "input", "meta", "link", "hr"}:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in self.stack:
                del self.stack[len(self.stack) - 1 - self.stack[::-1].index(tag) :]

    w = Walk()
    w.feed(html)
    return w.found


class TestLongDay:
    """A long day opens collapsed — a fact about volume, not about meaning.

    The grouped view this replaced inferred which scans belonged together
    from two matching fields. That is interpretation, and the logbook's
    rule is that it reports what the files say. What survives is the only
    honest part of the old behaviour: past a threshold the day is easier
    to read as a closed list.

    The first version of this class took the four-scan fixture, so ``many``
    was false in every test and it asserted nothing about collapsing at
    all — replacing the threshold with a literal ``false`` left the whole
    suite green. It needs a day that actually crosses the line.
    """

    @pytest.fixture
    def busy(self, make_run) -> TestClient:
        """A client over a day of 25 scans — past the 20 threshold."""
        app = FastAPI()
        app.include_router(
            create_log_router("Undulator", base_directory=make_run(25)),
            prefix="/log",
        )
        return TestClient(app)

    def test_a_long_day_starts_collapsed(self, busy: TestClient) -> None:
        """No scan block is open, and the button offers to expand."""
        html = busy.get("/log/day/2026-09-11").text
        assert html.count('<details class="panel scan"') == 25
        assert " open>" not in html
        assert "Expand all" in html

    def test_a_short_day_starts_open(self, client: TestClient) -> None:
        """Below the threshold every scan is readable without a click."""
        html = client.get("/log/day/2026-09-11").text
        assert "Collapse all" in html
        assert " open>" in html

    def test_no_day_groups_its_scans(self, busy: TestClient) -> None:
        """Every scan is a top-level block, whatever the day's length.

        Asserting on the rendered markup, not on the absence of the word
        "campaign": the old inline script carried `details.campaign` as a
        selector string, so a substring check passed for the wrong reason
        and would miss a grouping reintroduced under any other name.
        """
        html = busy.get("/log/day/2026-09-11").text
        # Structural, not spelling. A class-attribute regex missed two real
        # reintroductions: a grouping that is a <section> rather than a
        # <details> (the obvious next attempt, since the complaint was a
        # second *collapsible*), and a <details> whose class is not its
        # first attribute. Counting every <details> catches both.
        assert html.count("<details") == 26, "25 scans + the calendar"
        outer = re.findall(r'<details[^>]*class="([^"]*)"', html)
        assert set(outer) <= {"panel scan", "cal"}, outer
        # and nothing wraps the run. A regex cannot see this — consecutive
        # scan blocks legitimately sit next to each other — so parse, and
        # check each scan block's ancestors. A <section class="batch">
        # around the loop is the shape a class-attribute check misses.
        assert _scan_block_parents(html) == {("main",)}, _scan_block_parents(html)

    def test_the_rail_lists_scans_and_nothing_else(self, busy: TestClient) -> None:
        """The rail names scans, one row each, and groups nothing.

        The document-structure check above pins the body — but a grouping
        can come back without wrapping anything, as a rail section listing
        runs. That is not a hypothetical: it is the shape this change
        singles out as the harmful one, because "Campaigns · 15" rendered
        in the rail and told an operator the day held fifteen multi-week
        efforts. Verified by putting exactly that back and watching the
        suite stay green.

        So: one row per scan, and the rail's headings are exactly the two
        it is allowed to have.
        """
        html = busy.get("/log/day/2026-09-11").text
        assert html.count('class="scanrow"') == 25
        headings = {h.strip() for h in re.findall(r"<h4>(.*?)</h4>", html, re.S)}
        assert headings == {"Go to a day", "Scans &middot; 25"}, headings
