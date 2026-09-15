"""Sending a Plot-tab figure to the scan's logbook entry.

Two layers, tested apart: the four-call conversation with the logbook
(:mod:`geecs_portal.logbook_send`, driven against an in-process fake
logbook through an httpx transport) and the route's gate ladder (through
``TestClient``, with the conversation stubbed out).
"""

from __future__ import annotations

import base64
import dataclasses
import json
from typing import Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from geecs_portal import logbook_send
from geecs_portal.app import create_app
from test_app import FakeCatalog

_UID = "uid-002"  # TEST_DAY, Scan 002 in the fake catalog
_BASE = "http://logbook.example:8400"

#: A one-pixel PNG — the bytes never matter here, only that they survive.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_DATA_URL = "data:image/png;base64," + base64.b64encode(_PNG).decode()


class FakeLogbook:
    """The logbook's write verbs, enough of them to hold a conversation.

    Entries are dicts with a ``version`` that moves on every write, which
    is the part that matters: the real store rejects a stale one, so a
    client that patches with the version it read *before* uploading is a
    bug this fake can see.
    """

    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}
        self.created: list[dict] = []
        self.uploads: list[bytes] = []
        self._next = 0
        #: Set to fail the next N patches with 409, to drive the retry.
        self.conflicts = 0
        #: Entry ids that answer 404 to an upload (deleted under us).
        self.gone: set[str] = set()
        #: Every PATCH attempt, 409s included — the retry loop would
        #: otherwise heal a stale-version bug into an invisible one.
        self.patch_attempts = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/api/entries":
            body = json.loads(request.content)
            self._next += 1
            entry_id = f"e{self._next:04d}"
            self.entries[entry_id] = {
                "entry_id": entry_id,
                "version": 1,
                "body_md": body["body_md"],
                "day": body["day"],
                "scan": body["scan"],
                "author": body["author"],
                "book": body["book"],
            }
            self.created.append(body)
            return httpx.Response(201, json=self.entries[entry_id])

        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[:2] == ["api", "entries"]:
            entry_id = parts[2]
            if entry_id in self.gone or entry_id not in self.entries:
                return httpx.Response(404, json={"detail": "no such entry"})
            entry = self.entries[entry_id]
            if len(parts) == 4 and parts[3] == "attachments":
                self.uploads.append(request.content)
                entry["version"] += 1  # an upload moves the version
                name = f"plot-{len(self.uploads)}.png"
                return httpx.Response(
                    201,
                    json={
                        "attachment": {"filename": name},
                        "link": f"attachments/{entry_id}/{name}",
                    },
                )
            if request.method == "GET":
                return httpx.Response(200, json=entry)
            if request.method == "PATCH":
                body = json.loads(request.content)
                self.patch_attempts += 1
                if self.conflicts:
                    self.conflicts -= 1
                    return httpx.Response(409, json={"detail": "version moved"})
                if body["expected_version"] != entry["version"]:
                    return httpx.Response(409, json={"detail": "stale version"})
                entry["body_md"] = body["body_md"]
                entry["version"] += 1
                return httpx.Response(200, json=entry)
        return httpx.Response(404, json={"detail": f"no route {path}"})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def only_entry(self) -> dict:
        assert len(self.entries) == 1, self.entries
        return next(iter(self.entries.values()))


def _send(book: FakeLogbook, **over) -> logbook_send.SendResult:
    kwargs = {
        "base_url": _BASE,
        "day": "2026-07-12",
        "scan": 2,
        "author": "Ada",
        "png": _PNG,
        "caption": "jet_pressure vs shot",
        "source_url": "http://portal/run/uid-002?y=jet_pressure",
    }
    kwargs.update(over)
    with book.client() as client:
        return logbook_send.send_plot(client=client, **kwargs)


class TestPureShapes:
    """The bits of markdown the body is assembled from."""

    def test_image_markdown_keeps_the_caption_as_alt_text(self) -> None:
        md = logbook_send.image_markdown("jet vs shot", "attachments/e1/p.png")
        assert md == "![jet vs shot](attachments/e1/p.png)"

    def test_brackets_in_a_caption_cannot_break_the_link(self) -> None:
        """A ``]`` in alt text would end the link early and orphan the image."""
        md = logbook_send.image_markdown("U [kV] vs shot", "attachments/e1/p.png")
        assert md == "![U (kV) vs shot](attachments/e1/p.png)"
        assert md.count("](") == 1

    def test_an_empty_caption_still_names_something(self) -> None:
        assert logbook_send.image_markdown("   ", "a.png") == "![plot](a.png)"

    def test_append_separates_paragraphs_with_a_blank_line(self) -> None:
        """Consecutive image PARAGRAPHS are what the logbook grids."""
        out = logbook_send.appended_body("![a](1.png)\n", "![b](2.png)")
        assert out == "![a](1.png)\n\n![b](2.png)\n"

    def test_append_to_an_empty_body_adds_no_leading_blank_line(self) -> None:
        assert logbook_send.appended_body("", "![b](2.png)") == "![b](2.png)\n"


class TestDataUrl:
    """What the browser hands over, and what is refused."""

    def test_round_trip(self) -> None:
        assert logbook_send.decode_png_data_url(_DATA_URL) == _PNG

    def test_a_non_png_data_url_is_refused(self) -> None:
        """The logbook keys the stored extension off the content type."""
        jpeg = "data:image/jpeg;base64," + base64.b64encode(_PNG).decode()
        with pytest.raises(ValueError, match="data:image/png"):
            logbook_send.decode_png_data_url(jpeg)

    def test_a_bare_url_is_refused(self) -> None:
        with pytest.raises(ValueError, match="data:image/png"):
            logbook_send.decode_png_data_url("http://example/plot.png")

    def test_broken_base64_is_refused(self) -> None:
        with pytest.raises(ValueError, match="base64"):
            logbook_send.decode_png_data_url("data:image/png;base64,not!base64")


class TestSendCreates:
    """The first plot for a scan makes the entry it lives in."""

    def test_entry_is_created_on_the_scan(self) -> None:
        book = FakeLogbook()
        result = _send(book)
        assert result.appended is False
        created = book.created[0]
        assert created["day"] == "2026-07-12"
        assert created["scan"] == 2
        assert created["book"] == "scans"
        assert created["author"] == "Ada"

    def test_the_png_reaches_the_store(self) -> None:
        book = FakeLogbook()
        _send(book)
        assert len(book.uploads) == 1
        assert _PNG in book.uploads[0]  # multipart-wrapped, bytes intact

    def test_the_body_links_the_image_the_store_named(self) -> None:
        """The store claims the free filename; the body must use THAT one."""
        book = FakeLogbook()
        result = _send(book)
        body = book.only_entry()["body_md"]
        assert result.link == "attachments/e0001/plot-1.png"
        assert "![jet_pressure vs shot](attachments/e0001/plot-1.png)" in body

    def test_the_portal_link_sits_above_the_images(self) -> None:
        """A paragraph BETWEEN two images would split the run the renderer grids."""
        book = FakeLogbook()
        _send(book)
        body = book.only_entry()["body_md"]
        assert body.index("http://portal/run/uid-002") < body.index("![jet_pressure")

    def test_no_source_url_leaves_the_body_starting_with_the_image(self) -> None:
        book = FakeLogbook()
        _send(book, source_url="")
        assert book.only_entry()["body_md"].startswith("![jet_pressure")

    def test_the_patch_uses_the_version_from_after_the_upload(self) -> None:
        """The upload moves the version, so a value read before it is stale.

        Asserting on the RESULT would not catch this: the retry re-reads
        and the second attempt succeeds, so a stale-version bug heals
        itself into a body that looks right. The wasted attempt is the
        only evidence, so that is what is asserted.
        """
        book = FakeLogbook()
        _send(book)
        assert book.patch_attempts == 1
        assert book.only_entry()["body_md"].count("![") == 1


class TestSendAppends:
    """A second plot joins the first, which is what makes a figure grid."""

    def test_two_images_land_in_one_entry_as_a_run(self) -> None:
        book = FakeLogbook()
        first = _send(book)
        second = _send(book, entry_id=first.entry_id, caption="charge vs shot")
        assert second.appended is True
        assert second.entry_id == first.entry_id
        body = book.only_entry()["body_md"]
        assert "![jet_pressure vs shot](attachments/e0001/plot-1.png)" in body
        assert "![charge vs shot](attachments/e0001/plot-2.png)" in body
        # Blank line between them: two image PARAGRAPHS, not one paragraph.
        assert "plot-1.png)\n\n![charge" in body

    def test_appending_does_not_repeat_the_portal_link(self) -> None:
        book = FakeLogbook()
        first = _send(book)
        _send(book, entry_id=first.entry_id, source_url="http://portal/other")
        assert book.only_entry()["body_md"].count("http://portal/") == 1

    def test_a_conflict_is_retried_against_the_fresh_body(self) -> None:
        """Someone saving the entry mid-send must not lose their sentence."""
        book = FakeLogbook()
        first = _send(book)
        book.entries[first.entry_id]["body_md"] += "\n\nthe jet was misaligned\n"
        book.entries[first.entry_id]["version"] += 1
        book.conflicts = 1
        _send(book, entry_id=first.entry_id, caption="charge vs shot")
        body = book.only_entry()["body_md"]
        assert "the jet was misaligned" in body
        assert "![charge vs shot]" in body

    def test_a_relentless_conflict_gives_up_rather_than_spinning(self) -> None:
        book = FakeLogbook()
        first = _send(book)
        book.conflicts = 99
        with pytest.raises(logbook_send.LogbookRefused) as caught:
            _send(book, entry_id=first.entry_id)
        assert caught.value.status == 409

    def test_an_entry_that_is_gone_starts_a_fresh_one(self) -> None:
        """The browser's memory of an entry outlives the entry itself."""
        book = FakeLogbook()
        first = _send(book)
        book.gone.add(first.entry_id)
        result = _send(book, entry_id=first.entry_id)
        assert result.entry_id != first.entry_id
        assert result.appended is False
        assert len(book.created) == 2


class TestSendRefusals:
    """What comes back when the send cannot happen."""

    def test_an_empty_image_is_rejected_before_any_call(self) -> None:
        book = FakeLogbook()
        with pytest.raises(ValueError, match="empty"):
            _send(book, png=b"")
        assert not book.created

    def test_an_oversized_image_is_rejected_before_any_call(self) -> None:
        book = FakeLogbook()
        big = b"\x89PNG" + b"0" * logbook_send.MAX_PLOT_BYTES
        with pytest.raises(ValueError, match="send cap"):
            _send(book, png=big)
        assert not book.created

    def test_a_refusal_carries_the_logbook_own_words(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(415, json={"detail": "unsupported type 'text/csv'"})

        with httpx.Client(transport=httpx.MockTransport(refuse)) as client:
            with pytest.raises(logbook_send.LogbookRefused) as caught:
                logbook_send.send_plot(
                    base_url=_BASE,
                    day="2026-07-12",
                    scan=2,
                    author="Ada",
                    png=_PNG,
                    caption="c",
                    client=client,
                )
        assert caught.value.status == 415
        assert "text/csv" in caught.value.detail

    def test_a_dead_logbook_is_unreachable_not_refused(self) -> None:
        def dead(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with httpx.Client(transport=httpx.MockTransport(dead)) as client:
            with pytest.raises(logbook_send.LogbookUnreachable):
                logbook_send.send_plot(
                    base_url=_BASE,
                    day="2026-07-12",
                    scan=2,
                    author="Ada",
                    png=_PNG,
                    caption="c",
                    client=client,
                )


class _StubSend:
    """Stands in for the conversation, recording what the route asked for."""

    def __init__(self, result=None, error: Optional[Exception] = None) -> None:
        self.result = result or logbook_send.SendResult("e1", False, "attachments/x")
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _client(**over) -> TestClient:
    kwargs = {
        "default_experiment": "Undulator",
        "logbook_url": "http://logbook.example:8400",
    }
    kwargs.update(over)
    return TestClient(create_app(FakeCatalog(), **kwargs))


def _post(client: TestClient, **over) -> httpx.Response:
    body = {"author": "Ada", "image": _DATA_URL, "caption": "y vs shot"}
    body.update(over)
    return client.post(f"/api/run/{_UID}/logbook", json=body)


class TestRouteGate:
    """Who may send, and what the page is told."""

    def test_a_send_reaches_the_logbook_and_returns_the_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = _StubSend()
        monkeypatch.setattr(logbook_send, "send_plot", stub)
        response = _post(_client())
        assert response.status_code == 201
        assert response.json() == {
            "entry_id": "e1",
            "appended": False,
            "url": "http://logbook.example:8400/entry/e1",
        }
        call = stub.calls[0]
        assert call["day"] == "2026-07-12"
        assert call["scan"] == 2
        assert call["png"] == _PNG
        assert call["filename"] == "scan002-plot.png"

    def test_no_logbook_url_means_no_send_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = _StubSend()
        monkeypatch.setattr(logbook_send, "send_plot", stub)
        assert _post(_client(logbook_url="")).status_code == 404
        assert not stub.calls

    def test_a_path_shaped_url_links_but_cannot_send(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``/log`` is the browser's front door and names no host to dial."""
        stub = _StubSend()
        monkeypatch.setattr(logbook_send, "send_plot", stub)
        client = _client(logbook_url="/log")
        assert _post(client).status_code == 404
        assert not stub.calls
        # ...and the link is still there, so this is a send gate, not a link one.
        assert '"/log/day/2026-07-12#Scan002"' in client.get(f"/run/{_UID}").text

    def test_a_run_from_another_experiment_cannot_send(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scan numbers restart per experiment: the entry would be another scan's."""
        stub = _StubSend()
        monkeypatch.setattr(logbook_send, "send_plot", stub)
        catalog = FakeCatalog()
        detail = catalog.details[_UID]
        catalog.details[_UID] = dataclasses.replace(
            detail, summary=dataclasses.replace(detail.summary, experiment="Thomson")
        )
        client = TestClient(
            create_app(
                catalog,
                default_experiment="Undulator",
                logbook_url="http://logbook.example:8400",
            )
        )
        assert _post(client).status_code == 404
        assert not stub.calls

    def test_the_page_advertises_the_button_only_when_it_works(self) -> None:
        """The modebar entry and its dialog appear on the gate, not always."""
        on = _client().get(f"/run/{_UID}").text
        off = _client(logbook_url="").get(f"/run/{_UID}").text
        assert "const LOGBOOK_SEND = true;" in on
        assert 'id="modal-sendlog"' in on
        assert "const LOGBOOK_SEND = false;" in off
        assert 'id="modal-sendlog"' not in off
        assert _client().get(f"/api/run/{_UID}").json()["logbook_send"] is True
        assert (
            _client(logbook_url="").get(f"/api/run/{_UID}").json()["logbook_send"]
            is False
        )


class TestRouteErrors:
    """Every failure lands as itself, never as a 500."""

    def test_a_malformed_image_is_the_caller_error(self) -> None:
        response = _post(_client(), image="not-a-data-url")
        assert response.status_code == 400
        assert "data:image/png" in response.json()["detail"]

    def test_a_nameless_send_is_rejected_by_validation(self) -> None:
        assert _post(_client(), author="").status_code == 422

    def test_an_unreachable_logbook_is_503(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            logbook_send,
            "send_plot",
            _StubSend(error=logbook_send.LogbookUnreachable("no answer")),
        )
        assert _post(_client()).status_code == 503

    def test_a_refusal_about_our_payload_keeps_its_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            logbook_send,
            "send_plot",
            _StubSend(error=logbook_send.LogbookRefused(413, "over the cap")),
        )
        response = _post(_client())
        assert response.status_code == 413
        assert response.json()["detail"] == "over the cap"

    def test_another_service_trouble_is_a_bad_gateway(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A logbook 500 is not the portal's 500 — the caller must see whose."""
        monkeypatch.setattr(
            logbook_send,
            "send_plot",
            _StubSend(error=logbook_send.LogbookRefused(500, "store is wedged")),
        )
        assert _post(_client()).status_code == 502


class TestReadOnlyDoctrineUnchanged:
    """Sending writes to the logbook's service — never to the scans tree."""

    def test_the_send_touches_no_scan_folder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        stub = _StubSend()
        monkeypatch.setattr(logbook_send, "send_plot", stub)
        before = sorted(p.name for p in tmp_path.iterdir())
        assert _post(_client()).status_code == 201
        assert sorted(p.name for p in tmp_path.iterdir()) == before
        # The conversation is entirely URL-addressed: no filesystem path
        # is ever handed to it.
        assert "folder" not in stub.calls[0]


class TestPeerTroubleIsNotOurs:
    """A logbook that answers oddly must not read as a bad request."""

    def test_a_success_that_is_not_json_is_a_bad_gateway(self) -> None:
        """A wrong port or a proxy maintenance page answers 200 text/html."""

        def html(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>maintenance</html>")

        with httpx.Client(transport=httpx.MockTransport(html)) as client:
            with pytest.raises(logbook_send.LogbookRefused) as caught:
                logbook_send.send_plot(
                    base_url=_BASE,
                    day="2026-07-12",
                    scan=2,
                    author="Ada",
                    png=_PNG,
                    caption="c",
                    client=client,
                )
        assert caught.value.status == 502

    def test_a_reply_missing_a_key_is_a_bad_gateway(self) -> None:
        """Version skew between two services on separate release cadences."""

        def terse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json={"id": "e1"})  # no entry_id

        with httpx.Client(transport=httpx.MockTransport(terse)) as client:
            with pytest.raises(logbook_send.LogbookRefused) as caught:
                logbook_send.send_plot(
                    base_url=_BASE,
                    day="2026-07-12",
                    scan=2,
                    author="Ada",
                    png=_PNG,
                    caption="c",
                    client=client,
                )
        assert caught.value.status == 502
        assert "entry_id" in caught.value.detail


class TestRouteRejectsOurOwnMalformedRequests:
    """What the peer would refuse is refused here, as ours."""

    def test_a_blank_author_is_a_bad_request_not_a_bad_gateway(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stripped to nothing, it is the logbook's 422 — but our mistake."""
        stub = _StubSend()
        monkeypatch.setattr(logbook_send, "send_plot", stub)
        assert _post(_client(), author="   ").status_code == 422
        assert not stub.calls

    def test_an_entry_id_cannot_carry_a_path_segment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It becomes a path segment in the URLs the sender builds."""
        stub = _StubSend()
        monkeypatch.setattr(logbook_send, "send_plot", stub)
        assert _post(_client(), entry="../../api/entries").status_code == 422
        assert _post(_client(), entry="e1/attachments").status_code == 422
        assert not stub.calls

    def test_a_non_json_peer_reply_does_not_read_as_a_bad_image(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The decode and the conversation no longer share one ValueError arm."""
        monkeypatch.setattr(
            logbook_send,
            "send_plot",
            _StubSend(error=logbook_send.LogbookRefused(502, "not JSON")),
        )
        assert _post(_client()).status_code == 502


class TestTheSendTouchesNoShare:
    """A logbook write has no business stat-ing the scans mount."""

    def test_no_scan_folder_is_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``resolve_scan_folder`` does is_dir() probes over SMB."""
        monkeypatch.setattr(logbook_send, "send_plot", _StubSend())
        import geecs_portal.app as app_module

        calls: list = []
        real = app_module.resolve_scan_folder

        def spy(*args, **kwargs):
            calls.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(app_module, "resolve_scan_folder", spy)
        client = _client()
        assert _post(client).status_code == 201
        assert calls == []
        # ...and the guard is meaningful: the run PAGE does resolve one.
        client.get(f"/run/{_UID}")
        assert calls
