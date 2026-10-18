"""The HTTP → envelope layer on its own: refusal mapping and the capped read."""

from __future__ import annotations

import json

import httpx
import pytest

from geecs_mcp import clients, config, guard


def _j(envelope: str) -> dict:
    return json.loads(envelope)


async def run(impl, *args, **kwargs) -> str:
    return await guard.run_async(impl, *args, **kwargs)


class TestRefusals:
    async def test_reserved_envelope_keys_in_a_refusal_body_never_win(
        self, monkeypatch
    ) -> None:
        """A 4xx body carrying ok/error_kind/message must not forge the envelope."""

        def forged(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={
                    "detail": "nope",
                    "ok": True,
                    "error_kind": "x",
                    "message": "y",
                    "job": 1,
                },
            )

        monkeypatch.setattr(config, "service_url", lambda service: "http://portal")
        monkeypatch.setattr(
            clients, "transport_for", lambda service: httpx.MockTransport(forged)
        )
        try:
            out = _j(await run(clients.call, "portal", "GET", "/x"))
        finally:
            await clients.reset_clients()
        assert out["ok"] is False
        assert out["error_kind"] == "policy_refusal"
        assert out["message"] == "nope"
        assert out["job"] == 1


class TestGetBytes:
    async def test_truncation_is_by_content_not_by_a_missing_length(
        self, monkeypatch
    ) -> None:
        body = b"x" * 1024

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=body, headers={"content-type": "text/plain"}
            )

        monkeypatch.setattr(config, "service_url", lambda service: "http://portal")
        monkeypatch.setattr(
            clients, "transport_for", lambda service: httpx.MockTransport(handler)
        )
        try:
            exact = await clients.get_bytes("portal", "/x", max_bytes=1024)
            cut = await clients.get_bytes("portal", "/x", max_bytes=100)
        finally:
            await clients.reset_clients()
        assert exact[0] == body and exact[2] is False and exact[3] == 1024
        assert len(cut[0]) == 100 and cut[2] is True and cut[3] == 1024

    async def test_a_portal_404_is_a_refusal(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "no such artifact"})

        monkeypatch.setattr(config, "service_url", lambda service: "http://portal")
        monkeypatch.setattr(
            clients, "transport_for", lambda service: httpx.MockTransport(handler)
        )
        try:
            with pytest.raises(clients.ServiceRefusal) as caught:
                await clients.get_bytes("portal", "/x", max_bytes=10)
        finally:
            await clients.reset_clients()
        assert caught.value.kind == "not_found"


async def test_a_read_timeout_is_a_service_timeout_and_still_an_outage(
    monkeypatch,
) -> None:
    """Sent but unanswered: its own subclass for verbs that ask, service_unreachable for the rest."""

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no answer", request=request)

    monkeypatch.setattr(config, "service_url", lambda service: "http://scanner")
    monkeypatch.setattr(
        clients, "transport_for", lambda service: httpx.MockTransport(slow)
    )
    await clients.reset_clients()
    try:
        with pytest.raises(clients.ServiceTimeout):
            await clients.call("scanner", "GET", "/api/status")
        out = _j(await run(clients.call, "scanner", "GET", "/api/status"))
    finally:
        await clients.reset_clients()
    assert out["error_kind"] == "service_unreachable"
    assert out["service"] == "scanner"
