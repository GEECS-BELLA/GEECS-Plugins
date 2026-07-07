"""Tests for tiled_integration: SafeDocumentCallback + reachability pre-check.

Pins two review findings:

* a single TiledWriter exception permanently disabled Tiled persistence on
  the session's long-lived RunEngine — a failure must only drop the remainder
  of the current run, and the next ``start`` document must re-enable the
  callback and be forwarded itself;
* ``subscribe_tiled`` called ``from_uri`` unconditionally, so building a
  session off the lab network hung for the Tiled client's full HTTP connect
  timeout — a bounded TCP pre-check must skip the subscription promptly.
"""

from __future__ import annotations

import logging
import socket
import time

import pytest

from geecs_bluesky.tiled_integration import (
    SafeDocumentCallback,
    subscribe_tiled,
    tiled_server_reachable,
)


def test_safe_document_callback_reenables_on_next_run_start(caplog) -> None:
    """A failure disables only the current run; the next start re-enables."""
    delivered: list[tuple[str, str | None]] = []

    def callback(name: str, doc: dict) -> None:
        if doc.get("boom"):
            raise RuntimeError("tiled connection dropped")
        delivered.append((name, doc.get("uid")))

    safe_callback = SafeDocumentCallback(callback, label="TiledWriter")

    with caplog.at_level(logging.WARNING):
        # Run 1: fails mid-run — subsequent run-1 documents are skipped.
        safe_callback("start", {"uid": "run-1"})
        safe_callback("descriptor", {"uid": "desc-1"})
        safe_callback("event", {"uid": "ev-1", "boom": True})
        safe_callback("event", {"uid": "ev-2"})
        safe_callback("stop", {"uid": "stop-1"})

        # Run 2: the start document re-enables and is itself forwarded.
        safe_callback("start", {"uid": "run-2"})
        safe_callback("descriptor", {"uid": "desc-2"})
        safe_callback("event", {"uid": "ev-3"})
        safe_callback("stop", {"uid": "stop-2"})

    assert delivered == [
        ("start", "run-1"),
        ("descriptor", "desc-1"),
        ("start", "run-2"),
        ("descriptor", "desc-2"),
        ("event", "ev-3"),
        ("stop", "stop-2"),
    ]

    # Loud on disable (which run, which document) ...
    assert (
        "TiledWriter failed while handling event document during run run-1"
        in caplog.text
    )
    # ... and loud on re-enable at the next run start.
    assert "TiledWriter re-enabled at start of run run-2" in caplog.text
    assert "failure during run run-1" in caplog.text


def test_safe_document_callback_failure_on_start_recovers_next_run(caplog) -> None:
    """A failure while handling start itself still recovers at the next start."""
    delivered: list[str] = []
    fail_next = {"flag": True}

    def callback(name: str, doc: dict) -> None:
        if name == "start" and fail_next["flag"]:
            fail_next["flag"] = False
            raise RuntimeError("server unreachable")
        delivered.append(str(doc.get("uid")))

    safe_callback = SafeDocumentCallback(callback, label="TiledWriter")

    with caplog.at_level(logging.WARNING):
        safe_callback("start", {"uid": "run-1"})  # raises internally
        safe_callback("event", {"uid": "ev-1"})  # skipped
        safe_callback("stop", {"uid": "stop-1"})  # skipped
        safe_callback("start", {"uid": "run-2"})  # re-enabled + forwarded
        safe_callback("stop", {"uid": "stop-2"})

    assert delivered == ["run-2", "stop-2"]
    assert "TiledWriter re-enabled at start of run run-2" in caplog.text


# ---------------------------------------------------------------------------
# Reachability pre-check (off-network construction must not block)
# ---------------------------------------------------------------------------


class _Engine:
    """RunEngine stand-in recording subscriptions (never a real RE needed)."""

    def __init__(self) -> None:
        self.subscribed: list = []

    def subscribe(self, callback) -> int:
        self.subscribed.append(callback)
        return len(self.subscribed)


def _dead_port() -> int:
    """A localhost port with no listener (bind-then-close): connect refuses."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_subscribe_tiled_unreachable_server_skips_promptly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No listener → warning + no subscription, well under the old HTTP timeout."""
    engine = _Engine()
    uri = f"http://127.0.0.1:{_dead_port()}"
    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        token = subscribe_tiled(engine, uri)
    elapsed = time.monotonic() - started

    assert token is None
    assert engine.subscribed == []
    assert elapsed < 3.0, f"pre-check took {elapsed:.1f}s — it hung"
    assert f"Tiled server {uri} unreachable" in caplog.text
    assert "Tiled persistence disabled" in caplog.text


def test_subscribe_tiled_reachable_server_still_subscribes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reachable server takes the pre-existing path: client built, writer on."""
    pytest.importorskip("tiled.client")

    class _FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    seen: dict = {}
    monkeypatch.setattr(
        "socket.create_connection",
        lambda address, timeout=None: seen.update(address=address, timeout=timeout)
        or _FakeConnection(),
    )
    monkeypatch.setattr(
        "tiled.client.from_uri",
        lambda uri, api_key=None: seen.update(uri=uri, api_key=api_key) or object(),
    )
    monkeypatch.setattr(
        "bluesky.callbacks.tiled_writer.TiledWriter",
        lambda client, patches=None: lambda name, doc: None,
    )

    engine = _Engine()
    token = subscribe_tiled(engine, "http://192.0.2.1:8000", api_key="secret")

    assert token == 1
    assert len(engine.subscribed) == 1
    assert isinstance(engine.subscribed[0], SafeDocumentCallback)
    assert seen["address"] == ("192.0.2.1", 8000)
    assert seen["uri"] == "http://192.0.2.1:8000"
    assert seen["api_key"] == "secret"


def test_tiled_server_reachable_parses_default_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheme-default ports (http→80, https→443) back a portless URI."""
    attempts: list = []

    def _refuse(address, timeout=None):
        attempts.append((address, timeout))
        raise ConnectionRefusedError

    monkeypatch.setattr("socket.create_connection", _refuse)

    assert tiled_server_reachable("http://tiled.lab:8000") is False
    assert tiled_server_reachable("http://tiled.lab") is False
    assert tiled_server_reachable("https://tiled.lab") is False
    assert [address for address, _timeout in attempts] == [
        ("tiled.lab", 8000),
        ("tiled.lab", 80),
        ("tiled.lab", 443),
    ]
    # Every attempt is bounded by the module constant, not the HTTP timeout.
    from geecs_bluesky.tiled_integration import TILED_REACHABILITY_TIMEOUT_S

    assert all(timeout == TILED_REACHABILITY_TIMEOUT_S for _addr, timeout in attempts)


def test_tiled_server_reachable_unparseable_uri_defers_to_client() -> None:
    """A hostless URI returns True so from_uri reports the real parse error."""
    assert tiled_server_reachable("not-a-uri") is True
