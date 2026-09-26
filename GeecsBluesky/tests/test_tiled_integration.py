"""Tests for tiled_integration: SafeDocumentCallback + the reachability pre-check.

Pins two review findings:

* a single writer exception permanently disabled Tiled persistence on the
  session's long-lived RunEngine — a failure must only drop the remainder
  of the current run, and the next ``start`` document must re-enable the
  callback and be forwarded itself;
* ``from_uri`` was called unconditionally, so building a session off the
  lab network hung for the Tiled client's full HTTP connect timeout — the
  bounded TCP pre-check (now the writer service's per-sweep check) must
  answer promptly.

The spool subscription itself is covered in ``test_tiled_spool.py``.
"""

from __future__ import annotations

import logging
import socket
import time

import pytest

from geecs_bluesky.tiled_integration import (
    SafeDocumentCallback,
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
# Reachability pre-check (an off-network sweep must not block)
# ---------------------------------------------------------------------------


def _dead_port() -> int:
    """A localhost port with no listener (bind-then-close): connect refuses."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_tiled_server_reachable_answers_promptly_for_a_dead_port() -> None:
    """No listener → False, well under the Tiled client's HTTP timeout."""
    uri = f"http://127.0.0.1:{_dead_port()}"
    started = time.monotonic()
    assert tiled_server_reachable(uri) is False
    elapsed = time.monotonic() - started
    assert elapsed < 3.0, f"pre-check took {elapsed:.1f}s — it hung"


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


def test_read_tiled_config_is_the_canonical_data_utils_reader() -> None:
    """Issue #527: [tiled] config parsing has ONE definition (geecs_data_utils).

    This package re-exports it rather than parsing the file itself, so a
    future config-semantics change (a second URI, a profiles file) cannot
    drift between packages.
    """
    from geecs_data_utils.tiled_catalog import read_tiled_config as canonical

    from geecs_bluesky import tiled_integration

    assert tiled_integration.read_tiled_config is canonical
