"""Tests for tiled_integration's SafeDocumentCallback run-scoped disabling.

Pins the fix for the review finding that a single TiledWriter exception
permanently disabled Tiled persistence on the session's long-lived RunEngine:
a failure must only drop the remainder of the current run, and the next
``start`` document must re-enable the callback and be forwarded itself.
"""

from __future__ import annotations

import logging

from geecs_bluesky.tiled_integration import SafeDocumentCallback


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
