"""Unit tests for the ``scan_progress`` stream picture (no zmq, no threads).

The document handler is exercised directly — feeding the same start /
descriptor / event / stop shapes the worker's proxy emits — because the
stream plumbing itself (RemoteDispatcher, ConsoleMonitor) is upstream
code this package only wires up.
"""

from __future__ import annotations

from geecs_bluesky.qs_client import FAILED_MOVE_LOG_PREFIX
from geecs_bluesky.qs_client.progress import ProgressStream

from geecs_mcp.scans.progress_stream import stream_view


class ProgressCache(ProgressStream):
    """The shared stream, read through the tool contract's view."""

    def snapshot(self):  # type: ignore[override]
        return stream_view(super().snapshot())


def _started_cache() -> ProgressCache:
    cache = ProgressCache()
    cache.mark_available(True, "")  # what _run_documents sets once the dispatcher is up
    return cache


def _fail_move(cache: ProgressCache, reason: str) -> None:
    cache.on_console_line(f"{FAILED_MOVE_LOG_PREFIX}: {reason}", FAILED_MOVE_LOG_PREFIX)


START = {
    "uid": "run-1",
    "scan_number": 42,
    "num_points": 11,
    "shots_per_step": 5,
}


def test_start_doc_seeds_totals_and_resets():
    cache = _started_cache()
    cache.on_document("start", START)
    snap = cache.snapshot()
    assert snap["scan_number"] == 42
    assert snap["shots_total"] == 55
    assert snap["shots_done"] == 0
    assert snap["exit_status"] is None


def test_optimize_falls_back_to_max_iterations():
    cache = _started_cache()
    cache.on_document(
        "start", {"uid": "run-2", "max_iterations": 20, "shots_per_step": 3}
    )
    assert cache.snapshot()["shots_total"] == 60


def test_totals_none_when_neither_field_present():
    cache = _started_cache()
    cache.on_document("start", {"uid": "run-3"})
    assert cache.snapshot()["shots_total"] is None


def test_only_primary_stream_events_advance_shots():
    cache = _started_cache()
    cache.on_document("start", START)
    cache.on_document("descriptor", {"uid": "d-prim", "name": "primary"})
    cache.on_document("descriptor", {"uid": "d-base", "name": "baseline"})
    cache.on_document("event", {"descriptor": "d-prim", "seq_num": 7})
    cache.on_document("event", {"descriptor": "d-base", "seq_num": 999})
    assert cache.snapshot()["shots_done"] == 7


def test_stop_doc_records_exit_status_for_matching_run_only():
    cache = _started_cache()
    cache.on_document("start", START)
    cache.on_document("stop", {"run_start": "someone-else", "exit_status": "failed"})
    assert cache.snapshot()["exit_status"] is None
    cache.on_document("stop", {"run_start": "run-1", "exit_status": "success"})
    assert cache.snapshot()["exit_status"] == "success"


def test_new_start_clears_previous_run():
    cache = _started_cache()
    cache.on_document("start", START)
    cache.on_document("descriptor", {"uid": "d-prim", "name": "primary"})
    cache.on_document("event", {"descriptor": "d-prim", "seq_num": 55})
    _fail_move(cache, "U_Hexapod move failed")
    cache.on_document("start", {"uid": "run-2", "num_points": 3, "shots_per_step": 1})
    snap = cache.snapshot()
    assert snap["shots_done"] == 0 and snap["shots_total"] == 3
    assert snap["paused_reason"] is None
    # The old primary descriptor no longer counts.
    cache.on_document("event", {"descriptor": "d-prim", "seq_num": 9})
    assert cache.snapshot()["shots_done"] == 0


def test_primary_progress_clears_the_paused_reason():
    # Review finding #683-1: the failed-move reason must not survive a
    # successful resume — otherwise a SECOND (manual) pause of the same
    # run reports the first pause's text as the current why.  Progress
    # on the primary stream proves the resume.
    cache = _started_cache()
    cache.on_document("start", START)
    cache.on_document("descriptor", {"uid": "d-prim", "name": "primary"})
    _fail_move(cache, "U_Hexapod move failed")
    cache.on_document("event", {"descriptor": "d-prim", "seq_num": 8})
    assert cache.snapshot()["paused_reason"] is None


def test_ensure_started_ignores_a_different_address_after_latch():
    cache = ProgressCache()
    cache.ensure_started(None, None)  # latches unconfigured
    cache.ensure_started("otherhost:5568", None)  # ignored with a warning
    assert cache.snapshot()["available"] is False


def test_malformed_documents_never_raise():
    cache = _started_cache()
    cache.on_document("start", {"uid": "run-1", "num_points": "eleven"})
    cache.on_document("event", {"descriptor": None, "seq_num": object()})
    snap = cache.snapshot()  # still answers
    assert snap["available"] is True


def test_unstarted_cache_reports_unavailable():
    snap = ProgressCache().snapshot()
    assert snap["available"] is False and "not started" in snap["detail"]


def test_ensure_started_without_doc_addr_degrades_honestly(caplog):
    import logging

    cache = ProgressCache()
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.qs_client.progress"):
        cache.ensure_started(None, "tcp://localhost:60625")
    snap = cache.snapshot()
    assert snap["available"] is False
    assert "no document-stream address" in snap["detail"]
    # The cache itself says why (the startup warm-up and the lazy path
    # share this one log line — #748 review finding 2).
    assert "no document-stream address" in caplog.text
    # Idempotent: a second call must not spawn anything or reset state.
    cache.ensure_started(None, None)
    assert cache.snapshot()["detail"] == snap["detail"]


def test_start_for_client_reads_the_client_addresses(monkeypatch):
    # THE one address resolution shared by scan_progress and the HTTP
    # startup warm-up (#685): both go through start_for_client.
    from types import SimpleNamespace

    from geecs_mcp.scans import progress_stream

    seen = []

    class _Recorder:
        def ensure_started(self, doc_addr, info_addr):
            seen.append((doc_addr, info_addr))

    recorder = _Recorder()
    monkeypatch.setattr(progress_stream, "get_progress_cache", lambda: recorder)
    client = SimpleNamespace(
        doc_addr="localhost:5568", info_addr="tcp://localhost:60625"
    )
    assert progress_stream.start_for_client(client) is recorder
    # A stub client without the attributes resolves to None, not an error.
    progress_stream.start_for_client(object())
    assert seen == [("localhost:5568", "tcp://localhost:60625"), (None, None)]


def test_a_gated_runs_shots_rows_advance_shots_done():
    # A gated run's per-shot rows are the sampler's ``shots`` stream, sent as
    # event pages, while ``primary`` carries one datum event per step.  The
    # old primary-only reducer reported 2 here (the step count), not 8.
    cache = _started_cache()
    cache.on_document("start", {**START, "num_points": 2, "shots_per_step": 4})
    cache.on_document("descriptor", {"uid": "d-prim", "name": "primary"})
    cache.on_document("descriptor", {"uid": "d-shots", "name": "shots"})
    cache.on_document("event_page", {"descriptor": "d-shots", "seq_num": [1, 2, 3, 4]})
    cache.on_document("event", {"descriptor": "d-prim", "seq_num": 1})
    cache.on_document("event_page", {"descriptor": "d-shots", "seq_num": [5, 6, 7, 8]})
    cache.on_document("event", {"descriptor": "d-prim", "seq_num": 2})
    snap = cache.snapshot()
    assert snap["shots_done"] == 8 and snap["shots_total"] == 8


def test_the_tool_contract_keys():
    assert set(_started_cache().snapshot()) == {"available", "detail"}
    cache = _started_cache()
    cache.on_document("start", START)
    assert list(cache.snapshot()) == [
        "available",
        "detail",
        "run_uid",
        "scan_number",
        "shots_total",
        "shots_done",
        "exit_status",
        "paused_reason",
    ]


def test_a_stop_caught_mid_run_adds_no_run_keys():
    # Started consuming mid-run (stdio's lazy start, a service restart):
    # the stop alone must not read as a finished run of zero shots.
    cache = _started_cache()
    cache.on_document("stop", {"run_start": "run-0", "exit_status": "success"})
    assert cache.snapshot() == {"available": True, "detail": ""}
    _fail_move(cache, "U_Hexapod move failed")
    assert cache.snapshot() == {
        "available": True,
        "detail": "",
        "paused_reason": "U_Hexapod move failed",
    }
