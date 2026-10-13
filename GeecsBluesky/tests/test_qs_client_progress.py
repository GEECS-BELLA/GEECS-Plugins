"""The shared progress reduction every queue client renders (no zmq, no threads)."""

from __future__ import annotations

from geecs_bluesky.qs_client import FAILED_MOVE_LOG_PREFIX
from geecs_bluesky.qs_client.progress import ProgressStream, RunProgress


def test_row_streams_are_the_sfile_writers() -> None:
    from geecs_bluesky.callbacks import ROW_STREAMS as ENGINE_ROWS
    from geecs_bluesky.qs_client.progress import ROW_STREAMS

    assert tuple(ROW_STREAMS) == tuple(ENGINE_ROWS)


def _gated_run(stream: ProgressStream) -> None:
    """A gated run: datum-only ``primary`` per step, rows on ``shots`` as pages."""
    stream.on_document(
        "start",
        {
            "uid": "run-g",
            "scan_number": 7,
            "plan_name": "sweep",
            "num_points": 3,
            "shots_per_step": 4,
            "scan_folder": "/data/Scan007",
            "scan_tag": {"year": 2026, "month": 10, "day": 13},
        },
    )
    stream.on_document("descriptor", {"uid": "p", "name": "primary"})
    stream.on_document("descriptor", {"uid": "s", "name": "shots"})
    stream.on_document("descriptor", {"uid": "b", "name": "baseline"})
    stream.on_document("event_page", {"descriptor": "s", "seq_num": [1, 2, 3, 4]})
    stream.on_document("event", {"descriptor": "p", "seq_num": 1})
    stream.on_document("event_page", {"descriptor": "s", "seq_num": [5, 6, 7, 8]})
    stream.on_document("event", {"descriptor": "p", "seq_num": 2})
    stream.on_document("event", {"descriptor": "s", "seq_num": 9})
    stream.on_document("event", {"descriptor": "b", "seq_num": 99})


def test_a_gated_run_counts_its_shots_rows() -> None:
    stream = ProgressStream(clock=lambda: 3.0)
    _gated_run(stream)
    snap = stream.snapshot()
    assert snap.shots_done == 9  # primary's per-step seq (2) never wins
    assert snap.planned_total == 12
    assert snap.day == "2026-10-13" and snap.scan_folder == "/data/Scan007"
    assert snap.run_uid == "run-g" and snap.state == "running"
    stream.on_document("stop", {"run_start": "run-g", "exit_status": "success"})
    assert stream.snapshot().state == "done"


def test_a_failed_move_pauses_until_the_next_row() -> None:
    stream = ProgressStream()
    _gated_run(stream)
    stream.on_console_line(
        f"{FAILED_MOVE_LOG_PREFIX}: U_Hexapod did not reach 1.0", FAILED_MOVE_LOG_PREFIX
    )
    snap = stream.snapshot()
    assert (
        snap.state == "paused" and snap.paused_reason == "U_Hexapod did not reach 1.0"
    )
    stream.on_document("event", {"descriptor": "s", "seq_num": 10})
    snap = stream.snapshot()
    assert snap.state == "running" and snap.paused_reason is None


def test_a_stop_for_another_run_is_ignored() -> None:
    stream = ProgressStream()
    _gated_run(stream)
    stream.on_document("stop", {"run_start": "other", "exit_status": "fail"})
    assert stream.snapshot().exit_status is None


def test_a_malformed_document_is_skipped() -> None:
    stream = ProgressStream()
    stream.on_document("start", {"uid": "r", "num_points": "eleven"})
    stream.on_document("event", {"descriptor": None, "seq_num": object()})
    assert stream.snapshot().planned_total is None


def test_the_snapshot_is_a_copy() -> None:
    stream = ProgressStream()
    snap = stream.snapshot()
    snap.shots_done = 5
    assert stream.snapshot().shots_done == 0
    assert isinstance(snap, RunProgress)
