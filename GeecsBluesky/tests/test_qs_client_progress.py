"""The shared progress reduction every queue client renders (no zmq, no threads)."""

from __future__ import annotations

import math

from geecs_bluesky.qs_client import FAILED_MOVE_LOG_PREFIX
from geecs_bluesky.qs_client.progress import ProgressStream, RunProgress


def test_row_streams_are_the_sfile_writers() -> None:
    from geecs_bluesky.callbacks import ROW_STREAMS as ENGINE_ROWS
    from geecs_bluesky.qs_client.progress import ROW_STREAMS

    assert tuple(ROW_STREAMS) == tuple(ENGINE_ROWS)


def _strict_run(stream: ProgressStream) -> None:
    """A strict run: rows on ``primary``, a device stream's pages beside them."""
    stream.on_document(
        "start",
        {
            "uid": "run-s",
            "scan_number": 7,
            "plan_name": "sweep",
            "num_points": 3,
            "shots_per_step": 4,
            "scan_folder": "/data/Scan007",
            "scan_tag": {"year": 2026, "month": 10, "day": 13},
        },
    )
    stream.on_document("descriptor", {"uid": "p", "name": "primary"})
    stream.on_document("descriptor", {"uid": "c", "name": "uc_cam_stream"})
    stream.on_document("descriptor", {"uid": "b", "name": "baseline"})
    for seq in range(1, 10):
        stream.on_document("event", {"descriptor": "p", "seq_num": seq})
        stream.on_document("event_page", {"descriptor": "c", "seq_num": [seq + 20]})
    stream.on_document("event", {"descriptor": "b", "seq_num": 99})


def test_a_strict_run_counts_its_primary_rows() -> None:
    stream = ProgressStream(clock=lambda: 3.0)
    _strict_run(stream)
    snap = stream.snapshot()
    assert snap.shots_done == 9  # a device stream's seq (29) never wins
    assert snap.planned_total == 12
    assert snap.day == "2026-10-13" and snap.scan_folder == "/data/Scan007"
    assert snap.run_uid == "run-s" and snap.state == "running"
    stream.on_document("stop", {"run_start": "run-s", "exit_status": "success"})
    assert stream.snapshot().state == "done"


def test_a_failed_move_pauses_until_the_next_row() -> None:
    stream = ProgressStream()
    _strict_run(stream)
    stream.on_console_line(
        f"{FAILED_MOVE_LOG_PREFIX}: U_Hexapod did not reach 1.0", FAILED_MOVE_LOG_PREFIX
    )
    snap = stream.snapshot()
    assert (
        snap.state == "paused" and snap.paused_reason == "U_Hexapod did not reach 1.0"
    )
    stream.on_document("event", {"descriptor": "p", "seq_num": 10})
    snap = stream.snapshot()
    assert snap.state == "running" and snap.paused_reason is None


def test_a_stop_for_another_run_is_ignored() -> None:
    stream = ProgressStream()
    _strict_run(stream)
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


def _rows(stream: ProgressStream, rows: list[dict]) -> None:
    stream.on_document("start", {"uid": "r", "plan_name": "count", "num_points": 9})
    stream.on_document("descriptor", {"uid": "p", "name": "primary"})
    for seq, data in enumerate(rows, start=1):
        stream.on_document("event", {"descriptor": "p", "seq_num": seq, "data": data})


def _row(fire: float, device: str = "", delay: float = math.nan, **extra) -> dict:
    return {
        "fire_stamp": fire,
        "bin_number": extra.pop("bin_number", 1),
        "slowest_device": device,
        "slowest_s": delay,
        **extra,
    }


def test_a_device_is_named_only_when_it_held_the_rate() -> None:
    """Its delay past the period: it held the shot; within it, the trigger edge's wait."""
    stream = ProgressStream(clock=lambda: 3.0)
    _rows(
        stream,
        [
            _row(100.0, "CAM-Wedge", 1.8),
            _row(102.0, "CAM-Wedge", 2.3),  # period 2.0, delayed 2.3: held it
            _row(104.0, "CAM-Ghost", 2.6),  # period 2.0, delayed 2.6: held it
        ],
    )
    mid = stream.snapshot()
    assert mid.current_rate_hz == 0.5
    assert (mid.waiting_on, mid.waiting_s) == ("CAM-Ghost", 2.6)
    assert mid.worker_s == 0.0  # a period shorter than the previous wait (2.3)
    # an event page (columns of lists) of the fourth row: period 1.0, 0.7 s
    # after the fire is the edge's wait, not the device
    stream.on_document(
        "event_page",
        {
            "descriptor": "p",
            "seq_num": [4],
            "data": {
                "fire_stamp": [105.0],
                "bin_number": [1],
                "slowest_device": ["CAM-Wedge"],
                "slowest_s": [0.7],
            },
        },
    )
    snap = stream.snapshot()
    assert snap.current_rate_hz == 1.0
    assert snap.waiting_on is None and snap.waiting_s is None
    assert snap.worker_s == 0.0  # a period shorter than the previous wait
    assert [(d.device, d.held) for d in snap.slowest] == [
        ("CAM-Wedge", 1),
        ("CAM-Ghost", 1),
    ]
    # a new run starts clean
    stream.on_document("start", {"uid": "r2"})
    fresh = stream.snapshot()
    assert fresh.current_rate_hz is None and fresh.slowest == []


def test_no_period_spans_a_step_a_pause_or_an_interrupted_row() -> None:
    """Periods within a bin only, less the gate hold; a pause or an interrupted row breaks them."""
    stream = ProgressStream(clock=lambda: 3.0)
    _rows(
        stream,
        [
            _row(100.0),
            _row(101.5, gate_hold_s=0.5),  # 1.5 s, 0.5 of it the gate: 1.0
            _row(110.0, bin_number=2),  # a step move: no period
            _row(111.0, bin_number=2, after_pause=True),  # across a pause
            _row(112.0, "CAM-Wedge", 5.0, bin_number=2, interrupted=True),
            _row(113.0, bin_number=2),  # after an interrupted row
        ],
    )
    snap = stream.snapshot()
    assert snap.current_rate_hz == 1.0  # still the first bin's period
    assert snap.slowest == []  # the interrupted row's arrivals say nothing
    stream.on_document(
        "event",
        {"descriptor": "p", "seq_num": 7, "data": _row(114.25, bin_number=2)},
    )
    assert stream.snapshot().current_rate_hz == 0.8


def test_a_run_without_timing_columns_has_no_rates() -> None:
    stream = ProgressStream(clock=lambda: 3.0)
    _strict_run(stream)
    snap = stream.snapshot()
    assert snap.current_rate_hz is None
    assert snap.waiting_on is None and snap.slowest == []
