"""The shared progress reduction every queue client renders (no zmq, no threads)."""

from __future__ import annotations

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


def test_the_rates_and_the_devices_that_held_the_shots() -> None:
    """Current = 1 / the last period, average over the run's fires; the slowest ranked by rows held."""
    stream = ProgressStream(clock=lambda: 3.0)
    stream.on_document("start", {"uid": "r", "plan_name": "count", "num_points": 4})
    stream.on_document("descriptor", {"uid": "p", "name": "primary"})
    rows = [
        (100.0, float("nan"), "CAM-Wedge", 1.8),
        (102.0, 2.0, "CAM-Wedge", 1.6),
        (104.0, 2.0, "CAM-Ghost", 1.9),
    ]
    for seq, (fire, period, device, delay) in enumerate(rows, start=1):
        stream.on_document(
            "event",
            {
                "descriptor": "p",
                "seq_num": seq,
                "data": {
                    "fire_stamp": fire,
                    "shot_period_s": period,
                    "slowest_device": device,
                    "slowest_s": delay,
                },
            },
        )
    # an event page (columns of lists) of the fourth row
    stream.on_document(
        "event_page",
        {
            "descriptor": "p",
            "seq_num": [4],
            "data": {
                "fire_stamp": [105.0],
                "shot_period_s": [1.0],
                "slowest_device": ["CAM-Wedge"],
                "slowest_s": [0.7],
            },
        },
    )
    snap = stream.snapshot()
    assert snap.current_rate_hz == 1.0
    assert snap.average_rate_hz == 3 / 5.0
    assert (snap.waiting_on, snap.waiting_s) == ("CAM-Wedge", 0.7)
    top = snap.slowest[0]
    assert (top.device, top.held, top.max_s) == ("CAM-Wedge", 3, 1.8)
    assert abs(top.mean_s - (1.8 + 1.6 + 0.7) / 3) < 1e-9
    assert [d.device for d in snap.slowest] == ["CAM-Wedge", "CAM-Ghost"]
    # a new run starts clean
    stream.on_document("start", {"uid": "r2"})
    fresh = stream.snapshot()
    assert fresh.current_rate_hz is None and fresh.slowest == []


def test_a_run_without_timing_columns_has_no_rates() -> None:
    stream = ProgressStream(clock=lambda: 3.0)
    _strict_run(stream)
    snap = stream.snapshot()
    assert snap.current_rate_hz is None and snap.average_rate_hz is None
    assert snap.waiting_on is None and snap.slowest == []
