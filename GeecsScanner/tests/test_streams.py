"""The document reduction: totals from the start doc, shots from primary events, exits."""

from __future__ import annotations

from geecs_scanner.service.streams import ProgressCache


def test_start_seeds_total_and_never_inherits_the_previous_run() -> None:
    c = ProgressCache(clock=lambda: 5.0)
    c.on_document(
        "start",
        {
            "scan_number": 12,
            "plan_name": "scan",
            "num_points": 11,
            "shots_per_step": 10,
        },
    )
    assert c.snapshot().planned_total == 110 and c.snapshot().scan_number == 12
    c.on_document("descriptor", {"uid": "d1", "name": "primary"})
    c.on_document("descriptor", {"uid": "d2", "name": "baseline"})
    c.on_document("event", {"descriptor": "d1", "seq_num": 7})
    c.on_document("event", {"descriptor": "d2", "seq_num": 99})  # not primary: ignored
    assert c.snapshot().shots_done == 7
    # an adaptive plan: max_iterations stands in for num_points
    c.on_document(
        "start",
        {
            "scan_number": 13,
            "plan_name": "optimize",
            "max_iterations": 25,
            "shots_per_step": 1,
        },
    )
    s = c.snapshot()
    assert s.planned_total == 25 and s.shots_done == 0 and s.state == "running"
    # unknown totals reset the picture rather than keeping 25
    c.on_document("start", {"scan_number": 14, "plan_name": "x"})
    assert c.snapshot().planned_total is None


def test_gated_rows_on_the_shots_stream_count_too() -> None:
    c = ProgressCache()
    c.on_document(
        "start",
        {"scan_number": 5, "plan_name": "scan", "num_points": 3, "shots_per_step": 4},
    )
    c.on_document("descriptor", {"uid": "p", "name": "primary"})
    c.on_document("descriptor", {"uid": "s", "name": "shots"})
    c.on_document("event", {"descriptor": "s", "seq_num": 9})  # a gated run's rows
    assert c.snapshot().shots_done == 9


def test_row_streams_are_the_sfile_writers() -> None:
    from geecs_bluesky.callbacks import ROW_STREAMS as ENGINE_ROWS

    from geecs_scanner.service.streams import ROW_STREAMS

    assert tuple(ROW_STREAMS) == tuple(ENGINE_ROWS)


def test_count_total_is_its_num() -> None:
    c = ProgressCache()
    c.on_document(
        "start",
        {"scan_number": 1, "plan_name": "count", "num_points": 50, "shots_per_step": 1},
    )
    assert c.snapshot().planned_total == 50


def test_stop_records_exit_and_console_prefix_sets_the_paused_reason() -> None:
    c = ProgressCache(clock=lambda: 1.0)
    c.on_document(
        "start",
        {"scan_number": 3, "plan_name": "scan", "num_points": 2, "shots_per_step": 2},
    )
    c.push_console_line(
        "FAILED MOVE - pausing for operator: U_Hexapod did not reach 1.0",
        "FAILED MOVE - pausing for operator",
    )
    s = c.snapshot()
    assert s.state == "paused" and s.paused_reason == "U_Hexapod did not reach 1.0"
    # the next row proves the resume: the reason and the word go
    c.on_document("descriptor", {"uid": "d", "name": "primary"})
    c.on_document("event", {"descriptor": "d", "seq_num": 1})
    s = c.snapshot()
    assert s.state == "running" and s.paused_reason is None
    c.on_document("stop", {"exit_status": "abort", "reason": "halted"})
    s = c.snapshot()
    assert s.state == "aborted" and s.exit_status == "abort"
    lines = c.console_since(0)
    assert [ln.seq for ln in lines] == [1] and c.console_since(1) == []


def test_version_changes_with_every_update() -> None:
    c = ProgressCache()
    v0 = c.version()
    c.push_console_line("hello")
    assert c.version() > v0
    c.mark_available(True, "")
    assert c.snapshot().available is True
