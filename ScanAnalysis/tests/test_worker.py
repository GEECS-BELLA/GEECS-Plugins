"""The analysis worker: jobs over a day's scans through the real task queue."""

from __future__ import annotations

import threading
import time
from datetime import date
from pathlib import Path

from geecs_data_utils import ScanPaths, ScanTag
from geecs_data_utils.testing import write_closed

from scan_analysis.base import DataUnavailableWarning
from scan_analysis.task_queue import read_statuses
from scan_analysis.worker import JobSpec, WatchSpec, Worker

DAY = date(2026, 3, 10)


# GeecsAnalysisWorker/tests/test_api.py keeps a copy of the fixtures below
# (FakeAnalyzer, build_day, make_worker): change both. write_closed is
# geecs_data_utils.testing's.


class FakeAnalyzer:
    """Honours the ScanAnalyzer contract the task queue calls."""

    def __init__(self, id, *, priority=100, fail_on=(), no_data_on=(), gate=None):
        self.id = id
        self.priority = priority
        self.fail_on = set(fail_on)
        self.no_data_on = set(no_data_on)
        self.gate = gate
        self.ran: list[int] = []

    def run_analysis(self, tag):
        if self.gate is not None:
            self.gate.wait(5)
        self.ran.append(tag.number)
        if tag.number in self.fail_on:
            raise RuntimeError("boom")
        if tag.number in self.no_data_on:
            raise DataUnavailableWarning("no device folder")
        return []

    def cleanup(self):
        pass


def build_day(base: Path, closed=(1, 2), open_=(3,)) -> None:
    """A day's scan folders: closed ones carry their table and settled s-file."""
    for n in (*closed, *open_):
        tag = ScanTag(year=2026, month=3, day=10, number=n, experiment="Test")
        folder = ScanPaths.get_scan_folder_path(tag=tag, base_directory=base)
        folder.mkdir(parents=True)  # fixture acquisition
        if n in closed:
            write_closed(folder)


def make_worker(base: Path, analyzers: list) -> Worker:
    return Worker(
        base / "configs",
        default_experiment="Test",
        base_directory=base,
        loader=lambda group, config_dir: analyzers,
    )


def wait_finished(worker: Worker, job_id: str, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job = worker.get(job_id)
        if job.state not in ("queued", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def spec(**over) -> JobSpec:
    return JobSpec(**{"day": DAY, "group": "g", "experiment": "Test", **over})


def statuses(base: Path, n: int) -> dict[str, str]:
    tag = ScanTag(year=2026, month=3, day=10, number=n, experiment="Test")
    folder = ScanPaths.get_scan_folder_path(tag=tag, base_directory=base)
    return {s.analyzer_id: s.state for s in read_statuses(folder)}


def test_plan_lists_closed_scans_only_and_writes_nothing(tmp_path):
    build_day(tmp_path)
    worker = make_worker(tmp_path, [FakeAnalyzer("A"), FakeAnalyzer("B")])
    before = set(tmp_path.rglob("*"))
    plan = worker.plan(spec())
    assert sorted((t.scan, t.analyzer) for t in plan.tasks) == [
        (1, "A"),
        (1, "B"),
        (2, "A"),
        (2, "B"),
    ]
    assert plan.open_scans == [3]
    assert set(tmp_path.rglob("*")) == before


def test_a_job_runs_records_and_a_second_skips_what_finished(tmp_path):
    build_day(tmp_path)
    a = FakeAnalyzer("A", fail_on={2})
    b = FakeAnalyzer("B", no_data_on={1})
    worker = make_worker(tmp_path, [a, b])
    worker.start()
    try:
        job = wait_finished(worker, worker.submit(spec()).id)
        assert job.state == "done"
        assert job.counts() == {"done": 2, "failed": 1, "no_data": 1}
        failed = next(t for t in job.tasks if t.state == "failed")
        assert (failed.scan, failed.analyzer) == (2, "A")
        assert "boom" in failed.error
        assert statuses(tmp_path, 1) == {"A": "done", "B": "no_data"}
        assert statuses(tmp_path, 3) == {}  # still recording: never touched

        again = wait_finished(worker, worker.submit(spec()).id)
        assert again.tasks == []  # done, no_data and failed are all left alone
        assert again.already_done == 4

        redo = wait_finished(worker, worker.submit(spec(redo=True)).id)
        # redo runs every finished task again, no_data included (often a
        # typo in the job, which a redo after the fix must reach)
        assert sorted((t.scan, t.analyzer) for t in redo.tasks) == [
            (1, "A"),
            (1, "B"),
            (2, "A"),
            (2, "B"),
        ]
    finally:
        worker.stop(1)


def test_scans_limit_the_job(tmp_path):
    build_day(tmp_path, closed=(1, 2, 4), open_=())
    worker = make_worker(tmp_path, [FakeAnalyzer("A")])
    assert [t.scan for t in worker.plan(spec(scans=(2, 4))).tasks] == [2, 4]


def test_cancel_stops_before_the_next_task(tmp_path):
    build_day(tmp_path, closed=(1, 2, 4), open_=())
    gate = threading.Event()
    a = FakeAnalyzer("A", gate=gate)
    worker = make_worker(tmp_path, [a])
    worker.start()
    try:
        job_id = worker.submit(spec()).id
        end = time.monotonic() + 5
        while worker.get(job_id).state != "running" or not any(
            t.state == "running" for t in worker.get(job_id).tasks
        ):
            assert time.monotonic() < end
            time.sleep(0.01)
        worker.cancel(job_id)
        gate.set()
        job = wait_finished(worker, job_id)
        assert job.state == "cancelled"
        assert job.counts() == {"done": 1, "cancelled": 2}
        assert a.ran == [1]
    finally:
        worker.stop(1)


def test_a_missing_day_fails_the_job_not_the_worker(tmp_path):
    worker = make_worker(tmp_path, [FakeAnalyzer("A")])
    worker.start()
    try:
        job = wait_finished(worker, worker.submit(spec()).id)
        assert job.state == "failed"
        assert "no scans folder" in job.error
        assert not (tmp_path / "Test").exists()  # nothing created
    finally:
        worker.stop(1)


def test_a_job_runs_only_the_ticked_analyzers_from_its_start_scan(tmp_path):
    build_day(tmp_path, closed=(1, 2, 4), open_=())
    worker = make_worker(tmp_path, [FakeAnalyzer("A"), FakeAnalyzer("B")])
    plan = worker.plan(spec(analyzers=("B",), from_scan=2))
    assert sorted((t.scan, t.analyzer) for t in plan.tasks) == [(2, "B"), (4, "B")]


def wait_for(condition, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met")


def close_scan(base: Path, n: int) -> None:
    tag = ScanTag(year=2026, month=3, day=10, number=n, experiment="Test")
    folder = ScanPaths.get_scan_folder_path(tag=tag, base_directory=base)
    write_closed(folder)


def test_a_scan_waits_for_the_scanner_and_not_for_analysis(tmp_path):
    from geecs_data_utils.data.sfile import sfile_path_for_scan

    build_day(tmp_path, closed=(), open_=(1,))
    tag = ScanTag(year=2026, month=3, day=10, number=1, experiment="Test")
    folder = ScanPaths.get_scan_folder_path(tag=tag, base_directory=tmp_path)
    worker = make_worker(tmp_path, [FakeAnalyzer("A")])
    # the table just written: still recording
    write_closed(folder, age=0.0)
    assert worker.plan(spec()).open_scans == [1]
    # settled; then an analysis merge rewrites the s-file: still settled
    write_closed(folder)
    sfile_path_for_scan(folder).write_text("Shotnumber\tA x\n1\t2\n")
    plan = worker.plan(spec())
    assert plan.open_scans == []
    assert [(t.scan, t.analyzer) for t in plan.tasks] == [(1, "A")]


def test_redo_reaches_a_no_data_task(tmp_path):
    build_day(tmp_path, closed=(1,), open_=())
    a = FakeAnalyzer("A", no_data_on=(1,))
    worker = make_worker(tmp_path, [a])
    worker.start()
    try:
        wait_finished(worker, worker.submit(spec()).id)
        assert statuses(tmp_path, 1) == {"A": "no_data"}
        a.no_data_on.clear()  # the job's typo fixed
        wait_finished(worker, worker.submit(spec()).id)
        assert a.ran == [1]  # finished: not run again without redo
        wait_finished(worker, worker.submit(spec(redo=True)).id)
        assert a.ran == [1, 1]
        assert statuses(tmp_path, 1) == {"A": "done"}
    finally:
        worker.stop()


def test_a_watch_analyzes_each_scan_once_it_closes(tmp_path):
    build_day(tmp_path, closed=(1, 2), open_=(3,))
    a, b = FakeAnalyzer("A"), FakeAnalyzer("B")
    worker = make_worker(tmp_path, [a, b])
    worker.watch_poll_seconds = 0.05
    worker.start()
    try:
        worker.watch(
            WatchSpec(
                day=DAY, group="g", experiment="Test", analyzers=("A",), from_scan=2
            )
        )
        wait_for(lambda: a.ran == [2])
        assert statuses(tmp_path, 3) == {}  # still recording: left alone
        close_scan(tmp_path, 3)
        wait_for(lambda: a.ran == [2, 3])
        watch = worker.watching()
        assert all(worker.get(j).spec.origin == "watch" for j in watch.jobs)
        time.sleep(0.3)  # more looks: nothing new, so no more jobs
        assert a.ran == [2, 3] and b.ran == []  # scan 1 is before its start
        assert len(worker.watching().jobs) == 2
        assert worker.unwatch() is watch and worker.watching() is None
    finally:
        worker.stop(1)


def test_a_watch_waits_for_a_day_not_started_yet(tmp_path):
    worker = make_worker(tmp_path, [FakeAnalyzer("A")])
    worker.watch_poll_seconds = 0.05
    worker.start()
    try:
        worker.watch(WatchSpec(day=DAY, group="g", experiment="Test"))
        wait_for(lambda: (worker.watching().note or "").startswith("waiting for"))
        assert not (tmp_path / "Test").exists()  # nothing created
        assert worker.watching().jobs == []
    finally:
        worker.stop(1)


def test_a_worker_starts_with_no_watch(tmp_path):
    """Nothing watches until a caller asks: building, starting and running a job start none."""
    build_day(tmp_path)
    worker = make_worker(tmp_path, [FakeAnalyzer("A")])
    assert worker.watching() is None
    worker.start()
    try:
        assert worker.watching() is None
        wait_finished(worker, worker.submit(spec()).id)
        assert worker.watching() is None
        assert worker._watch_thread is None
    finally:
        worker.stop(1)
