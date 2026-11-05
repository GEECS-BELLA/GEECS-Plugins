"""The analysis worker: jobs over a day's scans through the real task queue."""

from __future__ import annotations

import threading
import time
from datetime import date
from pathlib import Path

import pytest
from geecs_data_utils import ScanPaths, ScanTag

from scan_analysis.base import DataUnavailableWarning
from scan_analysis.task_queue import read_statuses
from scan_analysis.worker import JobSpec, Worker

DAY = date(2026, 3, 10)


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
    """A day's scan folders: closed ones carry their ScanData table."""
    for n in (*closed, *open_):
        tag = ScanTag(year=2026, month=3, day=10, number=n, experiment="Test")
        folder = ScanPaths.get_scan_folder_path(tag=tag, base_directory=base)
        folder.mkdir(parents=True)  # fixture acquisition
        if n in closed:
            (folder / f"ScanDataScan{n:03d}.txt").write_text("Shotnumber\n1\n")


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
        # redo runs done and failed again; no_data stays a fact about the scan
        assert sorted((t.scan, t.analyzer) for t in redo.tasks) == [
            (1, "A"),
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


class TestApi:
    @pytest.fixture()
    def client(self, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from scan_analysis.worker.app import create_worker_app

        build_day(tmp_path)
        groups = tmp_path / "configs" / "groups" / "PW"
        groups.mkdir(parents=True)
        (tmp_path / "configs" / "analyzers").mkdir()
        (groups / "empty.yaml").write_text("name: empty\nanalyzers: []\n")
        worker = make_worker(tmp_path, [FakeAnalyzer("A")])
        with TestClient(create_worker_app(worker)) as c:
            yield c

    def test_health_groups_plan_submit(self, client):
        assert client.get("/health").json()["ok"] is True
        groups = client.get("/api/groups").json()["groups"]
        assert [g["id"] for g in groups] == ["PW/empty"]
        body = {"day": "2026-03-10", "group": "PW/empty"}
        plan = client.post("/api/plan", json=body).json()
        assert [t["scan"] for t in plan["tasks"]] == [1, 2]
        assert plan["open_scans"] == [3]
        job = client.post("/api/jobs", json=body)
        assert job.status_code == 202
        job_id = job.json()["id"]
        for _ in range(200):
            got = client.get(f"/api/jobs/{job_id}").json()
            if got["state"] == "done":
                break
            time.sleep(0.02)
        assert got["counts"] == {"done": 2}
        assert client.get("/api/jobs").json()["jobs"][0]["id"] == job_id

    def test_errors(self, client):
        bad = {"day": "2026-03-10", "group": "nope"}
        assert client.post("/api/plan", json=bad).status_code == 404
        assert client.post("/api/jobs", json=bad).status_code == 404
        missing = {"day": "2026-03-11", "group": "PW/empty"}
        assert client.post("/api/plan", json=missing).status_code == 404
        assert client.post("/api/jobs", json={"group": "x"}).status_code == 422
        assert client.get("/api/jobs/j999").status_code == 404
