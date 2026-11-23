"""The day page's batch routes against the analysis worker's own app, in process.

The worker is built the way its own suite builds it (a real ``Worker`` over a
temp day, a fake analyzer through the real task queue) and its FastAPI app is
reached through ``WorkerLink`` on a ``TestClient``, so a route, body or reply
field change in GeecsAnalysisWorker fails this suite in the same PR.
geecs-analysis-worker is a dev-only dependency: ``geecs_portal`` never imports
it (the ``portal-edges`` contract).

The bodies are the ones ``day.html`` sends (``batchBody()`` and the watch's
start), and the reply fields asserted are the ones it reads.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from geecs_data_utils import ScanPaths, ScanTag
from geecs_data_utils.data.sfile import sfile_path_for_scan
from scan_analysis.worker import Worker

from geecs_analysis_worker.app import create_worker_app
from geecs_portal.batch import WorkerLink
from test_app import FakeCatalog, _client

URL = "http://worker.test"
EXPERIMENT = "Test"
DAY = "2026-03-10"
GROUP = "PW/empty"

#: Every field day.html reads from a job row, and from a job's task.
JOB_FIELDS = {
    "id", "day", "group", "scans", "redo", "analyzers", "from_scan", "origin",
    "state", "error", "total", "counts", "current", "cancel_requested",
}  # fmt: skip
TASK_FIELDS = {"scan", "analyzer", "state", "error"}
#: Every field day.html reads from the watch.
WATCH_FIELDS = {"day", "group", "analyzers", "from_scan", "last_check", "jobs", "note"}  # fmt: skip


class FakeAnalyzer:
    """The ScanAnalyzer contract the task queue calls."""

    def __init__(self, id: str, *, fail_on=()) -> None:
        self.id = id
        self.priority = 100
        self.fail_on = set(fail_on)

    def run_analysis(self, tag):
        if tag.number in self.fail_on:
            raise RuntimeError("boom")
        return []

    def cleanup(self):
        pass


def _scan(base: Path, number: int, *, closed: bool) -> None:
    """One scan folder; a closed one carries its table and a settled s-file."""
    tag = ScanTag(year=2026, month=3, day=10, number=number, experiment=EXPERIMENT)
    folder = ScanPaths.get_scan_folder_path(tag=tag, base_directory=base)
    folder.mkdir(parents=True)  # fixture acquisition
    if not closed:
        return
    (folder / f"ScanDataScan{number:03d}.txt").write_text("Shotnumber\n1\n")
    sfile = sfile_path_for_scan(folder)
    sfile.parent.mkdir(parents=True, exist_ok=True)  # fixture acquisition
    sfile.write_text("Shotnumber\n1\n")
    then = time.time() - 60.0
    os.utime(sfile, (then, then))


@pytest.fixture
def portal(tmp_path: Path):
    """The portal's test client, its WorkerLink dialling the worker's real app."""
    for number, closed in ((1, True), (2, True), (3, False)):
        _scan(tmp_path, number, closed=closed)
    groups = tmp_path / "configs" / "groups" / "PW"
    groups.mkdir(parents=True)
    (tmp_path / "configs" / "analyzers").mkdir()
    (groups / "empty.yaml").write_text("name: empty\nanalyzers: []\n")
    worker = Worker(
        tmp_path / "configs",
        default_experiment="Elsewhere",  # the portal's experiment must reach it
        base_directory=tmp_path,
        loader=lambda group, config_dir: [FakeAnalyzer("A", fail_on={2})],
    )
    with TestClient(create_worker_app(worker), base_url=URL) as worker_client:
        link = WorkerLink(URL, client=worker_client)
        yield _client(FakeCatalog(), default_experiment=EXPERIMENT, worker=link)


def _finished(portal, job_id: str) -> dict:
    for _ in range(250):
        job = portal.get(f"/api/batch/jobs/{job_id}").json()
        if job["state"] not in ("queued", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


def test_status_says_the_worker_answers(portal) -> None:
    assert portal.get("/api/batch/status").json() == {
        "configured": True,
        "reachable": True,
    }


def test_groups_carry_the_id_and_members(portal) -> None:
    (group,) = portal.get("/api/batch/groups").json()["groups"]
    assert group["id"] == GROUP
    assert group["members"] == []


def test_the_plan_counts_what_a_job_would_run(portal) -> None:
    body = {"day": DAY, "group": GROUP, "scans": None, "redo": False, "analyzers": None}  # fmt: skip
    plan = portal.post("/api/batch/plan", json=body)
    assert plan.status_code == 200, plan.text
    plan = plan.json()
    assert plan["total"] == 2
    assert [t["scan"] for t in plan["tasks"]] == [1, 2]
    assert plan["open_scans"] == [3]
    assert plan["already_done"] == 0


def test_a_job_runs_and_its_rows_carry_what_the_page_reads(portal) -> None:
    body = {"day": DAY, "group": GROUP, "scans": [1, 2], "redo": False, "analyzers": ["A"]}  # fmt: skip
    queued = portal.post("/api/batch/jobs", json=body)
    assert queued.status_code == 202, queued.text
    job_id = queued.json()["id"]
    detail = _finished(portal, job_id)
    assert detail["state"] == "done"
    assert detail["counts"] == {"done": 1, "failed": 1}
    assert detail["scans"] == [1, 2] and detail["analyzers"] == ["A"]
    failed = [t for t in detail["tasks"] if t["state"] == "failed"]
    assert len(failed) == 1 and TASK_FIELDS <= failed[0].keys()
    assert failed[0]["scan"] == 2 and failed[0]["error"]
    (row,) = portal.get("/api/batch/jobs").json()["jobs"]
    assert JOB_FIELDS <= row.keys()
    assert row["id"] == job_id and row["total"] == 2
    cancelled = portal.post(f"/api/batch/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["id"] == job_id


def test_the_portal_names_its_experiment(portal) -> None:
    """The worker's own default is another experiment: the day is found only through the portal's."""
    body = {"day": DAY, "group": GROUP, "scans": None, "redo": False, "analyzers": None}  # fmt: skip
    job = portal.post("/api/batch/jobs", json=body).json()
    assert _finished(portal, job["id"])["state"] == "done"


def test_an_unknown_group_and_job_are_the_workers_404(portal) -> None:
    body = {"day": DAY, "group": "nope", "scans": None, "redo": False, "analyzers": None}  # fmt: skip
    assert portal.post("/api/batch/jobs", json=body).status_code == 404
    assert portal.get("/api/batch/jobs/j999").status_code == 404
    assert portal.post("/api/batch/jobs/j999/cancel").status_code == 404


def test_a_watch_starts_reads_and_stops(portal) -> None:
    assert portal.get("/api/batch/watch").json() == {"watch": None}
    body = {"day": DAY, "group": GROUP, "analyzers": ["A"], "from_scan": 2}
    started = portal.post("/api/batch/watch", json=body)
    assert started.status_code == 200, started.text
    watch = started.json()["watch"]
    assert WATCH_FIELDS <= watch.keys()
    assert watch["day"] == DAY and watch["group"] == GROUP
    assert watch["analyzers"] == ["A"] and watch["from_scan"] == 2
    assert portal.get("/api/batch/watch").json()["watch"]["group"] == GROUP
    assert portal.post("/api/batch/watch/stop").json() == {"watch": None}
