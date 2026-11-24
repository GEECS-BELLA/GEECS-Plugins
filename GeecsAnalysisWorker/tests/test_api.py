"""The analysis worker's HTTP API and its command."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from geecs_data_utils import ScanPaths, ScanTag
from geecs_data_utils.testing import write_closed
from scan_analysis.base import DataUnavailableWarning
from scan_analysis.worker import Worker

from geecs_analysis_worker import __main__ as command
from geecs_analysis_worker.app import create_worker_app


# The fixtures below (FakeAnalyzer, build_day, make_worker) are a copy of
# ScanAnalysis/tests/test_worker.py's (write_closed is shared, from
# geecs_data_utils.testing): a change to what a closed scan is
# (geecs_data_utils.data.sfile.scan_is_settled) changes both.


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


class TestApi:
    """The routes over a real ``Worker`` (fake analyzers through the real queue)."""

    @pytest.fixture()
    def client(self, tmp_path):
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

    def test_watch_routes(self, client):
        assert client.get("/api/watch").json() == {"watch": None}
        body = {"day": "2026-03-10", "group": "PW/empty", "analyzers": ["A"]}
        watch = client.post("/api/watch", json=body).json()["watch"]
        assert watch["group"] == "PW/empty" and watch["analyzers"] == ["A"]
        assert client.get("/api/watch").json()["watch"]["day"] == "2026-03-10"
        assert client.post("/api/watch/stop").json() == {"watch": None}
        assert (
            client.post("/api/watch", json={**body, "group": "nope"}).status_code == 404
        )

    def test_no_watch_until_one_is_asked_for(self, client):
        """A served worker is idle: a job runs and still nothing watches."""
        body = {"day": "2026-03-10", "group": "PW/empty"}
        assert client.post("/api/jobs", json=body).status_code == 202
        assert client.get("/api/watch").json() == {"watch": None}


class TestCommand:
    """``scan-analysis-worker``: what it binds, and that it starts no watch."""

    @pytest.fixture()
    def served(self, monkeypatch, tmp_path):
        """Run ``main`` with uvicorn stubbed; what it was asked to serve."""
        import uvicorn

        seen: dict = {}

        def run(app, **kwargs):
            seen.update(kwargs, app=app)

        monkeypatch.setattr(uvicorn, "run", run)
        monkeypatch.delenv("GEECS_ANALYSIS_WORKER_BIND", raising=False)
        monkeypatch.delenv("GEECS_ANALYSIS_WORKER_PORT", raising=False)
        (tmp_path / "configs").mkdir()

        def serve(*extra: str) -> dict:
            command.main(
                ["--configs", str(tmp_path / "configs"), "--experiment", "Test", *extra]
            )
            return seen

        return serve

    def test_defaults_are_loopback_and_8900(self, served):
        seen = served()
        assert (seen["host"], seen["port"]) == ("127.0.0.1", 8900)

    def test_the_environment_names_the_bind_and_port(self, served, monkeypatch):
        monkeypatch.setenv("GEECS_ANALYSIS_WORKER_BIND", "0.0.0.0")
        monkeypatch.setenv("GEECS_ANALYSIS_WORKER_PORT", "8911")
        seen = served()
        assert (seen["host"], seen["port"]) == ("0.0.0.0", 8911)

    def test_a_flag_wins_over_the_environment(self, served, monkeypatch):
        monkeypatch.setenv("GEECS_ANALYSIS_WORKER_PORT", "8911")
        assert served("--port", "8912")["port"] == 8912

    def test_an_empty_environment_value_is_the_default(self, served, monkeypatch):
        monkeypatch.setenv("GEECS_ANALYSIS_WORKER_BIND", "")
        monkeypatch.setenv("GEECS_ANALYSIS_WORKER_PORT", "")
        seen = served()
        assert (seen["host"], seen["port"]) == ("127.0.0.1", 8900)

    def test_the_command_has_no_watch_flag(self, served, capsys):
        """A watch is an operator's request over HTTP, never a start-up option."""
        with pytest.raises(SystemExit):
            served("--watch", "2026-03-10")
        assert "unrecognized arguments" in capsys.readouterr().err

    def test_the_served_worker_starts_with_no_watch(self, served):
        with TestClient(served()["app"]) as client:
            assert client.get("/api/watch").json() == {"watch": None}

    def test_a_missing_configs_folder_is_refused(self, served, tmp_path):
        with pytest.raises(SystemExit):
            command.main(
                ["--configs", str(tmp_path / "absent"), "--experiment", "Test"]
            )

    def test_a_malformed_port_in_the_environment_is_a_usage_error(
        self, served, monkeypatch, capsys
    ):
        monkeypatch.setenv("GEECS_ANALYSIS_WORKER_PORT", "abc")
        with pytest.raises(SystemExit) as stopped:
            served()
        assert stopped.value.code == 2
        assert "invalid int value: 'abc'" in capsys.readouterr().err


class TestShutdown:
    """What a stop does to the running task."""

    def gated(self, tmp_path):
        import threading

        build_day(tmp_path, closed=(1,), open_=())
        groups = tmp_path / "configs" / "groups" / "PW"
        groups.mkdir(parents=True)
        (tmp_path / "configs" / "analyzers").mkdir()
        (groups / "empty.yaml").write_text("name: empty\nanalyzers: []\n")
        gate = threading.Event()
        analyzer = FakeAnalyzer("A", gate=gate)
        return gate, analyzer, make_worker(tmp_path, [analyzer])

    def submit(self, client, analyzer):
        body = {"day": "2026-03-10", "group": "PW/empty"}
        job_id = client.post("/api/jobs", json=body).json()["id"]
        for _ in range(200):
            if client.get(f"/api/jobs/{job_id}").json()["state"] == "running":
                return
            time.sleep(0.02)
        raise AssertionError("the job did not start")

    def test_a_stop_waits_for_the_running_task(self, tmp_path, caplog):
        import threading

        caplog.set_level("INFO", logger="geecs_analysis_worker.app")

        gate, analyzer, worker = self.gated(tmp_path)
        with TestClient(create_worker_app(worker, stop_seconds=10.0)) as client:
            self.submit(client, analyzer)
            threading.Timer(1.5, gate.set).start()
            began = time.monotonic()
        assert 1.4 < time.monotonic() - began < 6.0
        assert analyzer.ran == [1]
        assert not worker._thread.is_alive()
        assert "waiting up to 10 s for the running task" in caplog.text

    def test_a_stop_gives_up_on_a_task_longer_than_its_wait(self, tmp_path):
        gate, analyzer, worker = self.gated(tmp_path)
        try:
            with TestClient(create_worker_app(worker, stop_seconds=0.2)) as client:
                self.submit(client, analyzer)
                began = time.monotonic()
            assert time.monotonic() - began < 0.9
            assert analyzer.ran == []
        finally:
            gate.set()

    def test_the_default_wait_sits_under_the_units_stop_timeout(self):
        from geecs_analysis_worker.app import STOP_SECONDS

        unit = Path(__file__).parent.parent / "deploy/geecs-analysis-worker.service"
        (line,) = [
            x for x in unit.read_text().splitlines() if x.startswith("TimeoutStopSec=")
        ]
        assert STOP_SECONDS < float(line.split("=")[1])
