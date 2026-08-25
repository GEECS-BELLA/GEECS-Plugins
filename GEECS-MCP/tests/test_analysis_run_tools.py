"""Hermetic tests for the analysis-domain execution tools (#686).

Skips whole without the ``analysis-run`` extra (ScanAnalysis) — the same
pattern as GeecsBluesky's optional-stack suites.  A tmp data share stands
in for the netapp via the ``read_tools._base_directory`` seam, a tmp
configs tree via the ``run_tools._config_root`` seam, and the worker
spawn is captured through the ``run_tools._spawn_worker`` seam — no real
subprocess, no real analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("scan_analysis")

from geecs_mcp import runtime  # noqa: E402
from geecs_mcp.analysis import read_tools, run_tools  # noqa: E402

_BEAM = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"

DAY = "2026-08-22"


@pytest.fixture(autouse=True)
def _fresh_runtime(monkeypatch):
    runtime.clear_runtime_cache()
    monkeypatch.setattr(runtime, "get_experiment", lambda: "TestExp")
    run_tools._dispatched.clear()
    yield
    runtime.clear_runtime_cache()
    run_tools._dispatched.clear()


def _load(payload) -> dict:
    return json.loads(payload)


def _scan_folder(base: Path) -> Path:
    return base / "TestExp" / "Y2026" / "08-Aug" / "26_0822" / "scans" / "Scan007"


@pytest.fixture
def share(tmp_path, monkeypatch):
    """A fake data share holding one existing scan folder (Scan007)."""
    _scan_folder(tmp_path).mkdir(parents=True)
    monkeypatch.setattr(read_tools, "_base_directory", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def configs(tmp_path, monkeypatch):
    """A tmp scan-analysis configs tree: two diagnostics + one group."""
    root = tmp_path / "configs"
    diag_dir = root / "analyzers" / "HTU"
    diag_dir.mkdir(parents=True)
    (diag_dir / "test_diag.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "test_diag",
                "image_analyzer": _BEAM,
                "image": {"type": "camera", "bit_depth": 16},
                "scan": {"priority": 5},
            }
        )
    )
    (diag_dir / "windows_only.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "windows_only",
                # Stands in for a Windows-SDK class: unimportable here.
                "image_analyzer": "no_such_sdk.wavekit.HasoAnalyzer",
                "image": {"type": "camera", "bit_depth": 16},
                "scan": {},
            }
        )
    )
    group_dir = root / "groups" / "HTU"
    group_dir.mkdir(parents=True)
    (group_dir / "baseline.yaml").write_text(
        yaml.safe_dump({"name": "baseline", "analyzers": ["test_diag"]})
    )
    monkeypatch.setattr(run_tools, "_config_root", lambda: root)
    return root


@pytest.fixture
def spawned(monkeypatch):
    """Capture the worker spawn instead of launching a subprocess."""
    calls: list[dict] = []

    def fake_spawn(payload: dict) -> int:
        calls.append(payload)
        return 4242

    monkeypatch.setattr(run_tools, "_spawn_worker", fake_spawn)
    return calls


def _tree_snapshot(base: Path) -> set[str]:
    return {str(p.relative_to(base)) for p in base.rglob("*")}


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_requires_exactly_one_selector(self, share, configs, spawned):
        neither = _load(
            run_tools._run_scan_analysis_impl(7, DAY, None, None, False, False)
        )
        both = _load(
            run_tools._run_scan_analysis_impl(
                7, DAY, "test_diag", "baseline", False, False
            )
        )
        for result in (neither, both):
            assert not result["ok"]
            assert result["error_kind"] == "invalid_request"
            assert "exactly one" in result["message"]
        assert spawned == []

    def test_missing_scan_folder_refuses_and_creates_nothing(
        self, share, configs, spawned
    ):
        before = _tree_snapshot(share)
        result = _load(
            run_tools._run_scan_analysis_impl(99, DAY, "test_diag", None, False, False)
        )
        assert not result["ok"]
        assert result["error_kind"] == "not_found"
        assert "never" in result["message"]  # the invariant, stated to the agent
        assert _tree_snapshot(share) == before  # NOTHING created on the share
        assert spawned == []

    def test_unknown_analyzer_is_not_found(self, share, configs, spawned):
        result = _load(
            run_tools._run_scan_analysis_impl(
                7, DAY, "no_such_diag", None, False, False
            )
        )
        assert not result["ok"]
        assert result["error_kind"] == "not_found"
        assert spawned == []

    def test_unimportable_diagnostic_refused_before_enqueue(
        self, share, configs, spawned
    ):
        before = _tree_snapshot(share)
        result = _load(
            run_tools._run_scan_analysis_impl(
                7, DAY, "windows_only", None, False, False
            )
        )
        assert not result["ok"]
        assert result["error_kind"] == "invalid_request"
        assert "cannot run on this host" in result["message"]
        assert _tree_snapshot(share) == before
        assert spawned == []

    def test_unconfigured_root_refused(self, share, spawned, monkeypatch):
        monkeypatch.setattr(run_tools, "_config_root", lambda: None)
        result = _load(
            run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, False, False)
        )
        assert not result["ok"]
        assert "scan_analysis_configs_path" in result["message"]
        assert spawned == []


# ---------------------------------------------------------------------------
# the happy paths
# ---------------------------------------------------------------------------


class TestRun:
    def test_single_analyzer_enqueues_and_spawns(self, share, configs, spawned):
        result = _load(
            run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, False, False)
        )
        assert result["ok"] and result["started"]
        assert result["tasks"] == ["test_diag"]
        assert result["worker_pid"] == 4242
        # Server-side init: the queued status row is on disk already.
        status = yaml.safe_load(
            (_scan_folder(share) / "analysis_status" / "test_diag.yaml").read_text()
        )
        assert status["state"] == "queued"
        assert status["priority"] == 5
        # The worker payload carries everything it needs, verbatim.
        (payload,) = spawned
        assert payload["number"] == 7 and payload["experiment"] == "TestExp"
        assert payload["analyzer"] == "test_diag" and payload["group"] is None
        assert payload["base_directory"] == str(share)

    def test_group_enqueues_and_spawns(self, share, configs, spawned):
        result = _load(
            run_tools._run_scan_analysis_impl(7, DAY, None, "baseline", False, False)
        )
        assert result["ok"]
        assert result["tasks"] == ["test_diag"]
        (payload,) = spawned
        assert payload["group"] == "baseline" and payload["analyzer"] is None

    def test_rerun_flags_travel_to_the_worker(self, share, configs, spawned):
        run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, True, True)
        (payload,) = spawned
        assert payload["rerun_failed"] is True
        assert payload["rerun_completed"] is True


def _seed_status(share: Path, analyzer_id: str, **fields) -> Path:
    status_dir = _scan_folder(share) / "analysis_status"
    status_dir.mkdir(exist_ok=True)
    path = status_dir / f"{analyzer_id}.yaml"
    path.write_text(
        yaml.safe_dump({"analyzer_id": analyzer_id, "priority": 5, **fields})
    )
    return path


class TestExistingStatusRows:
    """The rerun/concurrency contract over pre-existing status rows."""

    def test_rerun_failed_resets_the_row_to_queued_before_spawn(
        self, share, configs, spawned
    ):
        """The dead-worker visibility contract holds on the rerun path:
        the failed row is re-queued server-side, so a worker that dies
        pre-claim leaves a visible queued row, not the stale failure."""
        path = _seed_status(
            share, "test_diag", state="failed", error="old failure text"
        )
        result = _load(
            run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, True, False)
        )
        assert result["ok"] and result["started"]
        assert result["tasks"] == ["test_diag"]
        status = yaml.safe_load(path.read_text())
        assert status["state"] == "queued"
        assert status["error"] is None
        assert len(spawned) == 1

    def test_done_without_rerun_flag_is_skipped_and_nothing_spawns(
        self, share, configs, spawned
    ):
        _seed_status(share, "test_diag", state="done")
        result = _load(
            run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, False, False)
        )
        assert result["ok"]
        assert result["started"] is False
        assert result["tasks"] == []
        assert result["skipped"] == {"test_diag": "done"}
        assert "rerun" in result["note"]
        assert spawned == []

    def test_actively_claimed_task_refuses_the_call(self, share, configs, spawned):
        """Two near-simultaneous calls must not double-run one task —
        and the refusal is side-effect-free: it happens before init/reset,
        so a rerun flag on the refused call must not re-queue anything."""
        from datetime import datetime, timezone

        path = _seed_status(
            share,
            "test_diag",
            state="claimed",
            claimed_by="another-runner",
            last_heartbeat=datetime.now(timezone.utc).isoformat(),
        )
        before_tree = _tree_snapshot(share)
        before_content = path.read_bytes()
        result = _load(
            run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, True, True)
        )
        assert not result["ok"]
        assert result["error_kind"] == "policy_refusal"
        assert "already running" in result["message"]
        assert _tree_snapshot(share) == before_tree
        assert path.read_bytes() == before_content  # not even a rewrite
        assert spawned == []

    def test_second_call_in_the_preclaim_window_refuses(
        self, share, configs, spawned, monkeypatch
    ):
        """Codex P1 (#690): a second run_scan_analysis call while the
        first worker is dispatched but has not claimed yet must not
        double-start — and the refusal is side-effect-free."""
        monkeypatch.setattr(run_tools, "_pid_alive", lambda pid: True)
        first = _load(
            run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, False, False)
        )
        assert first["ok"] and first["started"]
        before = _tree_snapshot(share)
        second = _load(
            run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, False, False)
        )
        assert not second["ok"]
        assert second["error_kind"] == "policy_refusal"
        assert "dispatched" in second["message"]
        assert _tree_snapshot(share) == before
        assert len(spawned) == 1  # exactly one worker

    def test_dead_dispatched_worker_allows_redispatch(
        self, share, configs, spawned, monkeypatch
    ):
        """A worker that died pre-claim (pid gone) must not block retries."""
        monkeypatch.setattr(run_tools, "_pid_alive", lambda pid: False)
        run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, False, False)
        second = _load(
            run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, False, False)
        )
        assert second["ok"] and second["started"]
        assert len(spawned) == 2

    def test_claimed_tasks_release_the_dispatch_ledger(
        self, share, configs, spawned, monkeypatch
    ):
        """Once the worker claims (rows leave queued), the ledger no longer
        gates — the heartbeat-based active-claim refusal takes over."""
        from datetime import datetime, timezone

        monkeypatch.setattr(run_tools, "_pid_alive", lambda pid: True)
        run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, False, False)
        # The worker claims: the status row leaves "queued".
        _seed_status(
            share,
            "test_diag",
            state="claimed",
            claimed_by="worker",
            last_heartbeat=datetime.now(timezone.utc).isoformat(),
        )
        second = _load(
            run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, False, False)
        )
        assert not second["ok"]
        assert second["error_kind"] == "policy_refusal"
        assert "already running" in second["message"]  # the claim path, not the ledger

    def test_stale_claim_is_runnable_again(self, share, configs, spawned):
        """A claim whose runner died (old heartbeat) does not block."""
        _seed_status(
            share,
            "test_diag",
            state="claimed",
            claimed_by="dead-runner",
            last_heartbeat="2026-08-22T00:00:00+00:00",
        )
        result = _load(
            run_tools._run_scan_analysis_impl(7, DAY, "test_diag", None, False, False)
        )
        assert result["ok"] and result["started"]
        assert result["tasks"] == ["test_diag"]
        assert len(spawned) == 1


# ---------------------------------------------------------------------------
# listings
# ---------------------------------------------------------------------------


class TestListings:
    def test_list_analyzers(self, configs):
        result = _load(run_tools._list_analyzers_impl())
        assert result["ok"]
        assert result["analyzers"] == ["test_diag", "windows_only"]
        assert result["count"] == 2
        assert result["truncated"] is False

    def test_list_groups_indexes_both_name_forms(self, configs):
        result = _load(run_tools._list_analysis_groups_impl())
        assert result["ok"]
        assert set(result["groups"]) == {"baseline", "HTU/baseline"}
        # count = distinct group FILES, not accepted names — one group
        # listed under two name forms must not read as two groups.
        assert result["count"] == 1

    def test_listings_refuse_without_root(self, monkeypatch):
        monkeypatch.setattr(run_tools, "_config_root", lambda: None)
        result = _load(run_tools._list_analyzers_impl())
        assert not result["ok"]
        assert "scan_analysis_configs_path" in result["message"]


# ---------------------------------------------------------------------------
# the worker
# ---------------------------------------------------------------------------


class TestWorker:
    def test_worker_builds_and_runs_the_worklist(self, share, configs, monkeypatch):
        """run_worker parses the payload and drives the real task queue
        (run_worklist itself is captured — no actual analysis)."""
        from scan_analysis import task_queue

        from geecs_mcp.analysis import run_worker

        ran: dict = {}

        def fake_run_worklist(worklist, *, base_directory=None, **kwargs):
            ran["worklist"] = worklist
            ran["base_directory"] = base_directory

        monkeypatch.setattr(task_queue, "run_worklist", fake_run_worklist)
        payload = {
            "year": 2026,
            "month": 8,
            "day": 22,
            "number": 7,
            "experiment": "TestExp",
            "analyzer": "test_diag",
            "group": None,
            "rerun_failed": False,
            "rerun_completed": False,
            "config_root": str(configs),
            "base_directory": str(share),
        }
        run_worker.main([json.dumps(payload)])
        assert ran["base_directory"] == Path(share)
        (item,) = ran["worklist"]
        priority, tag, analyzer = item
        assert priority == 5
        assert tag.number == 7 and tag.experiment == "TestExp"
        assert getattr(analyzer, "id", None) == "test_diag"
