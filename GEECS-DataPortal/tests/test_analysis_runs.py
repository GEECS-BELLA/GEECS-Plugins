"""Analysis runs: the 04-design run model over a fake analyzer.

No share, no hardware, no ScanAnalysis needed for the endpoint ladder
(the factory seam is injected); the one test that builds the REAL
factory skips without the ``analysis`` extra.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_portal import analysis_runs
from geecs_portal.app import create_app
from test_app import FakeCatalog, _detail

logger = logging.getLogger("fake.analyzer")


# ---------------------------------------------------------------- fixtures


@pytest.fixture()
def scan_folder(tmp_path) -> Path:
    """A Scan002 folder with a ``cam`` device dir (data files irrelevant here)."""
    # Dated the day AFTER test_app.TEST_DAY (the start doc's time) on
    # purpose: the run's tag must come from the folder, never the doc.
    folder = (
        tmp_path / "Undulator" / "Y2026" / "07-Jul" / "26_0713" / "scans" / "Scan002"
    )
    (folder / "cam").mkdir(parents=True)
    (folder / "cam" / "Scan002_cam_001.png").write_bytes(b"not-a-real-png")
    return folder


@pytest.fixture()
def configs_tree(tmp_path) -> Path:
    """Two loadable diagnostics (one per device) + one legacy flat YAML."""
    yaml = pytest.importorskip("yaml")
    tree = tmp_path / "proc_configs"
    analyzers = tree / "analyzers" / "HTU"
    analyzers.mkdir(parents=True)

    def diag(name: str, device: str) -> None:
        (analyzers / f"{name}.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": name,
                    "image_analyzer": (
                        "image_analysis.analyzers.standard_analyzer.StandardAnalyzer"
                    ),
                    "image": {"type": "camera", "bit_depth": 16},
                    "scan": {"priority": 100, "device": device},
                }
            )
        )

    diag("UC_Crop", "cam")
    diag("UC_Other", "elsewhere")
    (analyzers / "UC_Legacy.yaml").write_text(
        yaml.safe_dump({"name": "UC_Legacy", "bit_depth": 16})
    )
    return tree


class FakeAnalyzer:
    """The two-call contract, scripted per test."""

    instances: list = []

    def __init__(self, behaviour: str, analysis_folder: Path):
        self.behaviour = behaviour
        self.analysis_folder = analysis_folder
        self.cleaned = False
        self.release = threading.Event()
        self.started = threading.Event()
        FakeAnalyzer.instances.append(self)

    def run_analysis(self, scan_tag):
        self.scan_tag = scan_tag
        self.started.set()
        logger.info("fake analyzer running on scan %s", scan_tag.number)
        if self.behaviour == "block":
            self.release.wait(timeout=10)
        if self.behaviour == "fail":
            logger.warning("about to fail")
            raise RuntimeError("boom")
        if self.behaviour == "none":
            return None
        if self.behaviour == "no_data":

            class DataUnavailableWarning(Exception):
                pass

            raise DataUnavailableWarning("device folder empty")
        out = self.analysis_folder / "UC_Crop" / "Array2DScanAnalyzer"
        out.mkdir(parents=True, exist_ok=True)
        figure = out / "summary.png"
        figure.write_bytes(b"\x89PNG fake")
        return [figure, "a label, not a path"]

    def cleanup(self):
        self.cleaned = True


def _factory(behaviour: str):
    def factory(analyzer_id: str, config_dir: Path):
        assert analyzer_id == "UC_Crop"
        return FakeAnalyzer(behaviour, factory.analysis_folder)

    return factory


def _client(scan_folder, configs_tree, behaviour="ok", **kwargs) -> TestClient:
    catalog = FakeCatalog()
    detail = _detail(2)
    detail.start_doc["scan_folder"] = str(scan_folder)
    catalog.details["uid-002"] = detail
    factory = _factory(behaviour)
    factory.analysis_folder = analysis_runs.analysis_folder_for(scan_folder)
    kwargs.setdefault("analysis_factory", factory)
    return TestClient(create_app(catalog, processing_config_dir=configs_tree, **kwargs))


def _wait(client: TestClient, analyzer="UC_Crop", timeout=10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get("/api/run/uid-002/analysis").json()
        job = next(a["job"] for a in body["analyzers"] if a["id"] == analyzer)
        if job is not None and job["state"] not in analysis_runs.ACTIVE:
            return job
        time.sleep(0.02)
    raise AssertionError("job never finished")


@pytest.fixture(autouse=True)
def _reset_instances():
    FakeAnalyzer.instances = []
    yield
    for inst in FakeAnalyzer.instances:
        inst.release.set()


# ------------------------------------------------------------ pure helpers


class TestHelpers:
    def test_analysis_folder_is_the_scans_sibling(self, scan_folder):
        assert analysis_runs.analysis_folder_for(scan_folder) == (
            scan_folder.parents[1] / "analysis" / "Scan002"
        )

    def test_contained_artifact_refuses_escapes(self, tmp_path):
        root = tmp_path / "analysis" / "Scan002"
        (root / "UC_Crop").mkdir(parents=True)
        inside = root / "UC_Crop" / "fig.png"
        inside.write_bytes(b"x")
        outside = tmp_path / "analysis" / "s2.txt"
        outside.write_text("secret")
        assert analysis_runs.contained_artifact(root, "UC_Crop/fig.png") == inside
        assert analysis_runs.contained_artifact(root, "../s2.txt") is None
        assert analysis_runs.contained_artifact(root, str(outside)) is None
        assert analysis_runs.contained_artifact(root, "") is None
        assert analysis_runs.contained_artifact(root, "UC_Crop") is None  # a dir
        assert analysis_runs.contained_artifact(root, "UC_Crop/missing.png") is None
        link = root / "UC_Crop" / "link.txt"
        os.symlink(outside, link)
        assert analysis_runs.contained_artifact(root, "UC_Crop/link.txt") is None

    def test_relativize_only_under_root(self, tmp_path):
        root = tmp_path / "analysis" / "Scan002"
        (root / "x").mkdir(parents=True)
        assert analysis_runs._relativize(root / "x" / "f.png", root) == "x/f.png"
        assert analysis_runs._relativize(tmp_path / "elsewhere.png", root) == str(
            tmp_path / "elsewhere.png"
        )
        assert analysis_runs._relativize("just a label", root) == "just a label"
        assert analysis_runs._relativize(root / "x" / "f.png", None) == str(
            root / "x" / "f.png"
        )

    def test_list_artifacts_is_scoped_sorted_and_capped(self, tmp_path):
        root = tmp_path / "analysis" / "Scan002"
        out = root / "UC_Crop" / "Array2DScanAnalyzer"
        out.mkdir(parents=True)
        for name in ("b.png", "a.png", "c.h5"):
            (out / name).write_bytes(b"x")
        (root / "UC_Other").mkdir()
        (root / "UC_Other" / "other.png").write_bytes(b"x")
        assert analysis_runs.list_artifacts(root, "UC_Crop") == [
            "UC_Crop/Array2DScanAnalyzer/a.png",
            "UC_Crop/Array2DScanAnalyzer/b.png",
            "UC_Crop/Array2DScanAnalyzer/c.h5",
        ]
        assert analysis_runs.list_artifacts(root, "UC_Crop", limit=2) == [
            "UC_Crop/Array2DScanAnalyzer/a.png",
            "UC_Crop/Array2DScanAnalyzer/b.png",
        ]
        assert (
            len(analysis_runs.list_artifacts(root, "UC_Crop")) == 3
        )  # uncapped default
        assert analysis_runs.list_artifacts(root, "UC_Missing") == []


class TestRunner:
    def test_one_active_job_per_scan(self):
        runner = analysis_runs.AnalysisRunner()
        gate = threading.Event()
        try:
            job = runner.start("u", "A", lambda: gate.wait(5) and [])
            with pytest.raises(analysis_runs.RunInProgress) as excinfo:
                runner.start("u", "B", lambda: [])
            assert excinfo.value.job is job
            # A different scan is not gated by this one.
            other = runner.start("v", "A", lambda: [])
            assert other.state in analysis_runs.ACTIVE
        finally:
            gate.set()
            runner.shutdown()

    def test_log_capture_is_the_window_minus_portal_traffic(self):
        """Everything the run emits — sub-threads included — minus the app's own."""
        runner = analysis_runs.AnalysisRunner()
        seen = threading.Event()

        def run():
            logging.getLogger("worker").warning("from the worker")
            helper = threading.Thread(
                target=lambda: logging.getLogger("scan_analysis.pool").warning(
                    "from an analyzer sub-thread"
                )
            )
            helper.start()
            helper.join()
            seen.set()
            time.sleep(0.05)
            return []

        try:
            job = runner.start("u", "A", run)
            assert seen.wait(5)
            # Portal / server traffic during the window is not the run's:
            # by logger name, and by thread name for the libraries a
            # concurrent browser exercises (the ephemeral Images path).
            logging.getLogger("uvicorn.access").warning("GET /health")
            logging.getLogger("geecs_portal.app").warning("request-side noise")
            browser = threading.Thread(
                name="AnyIO worker thread",
                target=lambda: logging.getLogger("image_analysis.ephemeral").warning(
                    "a concurrent browser's processing"
                ),
            )
            browser.start()
            browser.join()
            deadline = time.monotonic() + 5
            while job.state in analysis_runs.ACTIVE and time.monotonic() < deadline:
                time.sleep(0.01)
            assert job.state == analysis_runs.DONE
            assert job.log == [
                "WARNING worker: from the worker",
                "WARNING scan_analysis.pool: from an analyzer sub-thread",
            ]
            assert job.finished is not None  # assigned before the state flips
        finally:
            runner.shutdown()

    def test_shutdown_refuses_new_jobs(self):
        runner = analysis_runs.AnalysisRunner()
        runner.shutdown()
        with pytest.raises(RuntimeError):
            runner.start("u", "A", lambda: [])

    def test_base_exception_is_recorded_not_stuck(self):
        runner = analysis_runs.AnalysisRunner()

        def run():
            raise SystemExit(3)

        try:
            job = runner.start("u", "A", run)
            deadline = time.monotonic() + 5
            while job.state in analysis_runs.ACTIVE and time.monotonic() < deadline:
                time.sleep(0.01)
            assert job.state == analysis_runs.FAILED
            assert job.error == "SystemExit: 3"
            # The scan is not left permanently 409.
            assert runner.running_for("u") is None
        finally:
            runner.shutdown()


# ---------------------------------------------------------- the endpoints


class TestFeatureGates:
    def test_shutdown_turns_posts_into_503(self, scan_folder, configs_tree):
        pytest.importorskip("image_analysis")
        client = _client(scan_folder, configs_tree, behaviour="ok")
        with client:  # runs the lifespan: startup … shutdown
            assert (
                client.post(
                    "/api/run/uid-002/analysis", params={"analyzer": "UC_Crop"}
                ).status_code
                == 202
            )
            _wait(client)
        response = client.post(
            "/api/run/uid-002/analysis", params={"analyzer": "UC_Crop"}
        )
        assert response.status_code == 503
        assert "shutting down" in response.json()["detail"]

    def test_off_without_configs_tree(self, scan_folder):
        catalog = FakeCatalog()
        client = TestClient(create_app(catalog))
        assert client.get("/api/run/uid-002/analysis").status_code == 404
        response = client.post("/api/run/uid-002/analysis", params={"analyzer": "x"})
        assert response.status_code == 404
        assert "processing-configs" in response.json()["detail"]

    def test_real_factory_needs_the_extra(self, scan_folder, configs_tree, monkeypatch):
        pytest.importorskip("image_analysis")
        monkeypatch.setitem(__import__("sys").modules, "scan_analysis", None)
        client = _client(scan_folder, configs_tree, analysis_factory=None)
        response = client.get("/api/run/uid-002/analysis")
        assert response.status_code == 404
        assert "analysis" in response.json()["detail"]

    def test_folder_unresolvable(self, configs_tree):
        pytest.importorskip("image_analysis")
        catalog = FakeCatalog()  # uid-002 has no scan_folder; daily path is stubbed
        client = TestClient(
            create_app(
                catalog,
                processing_config_dir=configs_tree,
                analysis_factory=_factory("ok"),
            )
        )
        response = client.post(
            "/api/run/uid-002/analysis", params={"analyzer": "UC_Crop"}
        )
        assert response.status_code == 404
        assert "not resolvable" in response.json()["detail"]
        assert FakeAnalyzer.instances == []  # nothing built, nothing run


class TestListing:
    def test_applicability_and_empty_records(self, scan_folder, configs_tree):
        pytest.importorskip("image_analysis")
        client = _client(scan_folder, configs_tree)
        body = client.get("/api/run/uid-002/analysis").json()
        by_id = {a["id"]: a for a in body["analyzers"]}
        assert set(by_id) == {"UC_Crop", "UC_Other"}  # the legacy flat YAML is dropped
        assert by_id["UC_Crop"]["applicable"] is True
        assert by_id["UC_Crop"]["device"] == "cam"
        assert by_id["UC_Other"]["applicable"] is False
        assert all(a["job"] is None and a["files"] == [] for a in body["analyzers"])
        assert body["running"] is None

    def test_files_on_disk_survive_without_a_record(self, scan_folder, configs_tree):
        """A page loaded after a portal restart still lists earlier outputs."""
        pytest.importorskip("image_analysis")
        analysis = analysis_runs.analysis_folder_for(scan_folder)
        (analysis / "UC_Crop" / "Array2DScanAnalyzer").mkdir(parents=True)
        (analysis / "UC_Crop" / "Array2DScanAnalyzer" / "old.png").write_bytes(b"x")
        client = _client(scan_folder, configs_tree)
        body = client.get("/api/run/uid-002/analysis").json()
        crop = next(a for a in body["analyzers"] if a["id"] == "UC_Crop")
        assert crop["job"] is None
        assert crop["files"] == ["UC_Crop/Array2DScanAnalyzer/old.png"]
        assert crop["artifacts"] == [
            {
                "path": "UC_Crop/Array2DScanAnalyzer/old.png",
                "servable": True,
                "inline": True,
                "kind": "other",
                "bin": None,
            }
        ]

    def test_described_artifacts_policy(self, tmp_path):
        root = tmp_path / "analysis" / "Scan002"
        (root / "UC_Crop").mkdir(parents=True)
        (root / "UC_Crop" / "fig.svg").write_text("<svg/>")
        (root / "UC_Crop" / "fig.png").write_bytes(b"x")
        describe = analysis_runs.describe_artifact
        assert describe(root, "UC_Crop/fig.png") == {
            "path": "UC_Crop/fig.png",
            "servable": True,
            "inline": True,
            "kind": "other",
            "bin": None,
        }
        assert describe(root, "UC_Crop/fig.svg")["inline"] is False  # download only
        assert describe(root, "UC_Crop/fig.svg")["servable"] is True
        assert describe(root, "../s2.txt") == {
            "path": "../s2.txt",
            "servable": False,
            "inline": False,
            "kind": "other",
            "bin": None,
        }
        assert describe(root, "a label.")["servable"] is False

    def test_artifact_kinds_follow_scan_analysis_naming(self, tmp_path):
        """Classification is ScanAnalysis's own contract (parse_output_filename)."""
        pytest.importorskip("scan_analysis")
        classify = analysis_runs.classify_artifact
        # Real renderer names (RenderContext.get_filename + the summary names).
        assert classify("UC_X/Array2DScanAnalyzer/UC_X_16_processed_visual.png") == (
            "bin",
            16,
        )
        assert classify("UC_X/Array2DScanAnalyzer/UC_X_16_processed.h5") == ("bin", 16)
        assert classify(
            "UC_X/Array2DScanAnalyzer/UC_X_average_processed_visual.png"
        ) == ("summary", None)
        assert classify("UC_X/Array2DScanAnalyzer/UC_X_averaged_image_grid.png") == (
            "summary",
            None,
        )
        assert classify("UC_L/Array1DScanAnalyzer/UC_L_summary_waterfall.png") == (
            "summary",
            None,
        )
        assert classify("UC_X/Array2DScanAnalyzer/noscan.gif") == ("summary", None)
        assert classify("UC_X/UC_X_dynamic_background.npy") == ("other", None)
        assert classify("a label, not a path") == ("other", None)
        # Ordering for the tab: summaries, others, then bins by NUMBER (not
        # lexically); the cap applies to bins only, never to summaries.
        root = tmp_path / "analysis" / "Scan002"
        (root / "UC_X").mkdir(parents=True)
        names = [
            "UC_X_10_processed_visual.png",
            "UC_X_2_processed_visual.png",
            "UC_X_averaged_image_grid.png",
            "UC_X_2_processed.h5",
            "UC_X_dynamic_background.npy",
        ]
        for name in names:
            (root / "UC_X" / name).write_bytes(b"x")
        described = analysis_runs.describe_artifacts(root, [f"UC_X/{n}" for n in names])
        assert [(d["kind"], d["bin"]) for d in described] == [
            ("summary", None),
            ("other", None),
            ("bin", 2),
            ("bin", 2),
            ("bin", 10),
        ]
        capped = analysis_runs.describe_artifacts(
            root, [f"UC_X/{n}" for n in names], max_bins=1
        )
        assert [(d["kind"], d["bin"]) for d in capped] == [
            ("summary", None),
            ("other", None),
            ("bin", 2),
        ]

    def test_done_artifacts_and_serving(self, scan_folder, configs_tree, caplog):
        pytest.importorskip("image_analysis")
        # Capture respects the process's logging levels (no global level
        # mutation): the analyzer's INFO line is seen only when its
        # logger admits INFO — the portal runs at INFO by default.
        caplog.set_level(logging.INFO, logger="fake.analyzer")
        client = _client(scan_folder, configs_tree)
        before = sorted(p.relative_to(scan_folder) for p in scan_folder.rglob("*"))
        response = client.post(
            "/api/run/uid-002/analysis", params={"analyzer": "UC_Crop"}
        )
        assert response.status_code == 202
        assert response.json()["state"] in analysis_runs.ACTIVE
        job = _wait(client)
        assert job["state"] == "done"
        assert job["error"] is None
        assert job["artifacts"] == [
            "UC_Crop/Array2DScanAnalyzer/summary.png",
            "a label, not a path",
        ]
        assert "fake analyzer running on scan 2" in " ".join(job["log"])
        (analyzer,) = FakeAnalyzer.instances
        assert analyzer.cleaned is True
        tag = analyzer.scan_tag
        assert (tag.number, tag.experiment) == (2, "Undulator")
        # The tag comes from the RESOLVED FOLDER (26_0713), not from the
        # start doc's time (TEST_DAY = 07-12) — the midnight-claim / TZ hazard.
        assert (tag.year, tag.month, tag.day) == (2026, 7, 13)
        # The scan folder itself is untouched by the portal side.
        after = sorted(p.relative_to(scan_folder) for p in scan_folder.rglob("*"))
        assert after == before
        # Listing now shows the file on disk too; the artifact endpoint serves it.
        body = client.get("/api/run/uid-002/analysis").json()
        crop = next(a for a in body["analyzers"] if a["id"] == "UC_Crop")
        assert crop["files"] == ["UC_Crop/Array2DScanAnalyzer/summary.png"]
        # The tab renders the DESCRIBED artifacts: policy decided server-side.
        # (files on disk + the job's non-file labels, classified server-side)
        assert crop["artifacts"] == [
            {
                "path": "UC_Crop/Array2DScanAnalyzer/summary.png",
                "servable": True,
                "inline": True,
                "kind": "other",
                "bin": None,
            },
            {
                "path": "a label, not a path",
                "servable": False,
                "inline": False,
                "kind": "other",
                "bin": None,
            },
        ]
        served = client.get(
            "/run/uid-002/artifact",
            params={"path": "UC_Crop/Array2DScanAnalyzer/summary.png"},
        )
        assert served.status_code == 200
        assert served.content == b"\x89PNG fake"
        assert served.headers["cache-control"] == "no-cache"
        assert served.headers["content-type"].startswith("image/png")

    def test_failure_is_a_record_not_a_500(self, scan_folder, configs_tree):
        pytest.importorskip("image_analysis")
        client = _client(scan_folder, configs_tree, behaviour="fail")
        assert (
            client.post(
                "/api/run/uid-002/analysis", params={"analyzer": "UC_Crop"}
            ).status_code
            == 202
        )
        job = _wait(client)
        assert job["state"] == "failed"
        assert job["error"] == "RuntimeError: boom"
        assert "WARNING fake.analyzer: about to fail" in job["log"]
        (analyzer,) = FakeAnalyzer.instances
        assert analyzer.cleaned is True  # cleanup runs on failure too

    @pytest.mark.parametrize("behaviour", ["none", "no_data"])
    def test_skips_are_no_data(self, scan_folder, configs_tree, behaviour):
        pytest.importorskip("image_analysis")
        client = _client(scan_folder, configs_tree, behaviour=behaviour)
        client.post("/api/run/uid-002/analysis", params={"analyzer": "UC_Crop"})
        job = _wait(client)
        assert job["state"] == "no_data"
        assert job["artifacts"] == []
        assert job["error"]

    def test_second_post_409_while_running(self, scan_folder, configs_tree):
        pytest.importorskip("image_analysis")
        client = _client(scan_folder, configs_tree, behaviour="block")
        first = client.post("/api/run/uid-002/analysis", params={"analyzer": "UC_Crop"})
        assert first.status_code == 202
        (analyzer,) = FakeAnalyzer.instances
        assert analyzer.started.wait(5)
        again = client.post("/api/run/uid-002/analysis", params={"analyzer": "UC_Crop"})
        assert again.status_code == 409
        assert again.json()["job"]["state"] == "running"
        assert client.get("/api/run/uid-002/analysis").json()["running"] == "UC_Crop"
        analyzer.release.set()
        assert _wait(client)["state"] == "done"
        # Re-run after completion is allowed (overwrites by name).
        assert (
            client.post(
                "/api/run/uid-002/analysis", params={"analyzer": "UC_Crop"}
            ).status_code
            == 202
        )
        FakeAnalyzer.instances[-1].release.set()
        assert _wait(client)["state"] == "done"

    @pytest.mark.parametrize("analyzer", ["UC_Nope", "UC_Legacy"])
    def test_unknown_or_unloadable_diagnostic_404(
        self, scan_folder, configs_tree, analyzer
    ):
        pytest.importorskip("image_analysis")
        client = _client(scan_folder, configs_tree)
        response = client.post(
            "/api/run/uid-002/analysis", params={"analyzer": analyzer}
        )
        assert response.status_code == 404
        assert FakeAnalyzer.instances == []


class TestArtifactEndpoint:
    def test_gated_with_the_feature(self, scan_folder):
        analysis = analysis_runs.analysis_folder_for(scan_folder)
        (analysis / "UC_Crop").mkdir(parents=True)
        (analysis / "UC_Crop" / "fig.png").write_bytes(b"x")
        catalog = FakeCatalog()
        detail = _detail(2)
        detail.start_doc["scan_folder"] = str(scan_folder)
        catalog.details["uid-002"] = detail
        client = TestClient(create_app(catalog))  # feature off
        response = client.get(
            "/run/uid-002/artifact", params={"path": "UC_Crop/fig.png"}
        )
        assert response.status_code == 404

    def test_only_raster_images_render_inline(self, scan_folder, configs_tree):
        """A planted HTML/SVG on the share must never execute in the portal's origin."""
        pytest.importorskip("image_analysis")
        analysis = analysis_runs.analysis_folder_for(scan_folder)
        (analysis / "UC_Crop").mkdir(parents=True)
        (analysis / "UC_Crop" / "fig.png").write_bytes(b"\x89PNG")
        (analysis / "UC_Crop" / "evil.html").write_text("<script>alert(1)</script>")
        (analysis / "UC_Crop" / "plot.svg").write_text("<svg onload='alert(1)'/>")
        (analysis / "UC_Crop" / "data.h5").write_bytes(b"\x89HDF")
        client = _client(scan_folder, configs_tree)
        png = client.get("/run/uid-002/artifact", params={"path": "UC_Crop/fig.png"})
        assert png.status_code == 200
        assert png.headers["content-disposition"].startswith("inline")
        assert png.headers["x-content-type-options"] == "nosniff"
        for name in ("evil.html", "plot.svg", "data.h5"):
            response = client.get(
                "/run/uid-002/artifact", params={"path": f"UC_Crop/{name}"}
            )
            assert response.status_code == 200, name
            assert response.headers["content-disposition"].startswith("attachment"), (
                name
            )
            assert response.headers["x-content-type-options"] == "nosniff"

    def test_containment(self, scan_folder, configs_tree):
        pytest.importorskip("image_analysis")
        analysis = analysis_runs.analysis_folder_for(scan_folder)
        analysis.mkdir(parents=True)
        sfile = analysis.parent / "s2.txt"
        sfile.write_text("Shotnumber\n1\n")
        client = _client(scan_folder, configs_tree)
        for path in ("../s2.txt", str(sfile), "", "..", "UC_Crop/../../s2.txt"):
            response = client.get("/run/uid-002/artifact", params={"path": path})
            assert response.status_code == 404, path


class TestRealFactory:
    def test_builds_a_scan_analyzer_from_the_tree(self, configs_tree):
        """The default factory = load_diagnostic + create_scan_analyzer."""
        pytest.importorskip("image_analysis")
        pytest.importorskip("scan_analysis")
        analyzer = analysis_runs.scan_analysis_factory("UC_Crop", configs_tree)
        assert callable(analyzer.run_analysis) and callable(analyzer.cleanup)
        assert analyzer.id == "UC_Crop"
        assert analyzer.data_device_name == "cam"


class TestAnalysisTab:
    """The scan page offers the Analysis tab only when runs are possible."""

    def test_tab_present_and_sticky_when_enabled(self, scan_folder, configs_tree):
        pytest.importorskip("image_analysis")
        client = _client(scan_folder, configs_tree)
        page = client.get("/run/uid-002", params={"tab": "analysis"})
        assert page.status_code == 200
        assert 'data-pane="analysis"' in page.text
        assert 'id="pane-analysis"' in page.text
        assert "const ANALYSIS_ENABLED = true;" in page.text
        # The URL-carried tab survives into the page state (a link IS the view).
        assert 'tab: p.get("tab") || "analysis"' in page.text

    def test_tab_absent_when_feature_off(self, scan_folder):
        catalog = FakeCatalog()
        detail = _detail(2)
        detail.start_doc["scan_folder"] = str(scan_folder)
        catalog.details["uid-002"] = detail
        client = TestClient(create_app(catalog))
        page = client.get("/run/uid-002", params={"tab": "analysis"})
        assert page.status_code == 200
        assert 'data-pane="analysis"' not in page.text
        assert "const ANALYSIS_ENABLED = false;" in page.text
        # The server default is Plot; the URL value still wins in
        # readState, so the JS fallback (setTab: no pane → plot) is
        # what keeps a carried tab=analysis from lighting nothing.
        assert 'tab: p.get("tab") || "plot"' in page.text
        assert 'if (!document.getElementById("pane-" + tab)) tab = "plot";' in page.text

    def test_tab_absent_when_folder_unresolvable(self, configs_tree):
        pytest.importorskip("image_analysis")
        client = TestClient(
            create_app(
                FakeCatalog(),
                processing_config_dir=configs_tree,
                analysis_factory=_factory("ok"),
            )
        )
        page = client.get("/run/uid-002", params={"tab": "analysis"})
        assert page.status_code == 200
        assert 'data-pane="analysis"' not in page.text

    def test_tab_absent_when_extra_missing(
        self, scan_folder, configs_tree, monkeypatch
    ):
        pytest.importorskip("image_analysis")
        monkeypatch.setitem(__import__("sys").modules, "scan_analysis", None)
        client = _client(scan_folder, configs_tree, analysis_factory=None)
        page = client.get("/run/uid-002")
        assert page.status_code == 200
        assert 'data-pane="analysis"' not in page.text
