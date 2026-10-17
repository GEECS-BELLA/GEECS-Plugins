"""The contract fixtures: the services' own apps, in process, behind the tools.

The scanner is built the way its own suite builds it (``DemoQueueClient``
with ``period=0`` — a manager in memory that moves only when the test
calls ``step()``); the portal the way its suite builds it (a fake
``ScanCatalog`` over a temp scan tree, an injected fake analyzer factory);
the logbook the way its suite builds it (the same temp share, a temp
SQLite store).  All three are mounted on ``httpx.ASGITransport``s that
:mod:`geecs_mcp.clients` is pointed at, and the portal's own figure send
reaches the same logbook through a ``TestClient`` (it drives a sync
client), so a route or model change in any of them fails these tests in
the same PR (DESIGN.md §7).
"""

from __future__ import annotations

import io
import json
import logging
import threading
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest

from geecs_mcp import clients, config
from geecs_mcp.scans import tools as scan_tools

logger = logging.getLogger("tests.fake_analyzer")

#: The portal fixtures' day (the fake start documents' time) and scan.
PORTAL_DAY = date(2026, 7, 12)
PORTAL_SCAN = 2
PORTAL_UID = "uid-002"


# ----------------------------------------------------------------- scanner


@pytest.fixture
def scanner(tmp_path: Path):
    """The scanner service + its demo manager, wired behind the tools."""
    from geecs_scanner.service import ScannerService, StreamCache
    from geecs_scanner.service.demo import (
        DemoQueueClient,
        DemoReadback,
        DemoResolver,
        DemoSettables,
        demo_preflight,
    )
    from geecs_scanner.web import create_app

    streams = StreamCache(clock=lambda: 1_000.0)
    manager = DemoQueueClient(
        streams, period=0.0, first_scan=46, user="geecs-scanner test"
    )
    service = ScannerService(
        manager,
        DemoResolver(),
        experiment="Demo",
        identity="geecs-scanner test",
        streams=streams,
        preflight=demo_preflight,
        version="0.0.0+test",
        settables=DemoSettables(),
        readback=DemoReadback(manager),
        heartbeat_path=tmp_path / "scanner-heartbeat.json",
    )
    return service, manager, create_app(service)


# ----------------------------------------------------------------- logbook


@pytest.fixture
def logbook(tmp_path: Path):
    """The logbook app over the portal fixture's share, with a temp store."""
    from geecs_logbook.app import create_app

    return create_app(
        "Undulator", base_directory=tmp_path, notes_db=tmp_path / "notes.sqlite"
    )


# ------------------------------------------------------------------ portal


def _epoch(hour: int, minute: int = 0) -> float:
    return datetime(
        PORTAL_DAY.year, PORTAL_DAY.month, PORTAL_DAY.day, hour, minute
    ).timestamp()


def _start_doc(scan_number: int, scan_folder: Path) -> dict:
    return {
        "uid": f"uid-{scan_number:03d}",
        "time": _epoch(9, scan_number),
        "geecs_event_schema": 1,
        "plan_name": "geecs_step_scan",
        "motor": None,
        "num_points": 1,
        "shots_per_step": 3,
        "experiment": "Undulator",
        "scan_number": scan_number,
        "save_sets": ["Amp4In"],
        "scan_folder": str(scan_folder),
    }


class FakeCatalog:
    """A ``ScanCatalog`` of one day with two runs, the second on a temp scan tree."""

    def __init__(self, scan_folder: Path):
        import pandas as pd
        from geecs_data_utils.tiled_catalog import RunDetail, summary_from_metadata

        self.details = {}
        for n in (1, PORTAL_SCAN):
            start = _start_doc(
                n, scan_folder if n == PORTAL_SCAN else scan_folder.parent / "Scan001"
            )
            stop = {"time": start["time"] + 30.0, "exit_status": "success"}
            data = pd.DataFrame(
                {
                    "scan_event_index": [1, 2, 3],
                    "cam-MaxCounts": [10.0, 12.5, 11.0],
                    "cam-acq_timestamp": [3.87e9 + 1.0, 3.87e9 + 2.0, 3.87e9 + 3.0],
                    "mono": [4.0, 5.0, 6.0],
                }
            )
            self.details[start["uid"]] = RunDetail(
                summary=summary_from_metadata(start["uid"], start, stop),
                start_doc=start,
                stop_doc=stop,
                data=data,
            )

    def probe(self):
        from geecs_data_utils.tiled_catalog import CatalogStatus

        return CatalogStatus(ok=True, label="fake catalog")

    def list_runs(self, experiment: str, day: date):
        return sorted(
            (d.summary for d in self.details.values()),
            key=lambda s: s.start_time,
            reverse=True,
        )

    def load_run(self, uid: str):
        return self.details[uid]


def _png_bytes(width: int = 40, height: int = 30) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeAnalyzer:
    """ScanAnalysis's two-call contract: one figure, one JSON sidecar, one label."""

    behaviour = "ok"
    release = threading.Event()

    def __init__(self, analysis_folder: Path):
        self.analysis_folder = analysis_folder
        self.progress = None

    def run_analysis(self, scan_tag):
        logger.info("fake analyzer on scan %s", scan_tag.number)
        if FakeAnalyzer.behaviour == "block":
            FakeAnalyzer.release.wait(timeout=10)
        if FakeAnalyzer.behaviour == "fail":
            raise RuntimeError("boom")
        out = self.analysis_folder / "UC_Crop" / "Array2DScanAnalyzer"
        out.mkdir(parents=True, exist_ok=True)
        figure = out / "cam_summary_waterfall.png"
        figure.write_bytes(_png_bytes())
        sidecar = out / "cam_summary_scalar_fit.json"
        sidecar.write_text(json.dumps({"slope": 0.0508, "crossing_mm": -3.054}))
        table = out / "cam_shots.txt"
        table.write_text("".join(f"{i}\t{i * 0.5}\n" for i in range(5000)))
        (out / "cam_processed.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
        return [figure, "a label, not a path"]

    def cleanup(self):
        pass


@pytest.fixture
def portal(tmp_path: Path, logbook):
    """The portal app over a temp scan tree and a two-document configs tree.

    Its figure send talks to the in-process logbook (``logbook_client``).
    """
    import yaml
    from fastapi.testclient import TestClient
    from geecs_data_utils.scan_info import analysis_folder_for
    from geecs_portal.app import create_app

    scan_folder = (
        tmp_path
        / "Undulator"
        / "Y2026"
        / "07-Jul"
        / "26_0712"
        / "scans"
        / f"Scan{PORTAL_SCAN:03d}"
    )
    (scan_folder / "cam").mkdir(parents=True)
    analysis_folder = analysis_folder_for(scan_folder)
    tree = tmp_path / "proc_configs"
    analyzers = tree / "analyzers" / "HTU"
    analyzers.mkdir(parents=True)
    (analyzers / "UC_Crop.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "device": "cam",
                "output_name": "UC_Crop",
                "input": {"kind": "camera"},
                "scan": {"priority": 100},
            }
        )
    )
    (analyzers / "UC_Compact.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "himg_compact",
                "device": "cam",
                "output_name": "UC_Compact",
            }
        )
    )
    groups = tree / "groups" / "HTU"
    groups.mkdir(parents=True)
    (groups / "all.yaml").write_text(
        yaml.safe_dump({"name": "HTU_all", "analyzers": ["UC_Crop"]})
    )

    def factory(analyzer_id: str, config_dir: Path, *, allow_destructive: bool = False):
        return FakeAnalyzer(analysis_folder)

    FakeAnalyzer.behaviour = "ok"
    FakeAnalyzer.release = threading.Event()
    app = create_app(
        FakeCatalog(scan_folder),
        default_experiment="Undulator",
        processing_config_dir=tree,
        analysis_factory=factory,
        logbook_url="http://logbook",
        logbook_client=TestClient(logbook, base_url="http://logbook"),
    )
    return app, scan_folder, analysis_folder


@pytest.fixture
def portal_fakes():
    """The portal fixture's fake analyzer class and ids, for the test module.

    A fixture rather than ``from tests.conftest import …``: the
    subpackages' identically named ``tests`` trees collide on import.
    """
    from types import SimpleNamespace

    return SimpleNamespace(analyzer=FakeAnalyzer, scan=PORTAL_SCAN, uid=PORTAL_UID)


# ------------------------------------------------------------------ wiring


@pytest.fixture
def wired(scanner, portal, logbook, monkeypatch):
    """Point the clients at the in-process scanner, portal and logbook."""
    _, manager, scanner_app = scanner
    portal_app, _, _ = portal
    transports = {
        "scanner": httpx.ASGITransport(app=scanner_app),
        "portal": httpx.ASGITransport(app=portal_app),
        "logbook": httpx.ASGITransport(app=logbook),
    }
    urls = {
        "scanner": "http://scanner",
        "portal": "http://portal",
        "logbook": "http://logbook",
    }
    monkeypatch.setattr(config, "service_url", lambda service: urls[service])
    monkeypatch.setattr(
        clients, "transport_for", lambda service: transports.get(service)
    )
    yield manager
    import anyio

    anyio.run(clients.reset_clients)


@pytest.fixture
def stepping(wired, monkeypatch):
    """Every pause in a scans wait verb advances the demo manager one shot."""

    async def step_instead(_seconds: float) -> None:
        wired.step()

    monkeypatch.setattr(scan_tools, "_sleep", step_instead)
    return wired


@pytest.fixture
def frozen_clock(monkeypatch):
    """A clock the test advances by hand (the scans wait verbs' deadline)."""
    now = {"t": 0.0}
    monkeypatch.setattr(scan_tools, "_clock", lambda: now["t"])
    return now
