"""Phase-1 hardware acceptance: the plan layer end to end on HTU (#807, PR 3).

Hardware-marked **and** gated on ``GEECS_HW=1``: it arms the machine
trigger and fires shots (an explicit ``-m`` on the command line overrides
the ``addopts`` deselect, so the marker alone does not protect it).  Run by
hand on a host with CA reach to the GEECS gateway, the DB and the data
share (the qserver box), with the configs root at the presets corpus::

    GEECS_HW=1 GEECS_SCANNER_CONFIG_DIR=.../GEECS-Plugins-Configs/scanner_configs/experiments \\
    poetry run python -u -m pytest tests/test_phase1_hardware.py -m hardware -s

Two tests, the second only with ``GEECS_HW_QSERVER`` set:

1. **In-process** — the worker's own wiring (`make_run_engine(claim=True,
   …)`, the namespace on the shared path provider, the trigger profiles,
   the bound plans) drives a strict ``count`` and a strict ``scan`` of
   ``U_S1H:Current``; asserts every GEECS output of a run: the claimed
   ``ScanNNN/`` folder, the camera's native files named by the rows'
   stamps in ``ScanNNN/<device>/``, ``ScanInfoScanNNN.ini`` with the keys
   downstream parses, the s-file with ``Bin #`` per step, ``scan.log``,
   the ``baseline`` stream, ARMED → STANDBY around each run; and records
   the per-shot cadence (M2's every-other-edge on a motor scan is the
   number PR 3 measures).
2. **Through the RE Manager** — a second manager on this host
   (``GEECS_HW_QSERVER=tcp://localhost:60635``, see the runbook in
   ``Planning/native_bluesky/05_phase1_acceptance.md``): the client seam
   expands a ``Preset`` (``run_submit_preflight`` then ``submit_preset``),
   the manager runs it, and the same files are asserted from the newest
   scan folder.

**MOVES HARDWARE**: the swept setpoint is restored in a ``finally``.

Environment
-----------
``GEECS_HW_EXPERIMENT``       experiment (default ``Undulator``)
``GEECS_HW_CAMERA_DEVICE``    camera (default ``UC_Amp4_IR_input``)
``GEECS_HW_TRIGGER_PROFILE``  trigger profile (default ``HTU-NoGas``)
``GEECS_HW_SCAN_VARIABLE``    ``Device:Variable`` to sweep (default ``U_S1H:Current``)
``GEECS_HW_SCAN_START/END/NUM``  sweep (default −1 → 1 in 5 points)
``GEECS_HW_SHOTS``            shots for the count (default 3)
``GEECS_HW_SHOTS_PER_STEP``   rows per position for the scan (default 2)
``GEECS_HW_QSERVER``          control address of the acceptance manager (unset → test 2 skipped)
``GEECS_HW_DOCS_OUT``         optional path to dump the in-process documents as JSON
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from configparser import ConfigParser
from pathlib import Path

import pytest

from tests.ca_mock_helpers import DocCollector

pytestmark = pytest.mark.hardware
pytest.importorskip("aioca")
if os.environ.get("GEECS_HW") != "1":
    pytest.skip(
        "fires real shots: set GEECS_HW=1 to run on the lab network",
        allow_module_level=True,
    )

EXPERIMENT = os.environ.get("GEECS_HW_EXPERIMENT", "Undulator")
CAMERA = os.environ.get("GEECS_HW_CAMERA_DEVICE", "UC_Amp4_IR_input")
PROFILE = os.environ.get("GEECS_HW_TRIGGER_PROFILE", "HTU-NoGas")
SWEEP = os.environ.get("GEECS_HW_SCAN_VARIABLE", "U_S1H:Current")
START = float(os.environ.get("GEECS_HW_SCAN_START", "-1"))
END = float(os.environ.get("GEECS_HW_SCAN_END", "1"))
NUM = int(os.environ.get("GEECS_HW_SCAN_NUM", "5"))
SHOTS = int(os.environ.get("GEECS_HW_SHOTS", "3"))
SHOTS_PER_STEP = int(os.environ.get("GEECS_HW_SHOTS_PER_STEP", "2"))


def _ini(path: Path) -> dict[str, str]:
    parser = ConfigParser()
    parser.optionxform = str  # type: ignore[assignment]
    parser.read(path)
    return {k: v.strip('"') for k, v in parser.items("Scan Info")}


def _wait_for_files(
    directory: Path, expected: int, timeout: float = 15.0
) -> list[Path]:
    """Every expected native file exists and has stopped growing (two agreeing stats)."""
    deadline = time.monotonic() + timeout
    while True:
        files = sorted(directory.glob("*.png"))
        sizes = [f.stat().st_size for f in files]
        if len(files) >= expected and all(sizes):
            time.sleep(0.5)
            if [f.stat().st_size for f in files] == sizes:
                return files
        if time.monotonic() > deadline:
            raise AssertionError(
                f"{directory}: {len(files)} files after {timeout:.0f} s, expected {expected}"
            )
        time.sleep(0.5)


def _assert_scan_outputs(
    folder: Path,
    *,
    camera_name: str,
    expected_rows: int,
    expected_bins: list[int],
    scan_parameter: str,
    shots_per_step: int,
    plan_name: str,
) -> None:
    """The four GEECS outputs of one claimed run, checked from disk."""
    import pandas as pd

    number = int("".join(ch for ch in folder.name if ch.isdigit()))
    info = _ini(folder / f"ScanInfo{folder.name}.ini")
    assert info["Scan No"] == str(number)
    assert info["Scan Parameter"] == scan_parameter, info
    assert info["Shots per step"] == str(shots_per_step), info
    assert info["ScanEndInfo"] == "success", info
    assert info["Plan"] == plan_name and info["Scanner"] == "bluesky"
    sfile = pd.read_csv(folder.parent.parent / "analysis" / f"s{number}.txt", sep="\t")
    scan_txt = pd.read_csv(folder / f"ScanData{folder.name}.txt", sep="\t")
    assert len(sfile) == expected_rows and list(sfile.columns) == list(scan_txt.columns)
    assert list(sfile["Bin #"]) == expected_bins
    assert f"{camera_name} acq_timestamp" in sfile.columns
    assert (folder / "scan.log").read_text().count("finished (success)") == 1
    files = _wait_for_files(folder / camera_name, expected_rows)
    stamps = sfile[f"{camera_name} acq_timestamp"].tolist()
    names = {f.name for f in files}
    for stamp in stamps:
        assert any(f"{stamp:.3f}" in n for n in names), (stamp, sorted(names)[:3])
    print(
        f"{folder.name}: {expected_rows} rows, {len(files)} native files, bins {expected_bins}"
    )


@pytest.mark.hardware
def test_plan_layer_in_process_on_hardware() -> None:
    """The worker's wiring runs a strict count and a strict scan and leaves every file."""
    import bluesky.plan_stubs as bps
    from geecs_core.pv_naming import pv_name, setpoint_pv

    from geecs_bluesky.config_resolver import ConfigsRepoResolver
    from geecs_bluesky.devices.ca.oneshot import try_caget_once
    from geecs_bluesky.namespace import GeecsNamespace
    from geecs_bluesky.plans.claim_scan import GeecsScanPathProvider
    from geecs_bluesky.plans.registry import TriggerProfiles, bind_strict_plans
    from geecs_bluesky.run_engine import make_run_engine

    t_build = time.monotonic()
    provider = GeecsScanPathProvider()
    namespace = GeecsNamespace.from_experiment(EXPERIMENT, path_provider=provider)
    profiles = TriggerProfiles.from_resolver(
        ConfigsRepoResolver(EXPERIMENT), experiment=EXPERIMENT
    )
    assert PROFILE in profiles.names, profiles.names
    RE = make_run_engine(
        experiment=EXPERIMENT,
        tiled=True,
        claim=True,
        path_provider=provider,
        telemetry=namespace.telemetry(),
    )
    plans = bind_strict_plans(profiles)
    camera = namespace[CAMERA]
    assert camera.native_save, f"{CAMERA} has no native saving controls in the DB"
    motor = namespace.resolve(SWEEP)
    device_name, _, variable = SWEEP.partition(":")
    initial = float(
        try_caget_once(
            setpoint_pv(pv_name(EXPERIMENT, device_name, variable)), timeout=5.0
        )
    )
    print(
        f"\nbuilt in {time.monotonic() - t_build:.1f} s: {len(namespace)} devices, "
        f"profiles {profiles.names}, sweep {SWEEP} from setpoint {initial}"
    )

    docs = DocCollector()
    RE.subscribe(docs)

    t0 = time.monotonic()
    try:
        RE(
            plans["count"](
                [camera],
                SHOTS,
                trigger_profile=PROFILE,
                md={"description": "807 phase 1 acceptance: count"},
            )
        )
        RE(
            plans["scan"](
                [camera],
                motor,
                START,
                END,
                NUM,
                shots_per_step=SHOTS_PER_STEP,
                trigger_profile=PROFILE,
                md={"description": "807 phase 1 acceptance: scan"},
            )
        )
    finally:
        RE(bps.mv(motor, initial))
        print(f"restored {SWEEP} to {initial}")
    wall = time.monotonic() - t0

    out = os.environ.get("GEECS_HW_DOCS_OUT")
    if out:
        Path(out).write_text(json.dumps(docs.docs, default=str, indent=1))

    starts, stops = docs.docs["start"], docs.docs["stop"]
    assert [s["plan_name"] for s in starts] == ["count", "scan"]
    assert all(s["exit_status"] == "success" for s in stops), stops
    for start in starts:
        assert start["experiment"] == EXPERIMENT
        assert start["trigger_profile"] == PROFILE
        assert Path(start["scan_folder"]).is_dir()
        assert start["geecs_scalar_headers"]
    assert (
        starts[0]["shots_per_step"] == 1
        and starts[1]["shots_per_step"] == SHOTS_PER_STEP
    )
    assert starts[1]["scan_number"] == starts[0]["scan_number"] + 1

    primary = docs.primary_events()
    assert len(primary) == SHOTS + NUM * SHOTS_PER_STEP
    stream_names = {d["name"] for d in docs.docs["descriptor"]}
    assert "baseline" in stream_names
    baseline_uids = {
        d["uid"] for d in docs.docs["descriptor"] if d["name"] == "baseline"
    }
    baseline_rows = [e for e in docs.docs["event"] if e["descriptor"] in baseline_uids]
    assert len(baseline_rows) == 4  # open + close, two runs
    print(f"baseline stream: {len(baseline_rows[0]['data'])} columns")

    shot_control = profiles.resolve(PROFILE)
    assert shot_control.standing_state == "STANDBY"

    key = f"{camera.name}-acq_timestamp"
    stamps = [e["data"][key] for e in primary]
    assert len(set(stamps)) == len(stamps), "stamps did not advance per shot"
    # Cadence from the rows' own stamps (the shot times, §11.3), per run.
    count_stamps, scan_stamps = stamps[:SHOTS], stamps[SHOTS:]
    print(
        f"count cadence (s): {[round(b - a, 3) for a, b in zip(count_stamps, count_stamps[1:])]}"
    )
    print(
        f"scan cadence (s): {[round(b - a, 3) for a, b in zip(scan_stamps, scan_stamps[1:])]}"
    )
    print(f"wall: {wall:.1f} s for {len(primary)} shots")

    motor_header = starts[1]["geecs_scalar_headers"][motor.position.name]
    _assert_scan_outputs(
        Path(starts[0]["scan_folder"]),
        camera_name=CAMERA,
        expected_rows=SHOTS,
        expected_bins=[1] * SHOTS,
        scan_parameter="Shotnumber",
        shots_per_step=SHOTS,
        plan_name="count",
    )
    _assert_scan_outputs(
        Path(starts[1]["scan_folder"]),
        camera_name=CAMERA,
        expected_rows=NUM * SHOTS_PER_STEP,
        expected_bins=[b for b in range(1, NUM + 1) for _ in range(SHOTS_PER_STEP)],
        scan_parameter=motor_header,
        shots_per_step=SHOTS_PER_STEP,
        plan_name="scan",
    )
    info = _ini(
        Path(starts[1]["scan_folder"])
        / f"ScanInfo{Path(starts[1]['scan_folder']).name}.ini"
    )
    assert (float(info["Start"]), float(info["End"])) == (START, END)
    assert float(info["Step size"]) == pytest.approx((END - START) / (NUM - 1))
    save_state = asyncio.run_coroutine_threadsafe(
        camera.save.get_value(), RE._loop
    ).result(5)
    assert save_state == "off"


@pytest.mark.hardware
def test_preset_through_the_manager_on_hardware() -> None:
    """A preset expanded by the client seam runs on a second RE Manager and leaves the files."""
    control = os.environ.get("GEECS_HW_QSERVER")
    if not control:
        pytest.skip("set GEECS_HW_QSERVER=tcp://host:port to the acceptance manager")
    pytest.importorskip("bluesky_queueserver_api")
    from geecs_core.pv_naming import pv_name, setpoint_pv
    from geecs_schemas import Preset

    from geecs_bluesky.config_resolver import ConfigsRepoResolver
    from geecs_bluesky.devices.ca.oneshot import try_caget_once
    from geecs_bluesky.plan_names import GEECS_PLAN_NAMES
    from geecs_bluesky.qs_client import (
        QserverConfig,
        ZmqQueueClient,
        run_submit_preflight,
    )

    host, _, port = control.rpartition(":")
    client = ZmqQueueClient(
        QserverConfig(control, f"{host}:{int(port) + 10}", "OFF"), user="807-acceptance"
    )
    verdict = client.readiness(GEECS_PLAN_NAMES)
    assert verdict.ready, verdict.detail
    device_name, _, variable = SWEEP.partition(":")
    initial = float(
        try_caget_once(
            setpoint_pv(pv_name(EXPERIMENT, device_name, variable)), timeout=5.0
        )
    )
    preset = Preset.model_validate(
        {
            "name": "807-phase1-acceptance",
            "description": "807 phase 1 acceptance: preset through the manager",
            "trigger_profile": PROFILE,
            "devices": [{"device": CAMERA}],
            "plan": {
                "name": "scan",
                "args": [SWEEP, START, END, NUM],
                "kwargs": {"shots_per_step": SHOTS_PER_STEP},
            },
        }
    )
    catalog = ConfigsRepoResolver(EXPERIMENT).scan_variable_catalog().variables
    report = run_submit_preflight(preset, EXPERIMENT, client=client, catalog=catalog)
    assert report.refusal is None, report.refusal
    print(
        f"\npreflight: {report.outcomes}, questions {[q.check for q in report.questions]}"
    )

    scans_dir = _scans_dir()
    before = {p.name for p in scans_dir.glob("Scan*")}
    result = client.submit_preset(preset, catalog=catalog)
    assert result.ok, result.message
    try:
        item = _wait_for_item(client, result.item_uid, timeout=300.0)
        assert item.get("result", {}).get("exit_status") == "completed", item.get(
            "result"
        )
    finally:
        # The scan item is in the history by now (or the wait raised), so the
        # queue is empty and the restore is accepted; clear_pending covers a
        # scan that is still queued after a failed wait — the restore must
        # not be refused for the #648 front-of-queue guard.
        restore = client.submit_plan(
            "mv",
            args=[f"{device_name}.{variable.lower()}", initial],
            clear_pending=True,
        )
        print(f"restore {SWEEP} to {initial}: {restore.message}")
        assert restore.ok, restore.message
        _wait_for_item(client, restore.item_uid, timeout=120.0)
        client.close()
    new = sorted({p.name for p in scans_dir.glob("Scan*")} - before)
    assert len(new) == 1, new
    folder = scans_dir / new[0]
    catalog_entry = catalog.get(SWEEP)
    _assert_scan_outputs(
        folder,
        camera_name=CAMERA,
        expected_rows=NUM * SHOTS_PER_STEP,
        expected_bins=[b for b in range(1, NUM + 1) for _ in range(SHOTS_PER_STEP)],
        scan_parameter=f"{device_name} {variable}" if catalog_entry is None else SWEEP,
        shots_per_step=SHOTS_PER_STEP,
        plan_name="scan",
    )
    info = _ini(folder / f"ScanInfo{folder.name}.ini")
    assert info["ScanStartInfo"] == preset.description
    assert info["Trigger profile"] == PROFILE


def _wait_for_item(client, item_uid: str, *, timeout: float) -> dict:
    """Block until the queue item *item_uid* is in the manager's history; return it.

    The manager starts a freshly queued item asynchronously — a status poll
    right after ``queue_start`` still reads idle — so "done" is the item
    leaving the queue and the running slot and appearing in the history.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for entry in client.history_items():
            if entry.get("item_uid") == item_uid:
                return entry
        time.sleep(1.0)
    raise AssertionError(f"queue item {item_uid} did not finish within {timeout:.0f} s")


def _scans_dir() -> Path:
    from geecs_data_utils import ScanPaths

    if ScanPaths.paths_config is None:
        ScanPaths.reload_paths_config(default_experiment=EXPERIMENT)
    return ScanPaths.get_daily_scan_folder(experiment=EXPERIMENT)
