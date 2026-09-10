"""Phase-2 hardware acceptance: a stock plan run as a full GEECS scan (#807).

Hardware-marked (skipped in CI; ``-m integration`` does NOT select it — it
arms the machine trigger **and claims a real scan number**).  Run by hand on
a host with CA reach to the gateway and the GEECS DB (the qserver box)::

    GEECS_HW_SCAN_VARIABLE=S1H GEECS_HW_SCAN_START=-1 \\
    GEECS_HW_SCAN_END=1 GEECS_HW_SCAN_STEP=0.5 \\
    poetry run pytest tests/test_preamble_hardware.py -m hardware -s

**Side effects, deliberately real:** every run here claims a scan number,
creates ``scans/ScanNNN/`` with its ``ScanInfoScanNNN.ini``, registers the
run in the facility **Tiled** catalog and exports its s-file — that is the
behaviour under test, and it is what makes the run visible in the data
portal.  The claimed numbers are printed.

What it proves that the mock cannot: the preamble preprocessor prepares a
**stock** ``bluesky.plans`` verb exactly as the funnel prepares a submitted
ScanRequest — the same validation, resolution, claim, ScanInfo, native-save
configuration and run metadata — against the live gateway and DB.

Environment
-----------
``GEECS_HW_EXPERIMENT``       experiment (default ``Undulator``)
``GEECS_HW_SAVE_SET``         save set (default ``Amp4In``)
``GEECS_HW_TRIGGER_PROFILE``  trigger profile (default ``HTU-NoGas``)
``GEECS_HW_SHOTS``            shots (default 3)
``GEECS_HW_SCAN_VARIABLE``    catalog scan-variable name to sweep, e.g. ``S1H``
                              (unset → noscan only)
``GEECS_HW_SCAN_START/END/STEP``  sweep bounds (mandatory with the variable)
"""

from __future__ import annotations

import configparser
import os
from collections import defaultdict
from pathlib import Path

import pytest

pytestmark = pytest.mark.hardware
pytest.importorskip("aioca")


class Docs:
    def __init__(self) -> None:
        self.docs: dict[str, list[dict]] = defaultdict(list)

    def __call__(self, name: str, doc: dict) -> None:
        self.docs[name].append(doc)

    @property
    def start(self) -> dict:
        return self.docs["start"][0]

    def primary_events(self) -> list[dict]:
        uids = {d["uid"] for d in self.docs["descriptor"] if d["name"] == "primary"}
        return [e for e in self.docs["event"] if e["descriptor"] in uids]


def _sweep_points(start: float, end: float, step: float) -> list[float]:
    n = int(round(abs(end - start) / abs(step))) + 1
    sign = 1.0 if end >= start else -1.0
    return [round(start + sign * i * abs(step), 6) for i in range(n)]


def _assert_prepared(docs: Docs, experiment: str, save_set: str) -> Path:
    """Every GEECS key a downstream reader needs, plus the ScanInfo ini."""
    start = docs.start
    assert start["bluesky_backend"] is True
    assert start["experiment"] == experiment
    assert isinstance(start["scan_number"], int) and start["scan_number"] > 0
    assert start["scan_id"] == start["scan_number"]
    assert start["save_sets"] == [save_set]
    assert "geecs_scalar_headers" in start
    folder = Path(start["scan_folder"])
    assert folder.is_dir(), folder
    ini = folder / f"ScanInfoScan{start['scan_number']:03d}.ini"
    assert ini.exists(), ini
    parser = configparser.ConfigParser()
    parser.read_string(ini.read_text())
    assert parser["Scan Info"]["scan no"].strip('"') == str(start["scan_number"])
    assert docs.docs["stop"][0]["exit_status"] == "success"
    return folder


@pytest.mark.hardware
def test_stock_plans_run_as_geecs_scans_on_hardware() -> None:
    """bp.count and bp.list_scan with md={"geecs": request}, against the machine."""
    import bluesky.plans as bp

    from geecs_bluesky.config_resolver import ConfigsRepoResolver
    from geecs_bluesky.namespace import GeecsNamespace
    from geecs_bluesky.preprocessors import (
        install_connect_on_demand,
        install_geecs_preamble,
    )
    from geecs_bluesky.session import GeecsSession
    from geecs_schemas import ScanRequest

    experiment = os.environ.get("GEECS_HW_EXPERIMENT", "Undulator")
    save_set = os.environ.get("GEECS_HW_SAVE_SET", "Amp4In")
    profile = os.environ.get("GEECS_HW_TRIGGER_PROFILE", "HTU-NoGas")
    shots = int(os.environ.get("GEECS_HW_SHOTS", "3"))
    sweep_target = os.environ.get("GEECS_HW_SCAN_VARIABLE")

    resolver = ConfigsRepoResolver(experiment)
    namespace = GeecsNamespace.from_experiment(experiment)
    # Subscribe what the worker startup subscribes, or the run never reaches
    # the Tiled catalog the data portal lists from and never exports an
    # s-file — the preprocessor path has to be indistinguishable downstream.
    session = GeecsSession(experiment, tiled=True)
    from geecs_bluesky.sfile_callback import SFileExportCallback

    session.RE.subscribe(SFileExportCallback())
    install_geecs_preamble(
        session.RE, session=session, resolver=resolver, namespace=namespace
    )
    install_connect_on_demand(session.RE)
    print(f"\nnamespace: {len(namespace)} devices; preamble + connect installed")

    # Which devices the save set names — the stock plan is handed exactly
    # those, which is what makes the preamble's configuration apply to the
    # objects the plan reads.
    entries = resolver.resolve_save_set(save_set).entries
    detectors = [namespace[e.device] for e in entries]
    print(f"save set {save_set}: {[d.name for d in detectors]}")

    def request(**kw) -> dict:
        base = dict(
            mode="noscan",
            shots_per_step=shots,
            acquisition="strict",
            save_sets=[save_set],
            trigger_profile=profile,
            description="#807 phase 2 hardware acceptance",
        )
        base.update(kw)
        return ScanRequest.model_validate(base).model_dump(mode="json")

    # ---- 1. a stock count as a noscan -------------------------------------
    docs = Docs()
    session.RE(bp.count(detectors, num=shots, md={"geecs": request()}), docs)
    folder = _assert_prepared(docs, experiment, save_set)
    events = docs.primary_events()
    assert len(events) == shots
    stamps = [v for k, v in events[0]["data"].items() if k.endswith("-acq_timestamp")]
    print(
        f"count: scan {docs.start['scan_number']} in {folder}, "
        f"{len(events)} events, plan_name={docs.start['plan_name']}"
    )
    print(f"  first-event shot stamps: {stamps}")
    print(f"  columns: {sorted(events[0]['data'])}")
    assert docs.start["plan_name"] == "count"

    if not sweep_target:
        pytest.skip("GEECS_HW_SCAN_VARIABLE unset — noscan half only")

    # ---- 2. a stock list_scan as a step scan -------------------------------
    # The request names the CATALOG variable (e.g. "S1H"); the plan needs the
    # movable. Resolving the catalog target here is also the identity check
    # that matters: the object the plan moves is the object the preamble
    # resolved the axis to.
    target = resolver.resolve_scan_variable(sweep_target)
    device_name, _, variable = str(target.target).partition(":")
    movable = namespace.variable(device_name, variable)
    print(f"catalog {sweep_target} -> {target.target} -> {movable.name}")
    points = _sweep_points(
        float(os.environ["GEECS_HW_SCAN_START"]),
        float(os.environ["GEECS_HW_SCAN_END"]),
        float(os.environ["GEECS_HW_SCAN_STEP"]),
    )
    # A direct CA read: the namespace device is connected lazily by
    # connect_on_demand when a *plan* touches it, so it is not connected yet.
    from geecs_bluesky.devices.ca.oneshot import try_caget_once
    from geecs_core.pv_naming import pv_name, setpoint_pv

    initial = float(
        try_caget_once(
            setpoint_pv(pv_name(experiment, device_name, variable)), timeout=5.0
        )
    )
    print(f"sweep {sweep_target} over {points} (pre-scan setpoint {initial})")
    docs2 = Docs()
    try:
        session.RE(
            bp.list_scan(
                detectors,
                movable,
                points,
                md={
                    "geecs": request(
                        mode="step",
                        axes=[
                            {
                                "variable": sweep_target,
                                "positions": {"values": points},
                            }
                        ],
                    )
                },
            ),
            docs2,
        )
    finally:
        from bluesky.plan_stubs import mv

        session.RE(mv(movable, initial))
        print(f"restored {sweep_target} to {initial}")

    folder2 = _assert_prepared(docs2, experiment, save_set)
    events2 = docs2.primary_events()
    assert len(events2) == len(points)
    readbacks = [e["data"][movable.position.name] for e in events2]
    tol = getattr(movable, "_tolerance", 0.05)
    for want, got in zip(points, readbacks):
        assert abs(got - want) <= tol + 1e-9, f"{got} vs {want}"
    print(
        f"scan: scan {docs2.start['scan_number']} in {folder2}, "
        f"{len(events2)} points, readbacks {readbacks}"
    )
    assert docs2.start["plan_name"] == "list_scan"
    assert docs2.start["scan_number"] == docs.start["scan_number"] + 1
