"""Hardware-runnable ScanRequest end-to-end test (integration-marked).

Skipped in CI (the suite runs ``-m "not integration ..."``); run it by hand
over the lab network / VPN to verify the M3b ScanRequest path against the
real gateway:

.. code-block:: bash

    cd GeecsBluesky
    EPICS_CA_ADDR_LIST=<gateway host> \\
    poetry run pytest tests/test_scan_request_hardware.py -m integration -s

Prerequisites
-------------
- The GeecsCAGateway serving the experiment's PVs (``EPICS_CA_ADDR_LIST``
  set, or ``[epics] ca_addr_list`` in the shared config.ini).
- The configs repository resolvable (``GEECS_SCANNER_CONFIG_DIR`` env var or
  config.ini) — the save set and trigger profile are loaded from it via
  :class:`~geecs_bluesky.scan_request_runner.ConfigsRepoResolver`, converting
  the legacy corpus files on the fly.
- The NetApp data tree mounted if you want a scan number/folder claimed
  (otherwise the run proceeds with numbering disabled and a warning).

Every name is parameterizable via env vars so the coordinator can point it
at whatever is alive that day:

===========================  =======================================
``GEECS_HW_EXPERIMENT``      experiment name (default ``Undulator``)
``GEECS_HW_CAMERA``          save-set name — a corpus save element /
                             save set file stem under ``save_devices/``
                             (default ``Amp4In``)
``GEECS_HW_TRIGGER_PROFILE`` trigger profile / shot-control config file
                             stem (default ``HTU-LaserOFF`` — the
                             corpus laser-off config: internal
                             single-shot source, safe without beam)
``GEECS_HW_ACQUISITION``     ``strict`` (default) or ``free_run``
``GEECS_HW_SHOTS``           shots to take (default ``5``)
``GEECS_HW_REP_RATE``        machine rep rate in Hz (default ``1``)
===========================  =======================================
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("aioca")


@pytest.mark.integration
def test_scan_request_noscan_runs_on_hardware() -> None:
    """One noscan ScanRequest through session.run() against the real gateway.

    Exercises the full M3b request path end to end: ConfigsRepoResolver
    loading (and legacy-converting) the corpus save set + trigger profile,
    the TriggerProfile → ordered-writes adapter, ShotController over the
    gateway ``:SP`` PVs, device building from the save set, scan-number
    claiming, acquisition, and the s-file export — with zero
    NotImplementedErrors.
    """
    from geecs_bluesky.scan_request_runner import ConfigsRepoResolver
    from geecs_bluesky.session import GeecsSession
    from geecs_schemas import ScanRequest

    experiment = os.environ.get("GEECS_HW_EXPERIMENT", "Undulator")
    # One or more save sets, comma-separated — the engine unions their devices
    # (M4 multi-save-set). Default is a single set, so single-set runs are
    # unchanged; set e.g. GEECS_HW_CAMERA="Amp4In,AuxDiagnostics" to exercise
    # the union on hardware.
    save_sets = [
        name.strip()
        for name in os.environ.get("GEECS_HW_CAMERA", "Amp4In").split(",")
        if name.strip()
    ]
    trigger_profile = os.environ.get("GEECS_HW_TRIGGER_PROFILE", "HTU-LaserOFF")
    acquisition = os.environ.get("GEECS_HW_ACQUISITION", "strict")
    shots = int(os.environ.get("GEECS_HW_SHOTS", "5"))
    rep_rate_hz = float(os.environ.get("GEECS_HW_REP_RATE", "1"))

    request = ScanRequest.model_validate(
        {
            "mode": "noscan",
            "shots_per_step": shots,
            "acquisition": acquisition,
            "save_sets": save_sets,
            "trigger_profile": trigger_profile,
            "description": "M4 hardware verification (test_scan_request_hardware)",
        }
    )
    resolver = ConfigsRepoResolver(experiment)
    session = GeecsSession(experiment, rep_rate_hz=rep_rate_hz)

    uid = session.run(request, resolver)

    # A uid proves the RunEngine completed the run; None means the run was
    # not persisted (e.g. Tiled unreachable), which is still a pass for the
    # acquisition path — the RunEngine would have raised on failure.
    print(f"\nScanRequest hardware run complete; run uid: {uid}")
