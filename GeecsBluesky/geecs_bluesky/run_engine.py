"""One RunEngine with the GEECS pieces installed — the worker's and the tests'.

The successor of the deleted ``GeecsSession`` (phase 1 of the native-Bluesky
rebuild, ``Planning/native_bluesky/03_clean_room_rebuild.md`` §10.5): no
scan API of its own, just a stock :class:`~bluesky.run_engine.RunEngine`
with the GEECS preprocessors and callbacks installed.  Scans are the stock
``bluesky.plans`` verbs (bound strict by :mod:`geecs_bluesky.plans.registry`)
over the namespace's devices; everything per-run is the plan's arguments
and the devices' own lifecycles.

What ``claim=True`` installs (§4.C) — the GEECS scan, as three preprocessors
and three callbacks, each with one job:

- :func:`~geecs_bluesky.plans.claim_scan.claim_scan_preprocessor` — every
  run claims a scan number, its folder rides in the start document and
  the shared :class:`~geecs_bluesky.plans.claim_scan.GeecsScanPathProvider`
  points the native-saving detectors at it;
- :func:`~geecs_bluesky.preprocessors.scalar_headers` — the legacy
  ``Device Variable`` header map into the start document;
- :class:`~bluesky.preprocessors.SupplementalData` with *telemetry* as the
  baseline (read at open and close of every run, §4.B);
- the ScanInfo ini, the s-file and ``scan.log`` callbacks
  (:mod:`geecs_bluesky.callbacks`).

``connect_on_demand`` goes in last, outermost, so it also sees the messages
the other preprocessors inject.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

from bluesky import RunEngine
from bluesky.preprocessors import SupplementalData

from geecs_bluesky.plans.claim_scan import (
    GeecsScanPathProvider,
    claim_scan_preprocessor,
)
from geecs_bluesky.preprocessors import install_connect_on_demand, scalar_headers

logger = logging.getLogger(__name__)


def make_run_engine(
    *,
    experiment: str | None = None,
    mock: bool = False,
    tiled: bool = False,
    tiled_uri: str | None = None,
    tiled_api_key: str | None = None,
    claim: bool = False,
    path_provider: GeecsScanPathProvider | None = None,
    telemetry: Sequence[Any] = (),
) -> RunEngine:
    """Build the RunEngine every GEECS plan runs on.

    The namespace is not an argument: :func:`connect_on_demand` recognises
    namespace devices by their marker, and the startup profile binds them
    into its own module globals for the manager.

    Parameters
    ----------
    experiment :
        GEECS experiment name; required with *claim*.
    mock :
        Connect namespace devices with ophyd-async mock backends (hermetic
        tests).
    tiled :
        Subscribe the TiledWriter (:func:`~geecs_bluesky.tiled_integration.subscribe_tiled`):
        best-effort, skip-with-warning when the catalog is unreachable.
    tiled_uri, tiled_api_key :
        Explicit catalog location; the shared ``config.ini`` otherwise.
    claim :
        Every run is a GEECS scan: claim a scan number, write ScanInfo, the
        s-file and ``scan.log`` into its folder, point *path_provider* at it.
    path_provider :
        The provider the namespace's native-saving detectors hold (built
        by the caller, shared with the namespace).
    telemetry :
        Devices/signals read as the ``baseline`` stream at the open and
        close of every run (``GeecsNamespace.telemetry()``).

    Returns
    -------
    RunEngine
        ``context_managers=[]`` (no SIGINT handler: the worker runs the RE
        off the main thread), interruptions recorded.
    """
    RE = RunEngine(context_managers=[])
    RE.record_interruptions = True
    if tiled:
        from geecs_bluesky.tiled_integration import subscribe_tiled

        subscribe_tiled(RE, tiled_uri, tiled_api_key)
    if claim:
        if not experiment:
            raise ValueError("make_run_engine(claim=True) needs the experiment name")
        from geecs_bluesky.callbacks import subscribe_scan_outputs

        RE.preprocessors.append(
            partial(
                claim_scan_preprocessor,
                experiment=experiment,
                path_provider=path_provider,
            )
        )
        RE.preprocessors.append(scalar_headers)
        subscribe_scan_outputs(RE)
    if telemetry:
        RE.preprocessors.append(SupplementalData(baseline=list(telemetry)))
    # Installed LAST on purpose: connect_on_demand must be the OUTERMOST
    # preprocessor so it also sees messages later preprocessors inject
    # (SupplementalData baselines).  Re-run after appending anything else.
    install_connect_on_demand(RE, mock=mock)
    return RE
