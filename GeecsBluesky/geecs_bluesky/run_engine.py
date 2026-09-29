"""One RunEngine with the GEECS pieces installed — the worker's and the tests'.

The successor of the deleted ``GeecsSession`` (phase 1 of the native-Bluesky
rebuild): no
scan API of its own, just a stock :class:`~bluesky.run_engine.RunEngine`
with the GEECS preprocessors and callbacks installed.  Scans are the stock
``bluesky.plans`` verbs (bound strict by :mod:`geecs_bluesky.plans.registry`)
over the namespace's devices; everything per-run is the plan's arguments
and the devices' own lifecycles.

What ``claim=True`` installs — the GEECS scan, as two preprocessors and
three callbacks, each with one job:

- :func:`~geecs_bluesky.plans.claim_scan.claim_scan_preprocessor` — every
  run claims a scan number, its folder rides in the start document and
  the shared :class:`~geecs_bluesky.plans.claim_scan.GeecsScanPathProvider`
  points the native-saving detectors at it;
- :func:`~geecs_bluesky.preprocessors.scalar_headers` — the legacy
  ``Device Variable`` header map into the start document;
- the ScanInfo ini, the s-file and ``scan.log`` callbacks
  (:mod:`geecs_bluesky.callbacks`).

The experiment's background telemetry is not the engine's: every bound
scan verb reads it into its own rows
(:class:`~geecs_bluesky.devices.background.BackgroundSnapshot`, wired by
the registry).  There is no run-level baseline stream any more — the
open/close ``SupplementalData`` baseline was strict for every member and
one PV the gateway did not serve failed every scan after its claim
(GEECS-Plugins#1016).

``connect_on_demand`` goes in last, outermost, so it also sees the messages
the other preprocessors inject.
"""

from __future__ import annotations

import logging
from functools import partial

from bluesky import RunEngine

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
    claim: bool = False,
    path_provider: GeecsScanPathProvider | None = None,
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
        Subscribe the Tiled document spool
        (:func:`~geecs_bluesky.tiled_integration.subscribe_tiled_spool`):
        every run's documents to one file the ``geecs-tiled-writer``
        service registers off the engine thread.  On only when
        ``config.ini`` names a catalog; the engine never reaches Tiled.
    claim :
        Every run is a GEECS scan: claim a scan number, write ScanInfo, the
        s-file and ``scan.log`` into its folder, point *path_provider* at it.
    path_provider :
        The provider the namespace's native-saving detectors hold (built
        by the caller, shared with the namespace).

    Returns
    -------
    RunEngine
        ``context_managers=[]`` (no SIGINT handler: the worker runs the RE
        off the main thread), interruptions recorded.
    """
    RE = RunEngine(context_managers=[])
    RE.record_interruptions = True
    if tiled:
        from geecs_bluesky.tiled_integration import subscribe_tiled_spool

        subscribe_tiled_spool(RE)
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
        # Kept on the RunEngine so a shutdown (or a caller that wants the
        # s-file on disk before it moves on) can wait for the stack reads
        # and the joined s-file write, which finish on their own threads.
        RE.geecs_scan_outputs = subscribe_scan_outputs(RE)  # type: ignore[attr-defined]
    # Installed LAST on purpose: connect_on_demand must be the OUTERMOST
    # preprocessor so it also sees messages later preprocessors inject.
    # Re-run after appending anything else.
    install_connect_on_demand(RE, mock=mock)
    return RE
