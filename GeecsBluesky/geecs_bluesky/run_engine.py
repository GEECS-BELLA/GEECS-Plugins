"""One RunEngine with the GEECS pieces installed — the worker's and the tests'.

The successor of the deleted ``GeecsSession`` (phase 1 of the native-Bluesky
rebuild, ``Planning/native_bluesky/03_clean_room_rebuild.md`` §10.5): no
scan API of its own, just a stock :class:`~bluesky.run_engine.RunEngine`
with the device namespace exported, :func:`~geecs_bluesky.preprocessors.connect_on_demand`
installed outermost, and the run callbacks subscribed.  Scans are the stock
``bluesky.plans`` verbs over the namespace's devices; everything per-run is
the plan's arguments and the devices' own lifecycles.
"""

from __future__ import annotations

import logging
from bluesky import RunEngine

from geecs_bluesky.preprocessors import install_connect_on_demand

logger = logging.getLogger(__name__)


def make_run_engine(
    *,
    mock: bool = False,
    tiled: bool = False,
    tiled_uri: str | None = None,
    tiled_api_key: str | None = None,
    sfile: bool = False,
) -> RunEngine:
    """Build the RunEngine every GEECS plan runs on.

    The namespace is not an argument: :func:`connect_on_demand` recognises
    namespace devices by their marker, and the startup profile binds them
    into its own module globals for the manager.

    Parameters
    ----------
    mock :
        Connect namespace devices with ophyd-async mock backends (hermetic
        tests).
    tiled :
        Subscribe the TiledWriter (:func:`~geecs_bluesky.tiled_integration.subscribe_tiled`):
        best-effort, skip-with-warning when the catalog is unreachable.
    tiled_uri, tiled_api_key :
        Explicit catalog location; the shared ``config.ini`` otherwise.
    sfile :
        Subscribe the legacy s-file export at every stop document (exported
        from Tiled, so it needs *tiled* to do anything).

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
    if sfile:
        from geecs_bluesky.sfile_callback import SFileExportCallback

        RE.subscribe(SFileExportCallback())
    # Installed LAST on purpose: connect_on_demand must be the OUTERMOST
    # preprocessor so it also sees messages later preprocessors inject
    # (SupplementalData baselines).  Re-run after appending anything else.
    install_connect_on_demand(RE, mock=mock)
    return RE
