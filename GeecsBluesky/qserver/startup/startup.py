"""bluesky-queueserver RE Manager startup profile for a GEECS worker.

Loaded by ``start-re-manager --startup-dir <this directory>`` (see
``launch_re_manager.sh``). Defines the module-level ``RE`` the manager keeps
alive across queue items (``--keep-re`` — see ``qserver/README.md``'s
Troubleshooting section for the silent-bounce failure mode without it) and
registers :func:`~geecs_bluesky.plans.scan_request_plan.geecs_scan_request_plan`,
the one plan every ``ScanRequest`` (step, noscan, optimize) runs through.

Import order is load-bearing
-----------------------------
``geecs_bluesky`` is imported **first**, before anything that might pull in
``aioca``. Its ``__init__`` calls
:func:`~geecs_bluesky.epics_env.apply_epics_address_config`, which sets
``EPICS_CA_ADDR_LIST``/``EPICS_CA_AUTO_ADDR_LIST`` from
``~/.config/geecs_python_api/config.ini``'s ``[epics]`` section *before* the
device imports create libca's CA context — libca reads that env var once, at
context creation, and never again. A gateway address sourced from the GEECS
database instead of the config file would need a DB round trip at import
time (a network hazard this early) and would be circular besides (the
database itself is one of the devices CA reaches through the gateway).
config-file/systemd-env sourcing is deliberate, not a placeholder.

Experiment resolution
----------------------
``QS_EXPERIMENT`` wins when set (the natural queueserver/systemd knob —
one worker process per experiment); otherwise falls back to
``config.ini``'s ``[Experiment] expt`` via ``GeecsPathsConfig`` (the same
default every other headless entry point in this repo uses). Neither
present is a startup-time configuration error, not a runtime one: fail
loud here rather than have every submitted plan fail identically later.
"""

from __future__ import annotations

import logging
import os

# Must import geecs_bluesky before anything that could pull in aioca — see
# the module docstring above.
import geecs_bluesky  # noqa: F401

from geecs_bluesky.plans.scan_request_plan import (
    geecs_scan_request_plan,
    set_optimization_loader,
    set_plan_session,
)
from geecs_bluesky.session import GeecsSession
from geecs_bluesky.sfile_callback import SFileExportCallback

logger = logging.getLogger(__name__)


def _resolve_experiment() -> str:
    """``QS_EXPERIMENT`` env, falling back to ``config.ini``'s ``[Experiment]``.

    Raises
    ------
    RuntimeError
        Neither source yields a name — fail loud at worker startup.
    """
    experiment = os.environ.get("QS_EXPERIMENT")
    if experiment:
        return experiment

    from geecs_data_utils import GeecsPathsConfig

    experiment = GeecsPathsConfig().experiment
    if experiment:
        return experiment

    raise RuntimeError(
        "No GEECS experiment configured for this worker: set QS_EXPERIMENT "
        "or configure [Experiment] expt in "
        "~/.config/geecs_python_api/config.ini"
    )


_experiment = _resolve_experiment()

# tiled=True (default): GeecsSession's own [tiled] config mechanism
# (geecs_bluesky.tiled_integration.subscribe_tiled) subscribes a TiledWriter
# to session.RE — best-effort, skip-with-log if the catalog is unreachable,
# the same posture every other headless GeecsSession gets. Reused as-is
# (not reimplemented) because session.RE, not a second RunEngine, is the
# one this profile hands to the manager below.
session = GeecsSession(_experiment, tiled=True)

# The manager's --keep-re contract needs a top-level `RE` in this module's
# namespace; the plan preamble's own comment (scan_request_plan.py) requires
# running on `session.RE` specifically (a different RunEngine is
# unsupported), so the two must be the same object rather than two
# independently-constructed RunEngines.
RE = session.RE

set_plan_session(session)

# Best-effort legacy scalar (s-file) export on every completed run — #635.
RE.subscribe(SFileExportCallback())

# scan.log's root-logger attach + pre-claim buffer (GeecsBluesky 0.51.0,
# geecs_bluesky/scan_log.py) needs no wiring here: geecs_scan_request_plan
# itself calls begin_pre_scan_capture() at submission and the scan_log(...)
# context manager attaches the per-scan file handler directly to the root
# logger at the claim. The doc's "scan-log root capture" worker-startup
# build item is this existing mechanism relocated, not new design (see
# Planning/cutover_strategy/02_queueserver_migration.md, "Amendments from
# the console test-scan review").

# Optimize-mode ScanRequests: registered only when the `optimize` extra's
# heavy deps (xopt, ScanAnalysis) are importable — a headless worker without
# them refuses optimize-mode requests loudly at the plan (see
# set_optimization_loader's docstring) instead of failing mid-scan.
from geecs_bluesky.optimization.worker_loader import (  # noqa: E402
    make_worker_optimization_loader,
    warm_up_optimization_stack,
)

_optimization_loader = make_worker_optimization_loader()
set_optimization_loader(_optimization_loader)

# Pre-import the stack's heavy modules (torch/botorch/xopt) off-thread now,
# so the cold-import cost is paid here rather than freezing the worker's
# first optimize-mode request (mirrors GEECS-Console's main.py warm-up).
if _optimization_loader is not None:
    warm_up_optimization_stack()

__all__ = ["RE", "geecs_scan_request_plan"]
