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
    geecs_run_action_plan,
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

# ZMQ document publisher — the GUI progress stream (#648). bluesky documents
# go to a bluesky-0MQ-proxy (started by launch_re_manager.sh alongside
# Redis); clients (GEECS-Console) consume them with
# bluesky.callbacks.zmq.RemoteDispatcher on the proxy's out port. NOTE the
# manager's --zmq-publish-console stream is a different thing entirely
# (captured stdout/stderr text, not documents). Best-effort, same posture
# as Tiled: a worker without the stream still runs scans correctly — only
# live GUI progress is lost. A zmq PUB connect always "succeeds" (it is
# asynchronous and simply drops while unconnected), so an absent proxy is
# probed explicitly below — that TCP check is what makes the warning real
# (#652 review finding 2).
_doc_publish_addr = os.environ.get("QS_DOC_PUBLISH_ADDR", "localhost:5567")
if _doc_publish_addr.upper() != "OFF":
    try:
        import socket

        from bluesky.callbacks.zmq import Publisher

        _doc_host, _, _doc_port = _doc_publish_addr.rpartition(":")
        try:
            socket.create_connection(
                (_doc_host or "localhost", int(_doc_port)), timeout=1.0
            ).close()
        except OSError:
            logger.warning(
                "No document proxy listening at %s — GUI progress streams "
                "will be empty until one appears (zmq reconnects on its "
                "own; set QS_DOC_PUBLISH_ADDR, or OFF to silence this)",
                _doc_publish_addr,
            )
        RE.subscribe(Publisher(_doc_publish_addr))
        logger.info("Publishing documents to 0MQ proxy at %s", _doc_publish_addr)
    except Exception:
        logger.warning(
            "Could not set up the document publisher for %s — GUI progress "
            "streams will be empty (set QS_DOC_PUBLISH_ADDR, or OFF to "
            "silence this)",
            _doc_publish_addr,
            exc_info=True,
        )


def geecs_move_variable(name: str, value: float) -> dict:
    """Manually move a scan variable — the console Movable panel's verb (#648).

    Executed via the manager's ``function_execute`` API. **Foreground**
    execution (``run_in_background=False``, the client default) requires a
    fully idle manager — the queueserver enforcement of the session's own
    "scan in progress — move not started" refusal; background execution
    bypasses that gate, which is why both GEECS queue plans check the
    session's manual-move lock before starting. Not a plan on purpose:
    :meth:`~geecs_bluesky.session.GeecsSession.move_variable` moves outside
    the RunEngine (the RE stays idle, guarded by the session's manual-move
    lock), so wrapping it in a queue item would change its semantics.

    Parameters
    ----------
    name : str
        Catalog scan-variable name or raw ``Device:Variable``.
    value : float
        Target value, in the variable's own units.

    Returns
    -------
    dict
        ``move_variable``'s summary: ``{variable, kind, value, targets}``.
    """
    return session.move_variable(name, value)


def geecs_describe_action(name: str) -> list[dict]:
    """Dry-run a named ActionPlan against *this worker's* configs (#648).

    Executed via ``function_execute`` (foreground calls require an idle
    manager). Pure config resolution — no CA, no execution. Serving it
    worker-side means the preview describes what the worker would actually
    run, even when a client's configs checkout has drifted.

    Parameters
    ----------
    name : str
        Action-plan name in the experiment's action library.

    Returns
    -------
    list of dict
        One dict per concrete step, in execution order (see
        :meth:`~geecs_bluesky.session.GeecsSession.describe_action`).
    """
    return session.describe_action(name)


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

__all__ = [
    "RE",
    "geecs_scan_request_plan",
    "geecs_run_action_plan",
    "geecs_move_variable",
    "geecs_describe_action",
]
