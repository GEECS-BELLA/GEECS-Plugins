"""bluesky-queueserver RE Manager startup profile for a GEECS worker.

Loaded by ``start-re-manager --startup-dir <this directory>`` (see
``launch_re_manager.sh``).  Defines the module-level ``RE`` the manager
keeps alive across queue items (``--keep-re`` — see ``qserver/README.md``'s
Troubleshooting section for the silent-bounce failure mode without it),
exports every device of the experiment as a noun
(:class:`~geecs_bluesky.namespace.GeecsNamespace`) and registers the stock
``bluesky.plans`` verbs (:data:`~geecs_bluesky.plan_names.GEECS_PLAN_NAMES`)
over them with the strict ``take_reading`` pre-bound
(:mod:`geecs_bluesky.plans.registry`) — ``count([UC_Amp4_IR_input], 10)``,
``scan([UC_Amp4_IR_input], U_S1H.current, -1, 1, 5, shots_per_step=10)``,
``mv(U_S1H.current, 0)``.  Every run claims a GEECS scan number and leaves
ScanInfo, the s-file, ``scan.log`` and the detectors' native files in its
folder; every subscribed scalar of the experiment rides in the run as the
baseline stream (``make_run_engine``).

Import order is load-bearing
-----------------------------
``geecs_bluesky`` is imported **first**, before anything that might pull in
``aioca``.  Its ``__init__`` calls
:func:`~geecs_bluesky.epics_env.apply_epics_address_config`, which sets
``EPICS_CA_ADDR_LIST``/``EPICS_CA_AUTO_ADDR_LIST`` from
``~/.config/geecs_python_api/config.ini``'s ``[epics]`` section *before* the
device imports create libca's CA context — libca reads that env var once, at
context creation, and never again.  A gateway address sourced from the GEECS
database instead of the config file would need a DB round trip at import
time (a network hazard this early) and would be circular besides (the
database itself is one of the devices CA reaches through the gateway).
config-file/systemd-env sourcing is deliberate, not a placeholder.

Experiment resolution
----------------------
``QS_EXPERIMENT`` wins when set (the natural queueserver/systemd knob —
one worker process per experiment); otherwise falls back to
``config.ini``'s ``[Experiment] expt`` via ``GeecsPathsConfig`` (the same
default every other headless entry point in this repo uses).  Neither
present is a startup-time configuration error, not a runtime one: fail
loud here rather than have every submitted plan fail identically later.

``QS_DEVICE_NAMESPACE=off`` is the hermetic switch (tests, a box without DB
or data-share reach): no namespace, no trigger profiles, no scan claim —
the plans are registered but refuse to run.
"""

from __future__ import annotations

import logging
import os

# Must import geecs_bluesky before anything that could pull in aioca — see
# the module docstring above.
import geecs_bluesky  # noqa: F401

from geecs_bluesky.config_resolver import ConfigsRepoResolver
from geecs_bluesky.namespace import GeecsNamespace
from geecs_bluesky.plan_names import GEECS_PLAN_NAMES
from geecs_bluesky.plans.claim_scan import GeecsScanPathProvider
from geecs_bluesky.plans.registry import TriggerProfiles, bind_strict_plans
from geecs_bluesky.run_engine import make_run_engine

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
_hermetic = os.environ.get("QS_DEVICE_NAMESPACE", "db").strip().lower() == "off"

# ── Device namespace (GEECS-Plugins#807) ──────────────────────────────────
# Every enabled device of the experiment as a long-lived ophyd-async noun —
# built from the GEECS DB (loud on failure), connected on first use by the
# connect_on_demand preprocessor make_run_engine installs outermost.  The
# native-saving detectors share one path provider the claim preprocessor
# points at each run's folder.
_path_provider = GeecsScanPathProvider()
_DEVICE_NAMES: list[str] = []
_telemetry: list = []
if _hermetic:
    _profiles = TriggerProfiles({})
else:
    namespace = GeecsNamespace.from_experiment(
        _experiment, path_provider=_path_provider
    )
    _DEVICE_NAMES = namespace.export_into(globals())
    _telemetry = namespace.telemetry()
    # The trigger profiles (one ShotControl each) a plan's trigger_profile
    # argument resolves against; the experiment default from
    # experiment_defaults.yaml.
    _profiles = TriggerProfiles.from_resolver(
        ConfigsRepoResolver(_experiment), experiment=_experiment
    )

# The manager's --keep-re contract needs a top-level `RE` in this module's
# namespace.  tiled=True: the [tiled] config mechanism
# (geecs_bluesky.tiled_integration.subscribe_tiled) subscribes a TiledWriter
# — best-effort, skip-with-log if the catalog is unreachable.  claim=True:
# every run is a GEECS scan (number, folder, ScanInfo, s-file, scan.log).
RE = make_run_engine(
    experiment=_experiment,
    tiled=True,
    claim=not _hermetic,
    path_provider=_path_provider,
    telemetry=_telemetry,
)

# The plans the manager discovers (every generator function in this
# namespace is a plan to it — profile_ops.plans_from_nspace): the stock
# verbs bound strict, under their own names, pinned by
# geecs_bluesky.plan_names (the readiness check asserts them).  Never import
# a stray generator into this module.
globals().update(bind_strict_plans(_profiles))

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

__all__ = ["RE", *GEECS_PLAN_NAMES, *_DEVICE_NAMES]
