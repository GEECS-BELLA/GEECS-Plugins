"""The GEECS RE Manager client seam — for every queueserver client.

Extracted from GEECS-Console (2026-08-21) because the console is just one
client of the queue: notebooks and the GEECS MCP submit the same
``geecs_schemas.ScanRequest`` dicts through the same verbs.  Two modules:

- :mod:`.client` — the :class:`QueueClient` protocol and its
  implementations (:class:`ZmqQueueClient` over the manager's 0MQ control
  socket, :class:`StubQueueClient` offline), plus the ``[qserver]``
  section reader of the shared ``~/.config/geecs_python_api/config.ini``.
- :mod:`.submit_preflight` — the client-side pre-submit checks (engine
  validation, unserved variables, CONNECTED liveness, free-run staleness)
  and :func:`stamp_submission`, which records client identity and check
  outcomes into ``ScanRequest.submission`` for run-metadata provenance.

Importing this package is deliberately light: ``bluesky-queueserver-api``
(the ``qs-client`` extra) and the engine/CA internals load lazily inside
methods, and the parent package's device re-exports are lazy too — a
client that only submits scans never pays for aioca/ophyd-async.
"""

# The failed-move pause-reason line clients parse from the manager's
# console-output stream (the console pill, the MCP's scan_progress).
# Its definition lives in the import-light log_markers module precisely
# so this eager re-export cannot violate the pinned light-import
# contract (the engine home, plans.pause_semantics, pulls bluesky).
from geecs_bluesky.log_markers import FAILED_MOVE_LOG_PREFIX
from geecs_bluesky.qs_client.client import (
    QserverConfig,
    QueueClient,
    QueueStatus,
    StubQueueClient,
    SubmitResult,
    ZmqQueueClient,
    make_queue_client,
    read_qserver_config,
)
from geecs_bluesky.qs_client.submit_preflight import (
    PreflightQuestion,
    PreflightReport,
    run_submit_preflight,
    stamp_submission,
)

__all__ = [
    "FAILED_MOVE_LOG_PREFIX",
    "QserverConfig",
    "QueueClient",
    "QueueStatus",
    "StubQueueClient",
    "SubmitResult",
    "ZmqQueueClient",
    "make_queue_client",
    "read_qserver_config",
    "PreflightQuestion",
    "PreflightReport",
    "run_submit_preflight",
    "stamp_submission",
]
