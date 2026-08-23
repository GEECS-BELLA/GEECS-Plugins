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
