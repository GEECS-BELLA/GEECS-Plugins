"""The GEECS RE Manager client seam — for every queueserver client.

Extracted from GEECS-Console (2026-08-21) because the console is just one
client of the queue: notebooks and the GEECS MCP submit the same stock
plan items through the same verbs.  Three modules:

- :mod:`.client` — the :class:`QueueClient` protocol and its
  implementations (:class:`ZmqQueueClient` over the manager's 0MQ control
  socket, :class:`StubQueueClient` offline), the ``[qserver]``
  section reader of the shared ``~/.config/geecs_python_api/config.ini``,
  and :func:`readiness_verdict` — the ONE definition of "the manager can
  run the GEECS plans" (#793), shared by the pre-submit ``worker_ready``
  check, the ``geecs-qserver-ready`` service-start assertion, and any
  probe script.
- :mod:`.presets` — :func:`expand_preset`, a saved
  :class:`geecs_schemas.Preset` into the stock plan queue item it stands
  for (device names, scan-variable references, the run metadata).
- :mod:`.submit_preflight` — the client-side pre-submit checks (the
  preset expands, worker readiness — environment open + the plan
  allowed, #793 — CONNECTED liveness) and :func:`build_submission_record`,
  which records client identity and check outcomes into the
  ``SubmissionRecord`` submitted as run metadata
  (``submit_preset(preset, md={"geecs": {"submission": ...}})``) for
  provenance.

Importing this package is deliberately light: ``bluesky-queueserver-api``
(the ``qs-client`` extra) and the engine/CA internals load lazily inside
methods, and the parent package's device re-exports are lazy too — a
client that only submits scans never pays for aioca/ophyd-async.
"""

# The failed-move pause-reason line clients parse from the manager's
# console-output stream (the console pill, the MCP's scan_progress).
# Its definition lives in the import-light log_markers module precisely
# so this eager re-export cannot violate the pinned light-import
# contract (the historical engine home pulled bluesky).
from geecs_bluesky.log_markers import FAILED_MOVE_LOG_PREFIX
from geecs_bluesky.qs_client.client import (
    QserverConfig,
    QueueClient,
    QueueStatus,
    ReadinessVerdict,
    StubQueueClient,
    SubmitResult,
    ZmqQueueClient,
    make_queue_client,
    queue_status_from_manager,
    read_qserver_config,
    readiness_from_reads,
    readiness_verdict,
)
from geecs_bluesky.qs_client.presets import QueueItem, expand_preset
from geecs_bluesky.qs_client.submit_preflight import (
    PreflightQuestion,
    PreflightReport,
    build_submission_record,
    run_submit_preflight,
)

__all__ = [
    "FAILED_MOVE_LOG_PREFIX",
    "QserverConfig",
    "QueueClient",
    "QueueStatus",
    "ReadinessVerdict",
    "readiness_verdict",
    "readiness_from_reads",
    "queue_status_from_manager",
    "StubQueueClient",
    "SubmitResult",
    "ZmqQueueClient",
    "make_queue_client",
    "read_qserver_config",
    "QueueItem",
    "expand_preset",
    "PreflightQuestion",
    "PreflightReport",
    "build_submission_record",
    "run_submit_preflight",
]
