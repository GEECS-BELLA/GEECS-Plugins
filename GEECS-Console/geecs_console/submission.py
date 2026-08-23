"""The submission seam: the console's identity over the shared queue client.

The RE Manager client machinery lives in :mod:`geecs_bluesky.qs_client`
since the extraction (2026-08-21) — the console is one peer client of the
queue among several (notebooks, the GEECS MCP submit the same
``ScanRequest`` dicts through the same verbs).  What stays here is the
console's part:

- :data:`Submitter` — the console's historical name for the ONE client
  protocol, :class:`geecs_bluesky.qs_client.QueueClient` (the extraction
  collapsed the former twin protocols into it; window/controller type
  hints and docstrings keep the name).
- :func:`make_queue_submitter` — the default factory, building the shared
  client with the **console's** submitted-as identity (offline it returns
  the stub client, whose every verb refuses with the missing-``[qserver]``
  message).

Scan *state* is deliberately not on the protocol: the window observes it
from the manager status poll and the document stream
(:class:`~geecs_console.app.scan_monitor.ScanMonitorController`), never by
asking the submitter.  Threading contract: every member blocks (0MQ round
trips; manual moves poll a worker task to completion), so the window and
controllers dispatch them through their ``BackgroundResult`` workers —
with two deliberate exceptions, ``request_pause`` and ``request_resume``,
which are single short-timeout requests the window calls directly.
"""

from __future__ import annotations

from geecs_bluesky.qs_client import QueueClient, make_queue_client

#: The console's name for the one queue-client protocol.
Submitter = QueueClient


def make_queue_submitter(experiment: str = "") -> Submitter:
    """Build the shared queue client with the console's identity.

    Parameters
    ----------
    experiment : str, optional
        The selected experiment.  Currently informational — one manager
        serves one experiment by deployment contract (``QS_EXPERIMENT``),
        and the ``[qserver]`` config names that manager; a mismatch
        surfaces as the worker refusing the request's names at validation.

    Returns
    -------
    Submitter
        Ready to use; unconfigured installs get the stub client (every
        verb refuses with the missing-config message) and no stream
        addresses.
    """
    return make_queue_client(experiment, user="geecs-console")
