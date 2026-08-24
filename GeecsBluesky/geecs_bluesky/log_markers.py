"""Log-line contract strings that clients grep/parse — import-light on purpose.

A marker lives here (not beside the code that emits it) exactly when
out-of-process consumers parse it from a log/text stream: the emitting
module usually sits deep in the engine (bluesky, devices, aioca), and a
client that only *matches* the string must never pay for that stack —
the ``geecs_bluesky.qs_client`` package import is pinned light, and its
re-export of these markers is a plain eager import of this module.

This module may depend on nothing heavier than the standard library.
"""

from __future__ import annotations

#: The decision-4 failed-move pause reason line: the engine logs
#: ``f"{FAILED_MOVE_LOG_PREFIX}: <reason>"`` (ERROR) when a queue-plan
#: axis move fails and the scan pauses for the operator.  Stream
#: consumers (the console's paused pill, the MCP's ``scan_progress``)
#: match this prefix in the manager's console-output stream to surface
#: the *why*.  Emitted by ``plans/step_scan.py``; re-exported by
#: ``plans.pause_semantics`` (its historical home) and
#: ``geecs_bluesky.qs_client`` (the client-facing spelling).
FAILED_MOVE_LOG_PREFIX = "FAILED MOVE - pausing for operator"
