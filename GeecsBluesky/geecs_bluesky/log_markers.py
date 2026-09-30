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

#: The failed-move pause reason line, ``f"{FAILED_MOVE_LOG_PREFIX}:
#: <reason>"`` (ERROR), which stream consumers (the scanner's progress
#: stream, the MCP's ``scan_progress``) match in the manager's
#: console-output stream to surface the *why* of a pause.  No engine code
#: emits it today — the plan that paused on a failed axis move went with
#: the pre-rebuild scan path — and the clients keep matching it, so the
#: spelling is pinned here; re-exported by ``geecs_bluesky.qs_client``
#: (the client-facing spelling).
FAILED_MOVE_LOG_PREFIX = "FAILED MOVE - pausing for operator"
