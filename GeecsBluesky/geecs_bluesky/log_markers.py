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
#: <reason>"`` (ERROR), which stream consumers (the scanner, the MCP)
#: match in the manager's console-output stream.  No engine code emits it
#: today; the clients keep matching it, so the spelling is pinned here and
#: re-exported by ``geecs_bluesky.qs_client``.
FAILED_MOVE_LOG_PREFIX = "FAILED MOVE - pausing for operator"

#: An interlock's trip and clear lines, ``f"{INTERLOCK_TRIPPED_LOG_PREFIX}:
#: <name>: <reason>"`` and ``f"{INTERLOCK_CLEARED_LOG_PREFIX}: <name>"``
#: (WARNING, :mod:`geecs_bluesky.interlocks`), which the progress stream
#: matches in the manager's console output to show why a run paused.
INTERLOCK_TRIPPED_LOG_PREFIX = "INTERLOCK TRIPPED"
INTERLOCK_CLEARED_LOG_PREFIX = "INTERLOCK CLEARED"
