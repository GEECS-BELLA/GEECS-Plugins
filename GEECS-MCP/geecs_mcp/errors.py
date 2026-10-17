"""The structured result envelope every tool returns.

Tools never raise to the agent: every return is a JSON string with ``ok``
plus either the payload or ``{error_kind, message}``.  The services'
message text is preserved verbatim inside ``message`` — those strings are
the operator vocabulary, the same words the scanner page and the portal
show.

Error kinds — chosen so an agent can branch on the kind without parsing
``message``.  The scanner's own ``error.kind`` vocabulary is this one, so
its verdicts pass through unchanged; the portal's and the logbook's
status codes map onto it (:mod:`geecs_mcp.clients`):

- ``policy_refusal`` — the service refused in the current state: a
  preflight question not yet acknowledged (``needs_acknowledgement``),
  something already queued (``pending_items``), an idle-only item while
  a plan runs (``re_state`` / ``items_in_queue``), a halt with nothing
  running, an analysis job already active, a logbook version conflict.
  Retryable by changing the ask.
- ``invalid_request`` — schema/validation, a refused preflight, an
  argument this server checks (a blank operator); fix the request.
- ``not_found`` — a preset, run, artifact, entry or item that does not
  exist.
- ``service_unreachable`` — the scanner, portal or logbook did not
  answer, or answered with its own failure (5xx); ``service`` and ``url``
  say which.  The scanner's ``manager_unreachable`` (its 503 when the RE
  Manager is down) lands here with the scanner's message.
- ``internal_error`` — a bug in this server (the tools-never-raise
  backstop; the message carries the exception text for the log trail).

Retired with the HTTP-client rebuild (no producer left):
``manager_unreachable``, ``worker_refused``, ``tiled_unreachable``,
``task_timeout`` — the wait verbs' timeouts are ok results with
``finished: false``, never errors.
"""

from __future__ import annotations

import json
import math
from typing import Any

ERROR_KINDS = (
    "policy_refusal",
    "invalid_request",
    "not_found",
    "service_unreachable",
    "internal_error",
)


def _json_safe(value: Any) -> Any:
    """Recursively normalize non-finite floats to ``None``.

    The envelope contract is *strict* JSON: ``json.dumps`` writes bare
    ``NaN``/``Infinity`` tokens that strict consumers reject, and
    non-finite values arise routinely (statistics over sparse event
    rows, metadata echoed from upstream).  Enforced here — in the
    serializer, not at call sites — so no future payload field can
    regress it (codex review finding on the v0 PR).
    """
    if isinstance(value, float):  # bool is not float; numpy floats subclass it
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def make_ok(**payload: Any) -> str:
    """Serialize a success envelope: ``{"ok": true, **payload}``.

    ``default=str`` keeps odd scalar types (paths, datetimes) from
    crashing serialization — a readable string beats a tool error — and
    ``allow_nan=False`` backstops the :func:`_json_safe` normalization:
    a non-finite float that somehow evades it raises here, which the
    tool dispatch guard turns into an ``internal_error`` envelope
    instead of emitting invalid JSON.
    """
    return json.dumps({"ok": True, **_json_safe(payload)}, default=str, allow_nan=False)


def make_error(error_kind: str, message: str, **extra: Any) -> str:
    """Serialize a failure envelope: ``{"ok": false, error_kind, message}``.

    ``extra`` carries structured refusal context (``pending_items``) —
    same strict-JSON discipline as :func:`make_ok`.
    """
    if error_kind not in ERROR_KINDS:  # programmer error — fail loudly in tests
        raise ValueError(f"unknown error_kind {error_kind!r}")
    return json.dumps(
        {
            "ok": False,
            "error_kind": error_kind,
            "message": message,
            **_json_safe(extra),
        },
        default=str,
        allow_nan=False,
    )
