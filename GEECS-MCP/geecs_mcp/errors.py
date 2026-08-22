"""The structured result envelope every tool returns.

Tools never raise to the agent: every return is a JSON string with ``ok``
plus either the payload or ``{error_kind, message}``.  The engine's
message text is preserved verbatim inside ``message`` — those strings are
the operator vocabulary (the planning doc's error-taxonomy rule).

Error kinds (the planning doc's taxonomy, plus ``tiled_unreachable`` for
the archive's network failures, which are neither the manager's nor the
request's fault):

- ``policy_refusal`` — etiquette/cap/acknowledgement; retryable by
  changing the ask.
- ``invalid_request`` — schema/validation; fix the request.
- ``manager_unreachable`` — the RE Manager did not answer.
- ``worker_refused`` — the manager accepted the RPC, the worker said no.
- ``task_timeout`` — a bounded worker task did not finish in budget.
- ``not_found`` — results/configs that do not exist.
- ``tiled_unreachable`` — the archive did not answer.
- ``internal_error`` — a bug in this server (the tools-never-raise
  backstop; the message carries the exception text for the log trail).
"""

from __future__ import annotations

import json
import math
from typing import Any

ERROR_KINDS = (
    "policy_refusal",
    "invalid_request",
    "manager_unreachable",
    "worker_refused",
    "task_timeout",
    "not_found",
    "tiled_unreachable",
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

    ``extra`` carries structured refusal context (``pending_items``,
    ``needs_acknowledgement``) — same strict-JSON discipline as
    :func:`make_ok`.
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
