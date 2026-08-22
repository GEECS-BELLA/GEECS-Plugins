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
"""

from __future__ import annotations

import json
from typing import Any

ERROR_KINDS = (
    "policy_refusal",
    "invalid_request",
    "manager_unreachable",
    "worker_refused",
    "task_timeout",
    "not_found",
    "tiled_unreachable",
)


def make_ok(**payload: Any) -> str:
    """Serialize a success envelope: ``{"ok": true, **payload}``.

    ``default=str`` keeps odd scalar types (numpy numbers, paths,
    datetimes) from crashing serialization — a readable string beats a
    tool error.
    """
    return json.dumps({"ok": True, **payload}, default=str)


def make_error(error_kind: str, message: str) -> str:
    """Serialize a failure envelope: ``{"ok": false, error_kind, message}``."""
    if error_kind not in ERROR_KINDS:  # programmer error — fail loudly in tests
        raise ValueError(f"unknown error_kind {error_kind!r}")
    return json.dumps({"ok": False, "error_kind": error_kind, "message": message})
