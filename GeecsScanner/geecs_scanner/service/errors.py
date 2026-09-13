"""The one error the service raises, and how the web layer renders it.

A taxonomy, not a status code: the service does not know it is behind
HTTP.  :data:`HTTP_STATUS` is the one mapping the web layer applies, so a
kind means the same thing on every route.
"""

from __future__ import annotations

from typing import Any

#: Error kinds → HTTP status.  ``policy_refusal`` is the operator-facing
#: "not now / not like this" (an unacknowledged preflight question, a
#: verb refused while a plan runs); ``invalid_request`` is a document
#: that cannot become a plan call at all.
HTTP_STATUS: dict[str, int] = {
    "invalid_request": 400,
    "not_found": 404,
    "policy_refusal": 409,
    "manager_unreachable": 503,
    "task_timeout": 504,
    "internal_error": 500,
}


class ScannerError(Exception):
    """A refusal or failure the service wants the caller to see verbatim.

    Parameters
    ----------
    kind : str
        One of :data:`HTTP_STATUS`'s keys.
    message : str
        The operator-facing sentence.
    **extra :
        Structured detail the page renders (``needs_acknowledgement``,
        ``pending_items``, …).
    """

    def __init__(self, kind: str, message: str, **extra: Any) -> None:
        if kind not in HTTP_STATUS:
            raise ValueError(f"unknown error kind {kind!r}")
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.extra = extra

    def to_payload(self) -> dict[str, Any]:
        """The JSON body the web layer sends: ``{"error": {...}}``."""
        return {"error": {"kind": self.kind, "message": self.message, **self.extra}}

    @property
    def status_code(self) -> int:
        """The HTTP status for this kind."""
        return HTTP_STATUS[self.kind]
