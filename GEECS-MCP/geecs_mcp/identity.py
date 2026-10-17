"""The one identity string every write carries.

``agent, admin supervision (<operator>)`` — composed here from the
``operator`` argument every write verb requires, so a scan, a move and a
logbook entry all trace back to the person supervising the agent.  A
per-shift name is an argument, not a deployment value (DESIGN.md §6).
"""

from __future__ import annotations

#: The logbook's ``author`` field caps at 120 characters; the scanner's
#: ``operator`` is free text.  Keep the composed string under the cap.
MAX_IDENTITY_LENGTH = 120
_PREFIX = "agent, admin supervision ("


class InvalidArgument(ValueError):
    """An argument this server checks itself — the guard's ``invalid_request``.

    A subclass so the guard maps exactly these, never every ``ValueError``
    an impl might raise (a non-finite float in a payload, say, is a bug).
    """


def identity_for(operator: str) -> str:
    """``agent, admin supervision (<operator>)``, or ``InvalidArgument`` for a blank name."""
    name = " ".join(str(operator or "").split())
    if not name:
        raise InvalidArgument(
            "operator is required: the name of the person supervising this "
            "agent, as the submission and the logbook entry will record it"
        )
    composed = f"{_PREFIX}{name})"
    if len(composed) > MAX_IDENTITY_LENGTH:
        raise InvalidArgument(
            f"operator name too long: the identity string must stay within "
            f"{MAX_IDENTITY_LENGTH} characters"
        )
    return composed
