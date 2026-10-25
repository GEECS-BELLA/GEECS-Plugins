"""How fresh one value is: the rule every family's roles are judged by (Appendix A rule 1).

Pure. A value is fresh while its **producer** advances, never because the
value itself changed: NDStats posts a scalar only when it changes, so a
steady centroid sends nothing while its plugin's ``ArrayCounter_RBV``
ticks on every frame. A value whose ``Compute*`` flag is off is ``off``
(frozen, not measured) whatever the counter does. A value that is not
connected is ``unknown``, never ``stale`` (rule 9): disconnected is not
stale. A family names the producer and the flag per role (``fresh_by``,
``computed_by``); a role naming neither is fresh while connected.

The camera's per-kind page judges by this rule too
(:func:`geecs_panels.camera.judge`, which says ``disconnected`` for
``unknown``).
"""

from __future__ import annotations

from typing import Optional

from geecs_panels.contracts import Freshness


def judge_freshness(
    connected: bool,
    *,
    computed: Optional[tuple[bool, object]] = None,
    counter: Optional[tuple[bool, Optional[float]]] = None,
    stale_after: float,
) -> Freshness:
    """How fresh a value is now.

    Parameters
    ----------
    connected : bool
        Whether the value's own channel is connected.
    computed : (bool, object), optional
        The ``Compute*`` flag the value needs on: whether its channel is
        connected, and its value. Absent: the value needs none.
    counter : (bool, float or None), optional
        The producer's counter: whether its channel is connected, and
        seconds since it last advanced (``None``: not seen to advance).
        Absent: the value is fresh while connected (a setting).
    stale_after : float
        Seconds without an advance after which the value is stale.
    """
    if not connected:
        return "unknown"
    if computed is not None:
        flag_connected, flag = computed
        if not flag_connected:
            return "unknown"
        if not flag:
            return "off"
    if counter is None:
        return "fresh"
    counter_connected, age = counter
    if not counter_connected:
        return "unknown"
    if age is None or age > stale_after:
        return "stale"
    return "fresh"
