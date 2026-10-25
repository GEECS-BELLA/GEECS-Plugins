"""The scope contract's rules (``scope_contract``): its capture as one trace per channel.

The capture arrives as one ``ARR:ArrayData`` array, channels varying
fastest, read with its time axis and shape in one request (rule 3, read
on request); :func:`geecs_panels.scope_feed.traces_payload` splits it by
the channels the model has and min/max-decimates each against time, so a
one-sample spike survives. A disabled channel's all-NaN row is left out.
"""

from __future__ import annotations

from collections.abc import Mapping

from geecs_panels.bridge.arrays import Trace
from geecs_panels.bridge.feed import Feed
from geecs_panels.bridge.rules import Rules
from geecs_panels.scope import CHANNELS
from geecs_panels.scope_feed import traces_payload


def arrays(
    feed: Feed, read: Mapping[str, list[float]], columns: int
) -> dict[str, Trace]:
    """Each present channel's trace against the time axis (seconds)."""
    present = [
        ch
        for ch in CHANNELS
        if feed.device.has_role(f"ch_{ch.lower()}_present")
        and feed.value(f"ch_{ch.lower()}_present") == "Present"
    ]
    built = traces_payload(
        {
            "t": list(read.get("time_axis", [])),
            "data": list(read.get("data", [])),
            "channels": list(read.get("arr_channels", [])),
            "samples": list(read.get("arr_samples", [])),
        },
        present,
        columns=columns,
    )
    traces = built["traces"]
    assert isinstance(traces, dict)
    return {ch: Trace(x=t["t"], y=t["v"]) for ch, t in traces.items()}


RULES = Rules(arrays=arrays)
