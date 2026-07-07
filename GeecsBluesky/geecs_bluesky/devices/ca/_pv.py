"""Gateway PV names pinned to the Channel Access transport.

ophyd-async selects the EPICS transport for *un-prefixed* PV names by import
luck (``ophyd_async.epics.core._signal``): a successful ``p4p`` import sets
the module default to PVA, then a successful ``aioca`` import sets it back to
CA.  In an environment with ``p4p`` installed but ``aioca`` missing, every
un-prefixed PV silently flips to PVA and each connect against the CA-only
GeecsCAGateway times out with a generic connection error.

Prefixing PV strings with ``ca://`` pins the transport explicitly:
``split_protocol_from_pv`` routes the signal to ``CaSignalBackend`` regardless
of what is importable, and with ``aioca`` missing signal construction fails
loudly ("Protocol ca not available, did you `pip install ophyd_async[ca]`?")
instead of timing out per PV.

The prefix never leaks into event keys or source strings: ophyd-async strips
it before the backend stores the PV, and ``CaSignalBackend.source`` re-derives
the ``ca://…`` source string either way.
"""

from __future__ import annotations

from geecs_ca_gateway.pv_naming import pv_name

#: Explicit Channel Access scheme understood by
#: ``ophyd_async.epics.core._signal.split_protocol_from_pv``.
CA_TRANSPORT_PREFIX = "ca://"


def ca_pv(experiment: str | None, device: str, variable: str) -> str:
    """Gateway PV name for ``(experiment, device, variable)``, pinned to CA.

    The one place CA device modules should build PV strings for
    ``epics_signal_r/rw``; suffixes (e.g. ``:SP``) may be appended to the
    returned string — the transport prefix parses identically either way.
    """
    return f"{CA_TRANSPORT_PREFIX}{pv_name(experiment, device, variable)}"
