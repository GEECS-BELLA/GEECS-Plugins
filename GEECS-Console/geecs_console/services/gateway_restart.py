"""The CA gateway's restart verb, from the console (#773).

``{experiment}:cagateway:restart`` is the gateway's one *client-writable*
PV (``GeecsCAGateway/PV_CONTRACT.md`` § ``cagateway:restart``): an enum
``["Idle", "Restart"]``.  Writing ``Restart`` makes the gateway shut down
cleanly and exit with code 86, which the shipped systemd unit turns into
a relaunch — the operator fix for the frozen-readback symptom (a stale
GEECS TCP subscription after a device-app restart: half-open socket, the
supervisor only reconnects on socket *close*; live incident 2026-08-17)
and the DB-resync mechanism after a device / get-list edit, since the
served set rebuilds from the GEECS database at startup.  Outside that
supervisor (a foreground run) the process simply exits and **stays down**
— the console's confirmation text says so.

Rules this module keeps:

- The PV name comes only from the naming contract (``ca_pv`` /
  ``bare_pv`` — never hand-built; issue #490).
- The write is a :class:`~geecs_bluesky.devices.ca.gateway_put.GatewaySetpointPut`
  — the one blessed put primitive — with ``wire_value`` coercion, even
  though this PV carries no ``:SP`` suffix (it is the control PV itself).
- Import-safe offline: every CA import is lazy, inside the functions.
- Blocking: :func:`request_gateway_restart` is meant for a
  ``BackgroundResult`` daemon thread, never the GUI thread.
"""

from __future__ import annotations

import asyncio

#: The enum label that requests the restart (index 1; ``Idle``/0 is a no-op).
RESTART_VALUE = "Restart"

#: Put budget.  The gateway completes the CA write as soon as it has taken
#: the restart request — well under a second on a live gateway; a longer
#: wait means it is not answering at all.
RESTART_PUT_TIMEOUT_S = 5.0


def restart_pv(experiment: str) -> str:
    """The bare ``cagateway:restart`` PV name for *experiment*.

    Parameters
    ----------
    experiment : str
        The selected experiment (the PV prefix); must be non-empty.

    Returns
    -------
    str
        e.g. ``"undulator:cagateway:restart"`` — through the naming
        contract (lowercased, ``ca://`` stripped), never string-built.

    Raises
    ------
    ValueError
        With no experiment there is no prefixed PV to address.
    """
    if not experiment:
        raise ValueError(
            "no experiment selected — the restart PV is experiment-prefixed"
        )
    from geecs_bluesky.devices.ca._pv import ca_pv
    from geecs_bluesky.devices.ca.gateway_put import bare_pv

    return bare_pv(ca_pv(experiment, "CAGateway", "RESTART"))


def request_gateway_restart(
    experiment: str, *, timeout: float = RESTART_PUT_TIMEOUT_S
) -> str:
    """Write ``Restart`` to the experiment's gateway restart PV (blocking).

    Parameters
    ----------
    experiment : str
        The selected experiment.
    timeout : float, optional
        Put budget in seconds (:data:`RESTART_PUT_TIMEOUT_S`).

    Returns
    -------
    str
        The PV written, for the status-bar line.

    Raises
    ------
    Exception
        Whatever the put raises (timeout — the gateway is not answering —,
        no CA, a rejected value); the window renders it as a failure.

    Notes
    -----
    The put runs on a fresh ``asyncio.run`` loop rather than the shared
    one-shot loop (``geecs_bluesky.devices.ca.oneshot`` serves *reads*
    and keeps its loop private): aioca caches channels per loop, so each
    call strands one channel for the process lifetime.  Accepted for an
    incident-rate verb — one click per frozen gateway, not a poll.
    """
    pv = restart_pv(experiment)
    from geecs_bluesky.devices.ca.gateway_put import GatewaySetpointPut, wire_value

    put = GatewaySetpointPut(
        setpoint_pv=pv, coerce=wire_value, timeout=timeout, name="cagateway:restart"
    )
    asyncio.run(put.put(RESTART_VALUE))
    return pv
