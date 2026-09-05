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
- The write goes through the device-panel backend's ``put_pv`` seam
  (:class:`~geecs_console.services.device_panel.GatewayDevicePanel`) —
  a :class:`~geecs_bluesky.devices.ca.gateway_put.GatewaySetpointPut`,
  the one blessed put primitive, with ``wire_value`` coercion, even though
  this PV carries no ``:SP`` suffix (it is the control PV itself) — on the
  backend's **persistent** CA event loop.  Never a per-call
  ``asyncio.run``: aioca caches channels per loop and its connection
  callback posts to that loop unguarded, so the restart's own CONN_DOWN
  (the gateway exiting because of this very click) would hit a closed
  loop and print ``RuntimeError: Event loop is closed`` from the CA thread
  on every click (adversarial review, PR #796).
- Import-safe offline: every CA import is lazy, inside the backend.
- Blocking: :func:`request_gateway_restart` is meant for a
  ``BackgroundResult`` daemon thread, never the GUI thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geecs_console.services.device_panel import DevicePanelBackend

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
    experiment: str,
    *,
    backend: DevicePanelBackend,
    timeout: float = RESTART_PUT_TIMEOUT_S,
) -> str:
    """Write ``Restart`` to the experiment's gateway restart PV (blocking).

    Parameters
    ----------
    experiment : str
        The selected experiment.
    backend : DevicePanelBackend
        The console's device-panel backend; its ``put_pv`` runs the write
        on the persistent CA loop (the offline stub refuses with a clear
        message instead).
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
    The put rides the backend's persistent loop, not a per-call
    ``asyncio.run`` (see the module docstring for why that loop's closing
    is not survivable here) and not the shared one-shot loop either
    (``geecs_bluesky.devices.ca.oneshot`` serves *reads* and keeps its
    loop private).
    """
    pv = restart_pv(experiment)
    backend.put_pv(pv, RESTART_VALUE, timeout=timeout, name="cagateway:restart")
    return pv
