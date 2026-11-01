"""Shared GEECS-name → EPICS Channel Access PV-name policy.

This is the naming **contract** between the caproto gateway (which publishes
GEECS device variables as PVs) and the CA-backed ophyd-async devices (which
consume them).  Both sides must agree on it exactly, so it lives here — the one
module both import — rather than being duplicated where it could drift.

GEECS raw names are close to EPICS PV naming, which uses ``:`` as its hierarchy
separator, but LabVIEW tolerates characters CA does not.  Within a single name
*component* only ``[A-Za-z0-9_]`` is kept; every other run of characters maps to
one ``_``, and the result is lowercased.  The most important case is the
**dot**: EPICS reads ``.`` as the record/field separator, so ``Trigger.Source``
must become ``trigger_source`` or clients would parse ``.Source`` as a field.

The mapping is deliberately **lossy** (``Trigger.Source``, ``Trigger Source``,
and ``trigger source`` all collapse to ``trigger_source``); the gateway keeps
the authoritative ``geecs_var ↔ PV`` manifest, so never reverse-engineer a
GEECS name from a PV.

**GEECS-native names are the canonical vocabulary** (configs, schemas, the DB,
every user-facing surface); PV names are a derived wire encoding produced only
here, one-way.  The ``:SP`` setpoint suffix is part of the same contract and is
applied only by :func:`setpoint_pv` — never hand-assemble it.
"""

from __future__ import annotations

import re

# Any run of characters outside the CA-safe component set becomes one underscore.
_INVALID = re.compile(r"[^A-Za-z0-9_]+")


def normalize_component(name: str) -> str:
    """Return a Channel-Access-safe version of a single name component.

    Maps every run of characters outside ``[A-Za-z0-9_]`` (spaces, dots, dashes,
    parentheses, …) to a single underscore, strips leading/trailing
    underscores, and **lowercases** the result — case carries no meaning
    anywhere in GEECS, and case-folding here makes every derived name (PVs,
    event columns) case-insensitive to operator input for free.  Does not
    touch ``:`` — that is the reserved namespace separator applied by
    :func:`pv_name`, never inside a component.

    Parameters
    ----------
    name : str
        Raw GEECS device name or variable name.

    Returns
    -------
    str
        A CA-safe, lowercase name component.
    """
    return _INVALID.sub("_", name.strip()).strip("_").lower()


def pv_name(*parts: str | None) -> str:
    """Join namespace *parts* into a full PV name, normalizing each component.

    Falsy parts (``None``/``""``) are skipped, so an absent experiment prefix
    simply drops out.  The ``:`` separator is reserved and applied here.

    Parameters
    ----------
    *parts : str or None
        Ordered namespace components, e.g. ``experiment, device, variable``.

    Returns
    -------
    str
        The full ``a:b:c`` PV name.
    """
    return ":".join(normalize_component(part) for part in parts if part)


#: The setpoint-PV suffix — part of the naming contract (readback at
#: ``[expt:]dev:var``, setpoint at ``[expt:]dev:var:SP``).
SETPOINT_SUFFIX = ":SP"


def setpoint_pv(readback_pv: str) -> str:
    """Return the setpoint PV name for a readback PV name.

    The one place the ``:SP`` suffix is applied.  Works on bare EPICS names
    and on ``ca://``-schemed signal URIs alike (the suffix parses identically
    either way).

    Parameters
    ----------
    readback_pv : str
        A full readback PV name, e.g. from :func:`pv_name`.

    Returns
    -------
    str
        The corresponding ``…:SP`` setpoint PV name.
    """
    return f"{readback_pv}{SETPOINT_SUFFIX}"


#: The file plugin's PV prefix suffix — areaDetector's ``HDF1:`` under the
#: image variable's PV (GeecsPvaGateway ``file_plugin``, #806).  The plugin's
#: own PV names are the areaDetector ``NDFileHDF5`` suffixes verbatim (mixed
#: case: they are the stock ophyd-async ``NDFileHDF5IO`` contract), appended
#: to this prefix; only the GEECS part is normalized.
HDF_PLUGIN_SUFFIX = ":hdf1:"


def hdf_plugin_prefix(experiment: str, device: str, variable: str) -> str:
    """Return the file plugin's PV prefix for one image variable.

    ``<experiment>:<device>:<variable>:hdf1:`` — the server and the worker's
    ``GeecsHdfIO`` both mint it here, so the two sides cannot drift.

    Parameters
    ----------
    experiment, device, variable : str
        Raw GEECS names; normalized by :func:`pv_name`.
    """
    return f"{pv_name(experiment, device, variable)}{HDF_PLUGIN_SUFFIX}"


#: The PVA gateway's subscription-state PV suffix under an image variable's
#: PV (``<image PV>:connected``, GeecsPvaGateway 0.10.0, #854): the state of
#: that gateway's GEECS subscription for the variable — ``Idle`` (gated off,
#: nothing known) / ``Disconnected`` (watched and unreachable, MAJOR alarm)
#: / ``Connected``.  Per variable because the gateway's subscriptions are;
#: the bare ``<device>:connected`` is the CA gateway's, for its own
#: subscription.  Minted here, like :data:`HDF_PLUGIN_SUFFIX`, so the server
#: and a worker-side reader (the submit preflight's liveness gate) cannot
#: drift.
CONNECTED_SUFFIX = ":connected"


def connected_pv(experiment: str, device: str, variable: str) -> str:
    """Return the PVA gateway's subscription-state PV for one image variable.

    ``<experiment>:<device>:<variable>:connected``.

    Parameters
    ----------
    experiment, device, variable : str
        Raw GEECS names; normalized by :func:`pv_name`.
    """
    return f"{pv_name(experiment, device, variable)}{CONNECTED_SUFFIX}"


#: The CA gateway's per-device connection-state PV component
#: (``<experiment>:<device>:connected``, PV_CONTRACT §1): one GEECS subscription
#: per device, so one PV per device — unlike the PVA gateway's per-variable
#: :data:`CONNECTED_SUFFIX` above.  Minted here so the gateway's clients (the
#: archiver's archive set, a preflight) and the gateway cannot drift.
DEVICE_STATUS_VARIABLE = "connected"


def device_status_pv(experiment: str | None, device: str) -> str:
    """Return the CA gateway's connection-state PV for one device.

    ``<experiment>:<device>:connected`` (the experiment drops out when falsy).

    Parameters
    ----------
    experiment, device : str
        Raw GEECS names; normalized by :func:`pv_name`.
    """
    return pv_name(experiment or "", device, DEVICE_STATUS_VARIABLE)


#: The CA gateway's own device component — the reserved namespace of its
#: self-diagnostics (``<experiment>:cagateway:<name>``, PV_CONTRACT §1).
GATEWAY_DEVICE = "cagateway"
#: The gateway's 5 s liveness counter.
GATEWAY_HEARTBEAT = "heartbeat"
#: The gateway's count of connected GEECS devices.
GATEWAY_DEVICES_CONNECTED = "devices_connected"
#: The gateway's status PVs worth history, which the archiver derives;
#: ``uptime`` and ``version`` are diagnostics, and ``restart`` is a command —
#: never archived.
GATEWAY_ARCHIVED_STATUS: tuple[str, ...] = (
    GATEWAY_HEARTBEAT,
    GATEWAY_DEVICES_CONNECTED,
)


def gateway_status_pv(experiment: str | None, name: str) -> str:
    """Return one of the CA gateway's own PVs.

    ``<experiment>:cagateway:<name>`` (the experiment drops out when falsy).
    The gateway serves its diagnostics through it and the archiver derives
    :data:`GATEWAY_ARCHIVED_STATUS` through it, so the two cannot drift.

    Parameters
    ----------
    experiment : str or None
        Raw GEECS experiment name.
    name : str
        The diagnostic's name, e.g. :data:`GATEWAY_HEARTBEAT`; normalized by
        :func:`pv_name`.
    """
    return pv_name(experiment or "", GATEWAY_DEVICE, name)


def native_device_prefix(experiment: str | None, device: str) -> str:
    """Return the PV prefix a native EPICS IOC serves a registry device under.

    ``<experiment>:<device>:`` — the GEECS normalization of the two names
    and the separator, so a device keeps the prefix it had behind the
    gateway when it migrates to its own IOC (geecs-epics' chain contract).
    The record names under it are the template's own, verbatim (mixed
    case: ``cam1:AcquireTime_RBV``, a motor axis's ``x.RBV``).  The one
    home of the rule: the scan worker, the panels, GeecsCanopy and the
    archiver all mint it here.

    Parameters
    ----------
    experiment : str or None
        Raw GEECS experiment name (dropped when falsy).
    device : str
        Raw GEECS device name, as the registry lists it.
    """
    return pv_name(experiment or "", device) + ":"


#: devIocStats' liveness counter (``iocAdminSoft.db``), one per IOC.
IOC_HEARTBEAT = "HEARTBEAT"


def ioc_heartbeat_pv(ioc: str) -> str:
    """Return a native IOC's devIocStats heartbeat PV: ``<IOC>:HEARTBEAT``.

    geecs-epics instantiates ``devIocStats.iocAdminSoft`` with
    ``IOC: "{{ ioc_name | upper }}"``, the IOC's name (its compose
    ``IOC_NAME``) upper-cased; the per-IOC analogue of the gateway's
    ``cagateway:heartbeat``.  Not under the experiment's prefix: an IOC may
    serve several devices.

    Parameters
    ----------
    ioc : str
        The IOC's name as its host's compose file sets ``IOC_NAME``
        (``adtest-cam1``).
    """
    return f"{ioc.strip().upper()}:{IOC_HEARTBEAT}"
