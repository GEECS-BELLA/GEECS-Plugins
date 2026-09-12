"""DB-driven configuration: which cameras this host serves, and their PV names.

The served set is self-scoping: enumerate the experiment's enabled devices and
keep those whose GEECS endpoint IP belongs to this machine and that expose at
least one image-typed variable. The GEECS DB is the source of truth — there is
no per-host config file.
"""

from __future__ import annotations

import logging
import socket

from pydantic import BaseModel, Field

from geecs_core.db.scalar_policy import GeecsDbScalarPolicy
from geecs_core.db.variable_types import (  # noqa: F401 - image_variables re-exported
    effective_vartype,
    image_variables,
)
from geecs_core.pv_naming import normalize_component, pv_name
from geecs_core.transport.udp_client import detect_local_ip

logger = logging.getLogger(__name__)


def local_ip_addresses(probe_target: str | None = None) -> set[str]:
    """Return this machine's IPv4 addresses (hostname lookup + route probe)."""
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass
    if probe_target:
        # The address the OS would source from toward the lab (no packet sent).
        addresses.add(detect_local_ip(probe_target))
        addresses.discard("")
    return addresses


def instance_pv_prefix(experiment: str, host: str) -> str:
    """Prefix of one instance's identity PVs (``version``/``heartbeat``/``restart``).

    The identity component is the served host (an IP, dots normalised to
    underscores — PV_CONTRACT). The one composition shared by the server
    and the fleet tooling, so the probe and the screen can never drift from
    what an instance actually serves.
    """
    return pv_name(experiment, "pvagateway", normalize_component(host))


#: The timestamp ladder the gateway subscribes for every image variable
#: (``server._TIMESTAMP_VARS``): the frame's own stamp attributes already
#: carry it, so it is never repeated as a scalar attribute.
TIMESTAMP_VARIABLES = frozenset({"acq_timestamp", "systimestamp"})
#: Effective DB types the plugin can write as a ``DOUBLE`` per-frame
#: attribute: numbers, and enums whose wire value is numeric (a text enum
#: value lands as ``NaN`` — the plugin writes what it can represent, never
#: a second dtype).  ``string``/``path``/``image``/``1darray`` are not
#: per-frame scalars.
SCALAR_ATTRIBUTE_VARTYPES = frozenset({"numeric", "choice"})


def scalar_attribute_variables(rows: list[dict], subscribed: list[str]) -> list[str]:
    """The device's subscribed scalars the file plugin writes per frame.

    The subscribed (``get='yes'``) list in DB order — the same list the
    worker's namespace makes a device's event columns from, so a gated
    row and a strict row carry the same columns for that device
    (``Planning/native_bluesky/08_gated_batch.md`` §4.4) — restricted to
    the variables the plugin can carry as a ``DOUBLE`` attribute
    (:data:`SCALAR_ATTRIBUTE_VARTYPES`) and minus the timestamp ladder
    (:data:`TIMESTAMP_VARIABLES`, already the frame's stamp attributes).
    A subscribed name with no metadata row has no type and is skipped.
    """
    types = {
        str(row["name"]): effective_vartype(row.get("variabletype"), row.get("choices"))
        for row in rows
    }
    out: list[str] = []
    for name in subscribed:
        if name in TIMESTAMP_VARIABLES or name in out:
            continue
        if types.get(name) in SCALAR_ATTRIBUTE_VARTYPES:
            out.append(name)
    return out


class CameraSpec(BaseModel):
    """One GEECS camera device: endpoint, image variables, PV names."""

    device: str
    host: str
    port: int
    experiment: str
    image_variables: list[str] = Field(default_factory=lambda: ["image"])
    #: The subscribed scalars the file plugin writes beside every frame
    #: (:func:`scalar_attribute_variables`); joins the image variable's one
    #: TCP subscription.  Empty = frames and stamps only.
    scalar_variables: list[str] = Field(default_factory=list)

    def pv_name_for(self, variable: str) -> str:
        """Full PV name for one image variable, minted by the shared contract."""
        return pv_name(self.experiment, self.device, variable)


class PvaGatewayConfig(BaseModel):
    """The set of cameras one gateway instance serves."""

    experiment: str
    cameras: list[CameraSpec] = Field(default_factory=list)

    @classmethod
    def from_geecs_experiment(
        cls,
        experiment: str,
        *,
        host: str | None = None,
        devices: list[str] | None = None,
        enabled_only: bool = True,
    ) -> "PvaGatewayConfig":
        """Build the served set from the GEECS database (two batched queries).

        Parameters
        ----------
        experiment : str
            GEECS experiment name; also the PV namespace prefix.
        host : str, optional
            Endpoint IP to scope to. Default: this machine's own addresses.
        devices : list of str, optional
            Restrict to these device names (after host scoping).
        enabled_only : bool
            Skip devices not enabled in the experiment (default true).
        """
        from geecs_core.db.geecs_db import GeecsDb

        endpoints = GeecsDb.get_experiment_devices(
            experiment, enabled_only=enabled_only
        )
        var_map = GeecsDb.get_experiment_device_variables(
            experiment, enabled_only=enabled_only
        )
        # The per-frame scalar attributes: the same rule the worker builds a
        # device's row from, from its one home.  Degrades to empty with a
        # warning on a DB blip (the roster queries above already succeeded).
        subscribed = GeecsDbScalarPolicy(
            experiment, enabled_only=enabled_only, db=GeecsDb
        ).subscribed_by_device()
        if host:
            hosts = {host}
        else:
            # Probe toward any device endpoint so the lab-facing interface's
            # address is included even when hostname lookup misses it.
            any_ip = next(iter(endpoints.values()), ("", 0))[0]
            hosts = local_ip_addresses(probe_target=any_ip or None)

        cameras: list[CameraSpec] = []
        for device, (ip, port) in sorted(endpoints.items()):
            if ip not in hosts:
                continue
            if devices is not None and device not in devices:
                continue
            image_vars = image_variables(var_map.get(device, []))
            if not image_vars:
                continue  # not a camera (e.g. a timing box on the same host)
            cameras.append(
                CameraSpec(
                    device=device,
                    host=ip,
                    port=port,
                    experiment=experiment,
                    image_variables=image_vars,
                    scalar_variables=scalar_attribute_variables(
                        var_map.get(device, []), subscribed.get(device, [])
                    ),
                )
            )
        if devices:
            for name in set(devices) - {c.device for c in cameras}:
                logger.warning(
                    "requested device %s not served: not on host(s) %s, not "
                    "enabled, or has no image variables",
                    name,
                    sorted(hosts),
                )
        return cls(experiment=experiment, cameras=cameras)
